"""Exercise label-free final inference with real loaders and a tiny model."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
import inference
from config import parse_args


class _TinyPredictor(nn.Module):
    def __init__(self, bias=0.25):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(float(bias)))
        self.batch_sizes = []

    def forward(self, nbi, wli):
        if self.training or not torch.is_inference_mode_enabled():
            raise AssertionError("Pure inference must use eval mode and torch.inference_mode.")
        if nbi.shape != wli.shape or nbi.shape[1:] != (3, 512, 512):
            raise AssertionError("Expected paired RGB 512x512 inputs.")
        self.batch_sizes.append(nbi.shape[0])
        logits = torch.stack((-self.bias, self.bias)).view(1, 2, 1, 1)
        logits = logits.expand(nbi.shape[0], 2, *nbi.shape[-2:])
        return {"logits_final_nbi": logits, "logits_final_wli": -logits}


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pace_inference_", dir="/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.device = torch.device("cpu")
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(previous_threads, 2))
        self.addCleanup(torch.set_num_threads, previous_threads)

    def _arguments(self, dataset, *, save=False):
        data_root = self.root / dataset
        for modality in ("nbi", "wli"):
            directory = data_root / modality / "img" if dataset == "FAHSYSU" else data_root / "img" / modality.upper()
            directory.mkdir(parents=True, exist_ok=True)
            self.assertTrue(cv2.imwrite(str(directory / f"{modality}.png"), np.full((9, 11, 3), 127, dtype=np.uint8)))
        pairs = [{"id": str(index), "nbi_img": "nbi.png", "wli_img": "wli.png"} for index in range(3)]
        # There is no training manifest or label directory in this fixture.
        manifest = data_root / ("pairs_test.json" if dataset == "FAHSYSU" else "test.jsonl")
        manifest.write_text(json.dumps(pairs) if dataset == "FAHSYSU" else "\n".join(map(json.dumps, pairs)) + "\n", encoding="utf-8")
        return parse_args([
            "--dataset", dataset, "--dataset_root", str(data_root),
            "--devices", "cpu", "--dist_backend", "gloo",
            "--output_dir", str(self.root / f"output_{dataset}_{save}"),
            "--num_workers", "0", "--test_num_workers", "0",
            "--batch_test", "2", "--pin_memory", "false", "--use_amp", "false",
            "--save_predictions", str(save).lower(),
        ])

    def _checkpoint(self, dataset="FAHSYSU", **overrides):
        checkpoint = {
            "format": "PACE_opensource_v1", "epoch": 100, "completed_epochs": 100,
            "final": True, "dataset": dataset,
            "args": {"d_model": 12, "num_tokens": 4, "encoder_name": "saved-backbone"},
            "model": _TinyPredictor().state_dict(), "ema": None,
        }
        checkpoint.update(overrides)
        path = self.root / "PACE_final.pth"
        torch.save(checkpoint, path)
        return path, checkpoint

    def test_no_save_runs_label_free_inference_without_prediction_arrays(self):
        for dataset in ("FAHSYSU", "PICCOLO"):
            with self.subTest(dataset=dataset):
                args = self._arguments(dataset)
                model = _TinyPredictor()
                with contextlib.redirect_stdout(io.StringIO()):
                    result = inference.run_inference(model, args, self.device)
                self.assertEqual(result["pair_count"], 3)
                self.assertEqual(result["completed_epochs"], 100)
                self.assertFalse(result["predictions_saved"])
                self.assertEqual(model.batch_sizes, [2, 1])
                output = Path(args.output_dir)
                self.assertFalse((output / "predictions").exists())
                self.assertEqual([path.name for path in output.iterdir()], ["inference_run.json"])
                self.assertEqual(json.loads((output / "inference_run.json").read_text()), result)

    def test_save_writes_numeric_npz_predictions_for_each_pair(self):
        for dataset in ("FAHSYSU", "PICCOLO"):
            with self.subTest(dataset=dataset):
                args = self._arguments(dataset, save=True)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = inference.run_inference(_TinyPredictor(), args, self.device)
                self.assertTrue(result["predictions_saved"])
                files = sorted((Path(args.output_dir) / "predictions").iterdir())
                self.assertEqual([path.name for path in files], [f"{index:08d}.npz" for index in range(3)])
                nbi_probability = float(torch.sigmoid(torch.tensor(0.5)))
                for index, path in enumerate(files):
                    with np.load(path, allow_pickle=False) as prediction:
                        self.assertEqual(int(prediction["pair_index"]), index)
                        self.assertEqual(str(prediction["nbi_img"]), "nbi.png")
                        self.assertEqual(prediction["nbi_probability"].shape, (512, 512))
                        self.assertEqual(prediction["wli_mask"].dtype, np.uint8)
                        np.testing.assert_allclose(prediction["nbi_probability"], nbi_probability)
                        np.testing.assert_allclose(prediction["wli_probability"], 1 - nbi_probability, atol=1e-7)
                        self.assertTrue(np.all(prediction["nbi_mask"] == 1))
                        self.assertTrue(np.all(prediction["wli_mask"] == 0))
                self.assertFalse(list(Path(args.output_dir).rglob("*.png")))
                self.assertFalse(list(Path(args.output_dir).rglob("*.csv")))

    def test_only_matching_completed_final_checkpoints_are_accepted(self):
        path, checkpoint = self._checkpoint()
        loaded = inference.load_final_checkpoint(path, "FAHSYSU")
        self.assertEqual(loaded["epoch"], 100)
        invalid = (
            {"epoch": 99}, {"completed_epochs": 99}, {"final": False},
            {"format": "PACE_v1"}, {"args": None}, {"model": None, "final": False},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                torch.save(dict(checkpoint, **changes), path)
                with self.assertRaises(ValueError):
                    inference.load_final_checkpoint(path, "FAHSYSU")
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "differs"):
            inference.load_final_checkpoint(path, "PICCOLO")

    def test_inference_rejects_epoch_before_100_before_reading_data(self):
        args = self._arguments("FAHSYSU")
        args.dataset_root = str(self.root / "must_not_be_opened")
        with self.assertRaisesRegex(ValueError, "epoch 100"):
            inference.run_inference(_TinyPredictor(), args, self.device, epoch=99)

    def test_weight_source_uses_only_available_initialized_ema(self):
        _, checkpoint = self._checkpoint()
        online = checkpoint["model"]
        ema = _TinyPredictor(bias=0.75).state_dict()
        checkpoint["ema"] = {"initialized": True, "model": ema}
        state, source = inference.select_checkpoint_state(checkpoint, "auto", ema_eval=True)
        self.assertIs(state, ema)
        self.assertEqual(source, "ema")
        for requested, enabled in (("online", True), ("auto", False)):
            state, source = inference.select_checkpoint_state(checkpoint, requested, ema_eval=enabled)
            self.assertIs(state, online)
            self.assertEqual(source, "online")
        state, source = inference.select_checkpoint_state(checkpoint, "ema", ema_eval=False)
        self.assertIs(state, ema)
        self.assertEqual(source, "ema")
        for unavailable in (None, {"initialized": False, "model": ema}, {"initialized": True}):
            with self.subTest(ema=unavailable):
                checkpoint["ema"] = unavailable
                self.assertEqual(inference.select_checkpoint_state(checkpoint, "auto")[1], "online")
                with self.assertRaisesRegex(ValueError, "initialized EMA"):
                    inference.select_checkpoint_state(checkpoint, "ema")
        with self.assertRaises(ValueError):
            inference.select_checkpoint_state(checkpoint, "invalid")

    def test_standalone_inference_restores_saved_architecture_without_pretraining(self):
        args = self._arguments("FAHSYSU")
        path, checkpoint = self._checkpoint()
        args.resume_ckpt = str(path)
        # These deliberately differ from the architecture saved with training.
        args.d_model, args.num_tokens, args.encoder_name = 256, 128, "cli-backbone"
        restored = _TinyPredictor(bias=-9)
        with patch.object(inference, "build_model", return_value=restored) as build:
            with contextlib.redirect_stdout(io.StringIO()):
                result = inference.run_checkpoint_inference(args, self.device)
        build.assert_called_once_with(checkpoint["args"], pretrained=False)
        self.assertEqual(float(restored.bias), 0.25)
        self.assertEqual(restored.batch_sizes, [2, 1])
        self.assertEqual(result["checkpoint"], str(path))
        self.assertEqual(result["weights_source"], "online")

    def test_standalone_auto_preserves_training_weight_source(self):
        args = self._arguments("FAHSYSU")
        _, checkpoint = self._checkpoint()
        checkpoint["args"]["ema_eval"] = False
        checkpoint["ema"] = {"initialized": True, "model": _TinyPredictor(bias=0.75).state_dict()}
        path = self.root / "saved_online_final.pth"
        torch.save(checkpoint, path)
        args.resume_ckpt = str(path)
        args.ema_eval = True  # A CLI default must not change the recorded policy.
        restored = _TinyPredictor()
        with patch.object(inference, "build_model", return_value=restored):
            with contextlib.redirect_stdout(io.StringIO()):
                result = inference.run_checkpoint_inference(args, self.device)
        self.assertEqual(result["weights_source"], "online")
        self.assertEqual(float(restored.bias), 0.25)

    def test_rank_shards_cover_uneven_test_split_once(self):
        args = self._arguments("FAHSYSU", save=True)
        # Exercise each rank's real loader without requiring local TCP sockets.
        # The reduction is supplied with the known global count by the test.
        def total_count(count, **kwargs):
            count.fill_(3)

        with patch.object(inference.dist, "is_initialized", return_value=True), \
             patch.object(inference.dist, "get_world_size", return_value=2), \
             patch.object(inference.dist, "get_rank", return_value=0), \
             patch.object(inference.dist, "all_reduce", side_effect=total_count):
            with contextlib.redirect_stdout(io.StringIO()):
                first, second = _TinyPredictor(), _TinyPredictor()
                inference.run_inference(first, args, self.device, rank=0, world_size=2)
                inference.run_inference(second, args, self.device, rank=1, world_size=2)
        self.assertEqual(first.batch_sizes, [2])
        self.assertEqual(second.batch_sizes, [1])
        files = sorted((Path(args.output_dir) / "predictions").glob("*.npz"))
        self.assertEqual([path.name for path in files], [f"{index:08d}.npz" for index in range(3)])


if __name__ == "__main__":
    unittest.main()
