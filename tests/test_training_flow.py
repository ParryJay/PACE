"""Exercise the real training loop with a tiny model and in-memory train data.

No dataset, pretrained download, metric calculation or visualization is needed.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import Dataset

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
import inference
import trainer


class _InterruptedTraining(RuntimeError):
    pass


class _TinyPACE(nn.Module):
    """One parameter and the buffers consumed by the real training loop."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))
        self.audit = nn.Module()
        self.audit.register_buffer("mu", torch.tensor(0.2))
        self.audit.register_buffer("s0_sq", torch.tensor(0.1))
        self.register_buffer("p_les", torch.tensor(0.3))

    def forward(self, nbi, wli):
        del wli
        scalar = self.weight.expand(nbi.shape[0])
        audit = {"t": scalar.sigmoid(), "fallback": torch.zeros_like(scalar, dtype=torch.bool)}
        return {"rho": scalar.sigmoid(), "alpha": scalar.sigmoid(),
                "audit_nbi": audit, "audit_wli": audit, "objective": (self.weight - 1).square()}


class _TrainPairs(Dataset):
    def __init__(self):
        self.pairs = [{"nbi_img": "n.png", "wli_img": "w.png", "nbi_lbl": "n.png", "wli_lbl": "w.png"}]
        self.train_pairs = self.pairs

    def __len__(self):
        return 1

    def __getitem__(self, index):
        del index
        return {"nbi_img": torch.zeros(3, 2, 2, dtype=torch.uint8),
                "wli_img": torch.zeros(3, 2, 2, dtype=torch.uint8),
                "nbi_lbl": torch.zeros(2, 2, dtype=torch.uint8),
                "wli_lbl": torch.zeros(2, 2, dtype=torch.uint8)}

    def read_label(self, pair, modality):
        del pair, modality
        import numpy as np
        return np.zeros((2, 2), dtype=np.uint8)


def _loss(self, outputs, mask_nbi, mask_wli, epoch_progress=None):
    del self, mask_nbi, mask_wli, epoch_progress
    value = outputs["objective"]
    return value, {"total": value.detach()}


def _arguments(output, **overrides):
    values = {
        "dataset": "FAHSYSU", "dataset_root": str(PACKAGE_ROOT / ".in_memory_training_data"),
        "train_manifest": None, "test_manifest": "this_test_manifest_must_not_be_opened_during_training.json",
        "output_dir": str(output), "seed": 3407, "epochs": 100, "img_size": 512,
        "num_workers": 0, "persistent_workers": False, "prefetch_factor": 2,
        "batch_train": 1, "use_amp": False, "amp_dtype": "float16",
        "amp_init_scale": 65536., "amp_growth_interval": 2000,
        "use_ema": True, "ema_decay": .999, "ema_update_after_steps": 1, "ema_eval": True,
        "resume_checkpoint": None, "resume_ckpt": None, "weights_source": "auto", "weights_strict": True,
        "smoke_steps": 0, "checkpoint_interval": 0, "deterministic": True,
        "use_synthetic_quality_degradation": False, "use_spatial_augmentation": False,
        "lr": 1e-3, "lr_backbone": 1e-4, "weight_decay": 1e-3,
        "backbone_layer_decay": .8, "fusion_lr_multiplier": 1., "finetune_mode": "all",
        "freeze_backbone_epochs": 0, "lr_scheduler": "warmup_cosine", "warmup_epochs": 0.,
        "fusion_warmup_epochs": 0., "min_lr_ratio": .01, "lr_decay": .95,
        "max_grad_norm": 1., "ddp_find_unused_parameters": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TrainingFlowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix=".training-flow-", dir=PACKAGE_ROOT)
        self.base = Path(self.directory.name)
        self.events = []
        self.inferred = []
        self.model_builds = []
        self.saved_epochs = []

    def tearDown(self):
        self.directory.cleanup()

    def _dataset(self, dataset, root, manifest, *, split, img_size):
        del dataset, root, manifest, img_size
        self.assertEqual(split, "train", "Trainer opened the test split during training")
        self.events.append("open_train")
        return _TrainPairs()

    def _model(self, args, *, pretrained=None):
        del args
        self.model_builds.append(pretrained)
        return _TinyPACE()

    def _inference(self, model, args, device, **kwargs):
        del device
        path = Path(kwargs["checkpoint_path"])
        checkpoint = inference.load_final_checkpoint(path, args.dataset)
        self.assertEqual(checkpoint["epoch"], 100)
        self.assertTrue(checkpoint["final"])
        self.assertFalse(model.training)
        state, selected = inference.select_checkpoint_state(
            checkpoint, args.weights_source, ema_eval=args.ema_eval,
        )
        self.assertEqual(kwargs["weights_source"], selected)
        self.assertTrue(torch.equal(model.state_dict()["weight"].cpu(), state["weight"].cpu()))
        self.assertEqual(self.saved_epochs[-1], (100, True))
        # This represents the first test-manifest read in the pure inference
        # module, reached only after a complete final checkpoint was committed.
        self.events.append("open_test")
        self.inferred.append(kwargs)
        return {"pair_count": 1, "epoch": 100}

    def _run(self, args, *, interrupt_epoch=None):
        original_save = trainer.Trainer._save_checkpoint

        def save(instance, path, epoch, step, optimizer, scheduler, scaler, ema, config, generator, *, final=False):
            original_save(instance, path, epoch, step, optimizer, scheduler, scaler, ema, config, generator, final=final)
            self.saved_epochs.append((epoch, final))
            if not final and epoch == interrupt_epoch and Path(path).name == "PACE_last.pth":
                raise _InterruptedTraining("simulate an interruption at an epoch boundary")

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(trainer, "PairedDataset", side_effect=self._dataset))
            stack.enter_context(patch.object(trainer, "build_model", side_effect=self._model))
            stack.enter_context(patch.object(trainer.Trainer, "compute_loss", _loss))
            stack.enter_context(patch.object(trainer.Trainer, "_save_checkpoint", save))
            stack.enter_context(patch.object(inference, "run_inference", side_effect=self._inference))
            # Force CPU even on a GPU machine; the production loop is unchanged.
            stack.enter_context(patch.object(torch.cuda, "is_available", return_value=False))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            instance = trainer.Trainer(0, 0, 1, args)
            return instance(args, pin_memory=False)

    def test_only_epoch_100_final_weights_are_tested(self):
        output = self.base / "complete"
        result = self._run(_arguments(output))
        self.assertEqual(result["epoch"], 100)
        self.assertEqual(self.events, ["open_train", "open_test"])
        self.assertEqual(len(self.inferred), 1)
        self.assertEqual(self.saved_epochs, [(epoch, False) for epoch in range(1, 101)] + [(100, True)])
        self.assertTrue((output / "PACE_last.pth").is_file())
        self.assertTrue((output / "PACE_final.pth").is_file())
        self.assertFalse(any("best" in p.name for p in output.iterdir()))
        self.assertEqual(self.inferred[0]["weights_source"], "ema")

    def test_explicit_online_weights_are_used_after_epoch_100(self):
        self._run(_arguments(self.base / "online", weights_source="online"))
        self.assertEqual(len(self.inferred), 1)
        self.assertEqual(self.inferred[0]["weights_source"], "online")

    def test_smoke_training_never_opens_test_or_saves_final(self):
        output = self.base / "smoke"
        result = self._run(_arguments(output, smoke_steps=1))
        self.assertEqual(result["steps"], 1)
        self.assertEqual(self.events, ["open_train"])
        self.assertEqual(self.inferred, [])
        self.assertEqual(self.saved_epochs, [])
        self.assertFalse((output / "PACE_final.pth").exists())

    def test_resume_at_epoch_50_or_99_has_no_test_baseline(self):
        for epoch in (50, 99):
            with self.subTest(epoch=epoch):
                output = self.base / f"interrupted_{epoch}"
                with self.assertRaises(_InterruptedTraining):
                    self._run(_arguments(output), interrupt_epoch=epoch)
                self.assertEqual(self.inferred, [])
                snapshot = output / "PACE_last.pth"
                self.assertFalse((output / "PACE_final.pth").exists())
                self._run(_arguments(self.base / f"resumed_{epoch}", resume_checkpoint=str(snapshot)))
                self.assertEqual(len(self.inferred), 1)
                self.assertEqual(self.events[-2:], ["open_train", "open_test"])
                self.assertIs(self.model_builds[-1], False, "Resume must not download pretrained weights")
                final = inference.load_final_checkpoint(self.base / f"resumed_{epoch}" / "PACE_final.pth", "FAHSYSU")
                self.assertEqual(final["global_step"], 100)
                self.events.clear()
                self.inferred.clear()
                self.saved_epochs.clear()

    def test_resume_with_another_dataset_is_rejected_before_testing(self):
        output = self.base / "wrong_dataset_source"
        with self.assertRaises(_InterruptedTraining):
            self._run(_arguments(output), interrupt_epoch=1)
        with self.assertRaisesRegex(ValueError, "dataset"):
            self._run(_arguments(self.base / "wrong_dataset", dataset="PICCOLO",
                                 resume_checkpoint=str(output / "PACE_last.pth")))
        self.assertEqual(self.inferred, [])
        self.assertNotIn("open_test", self.events)

    def test_resume_rejects_changed_training_semantics_without_shape_changes(self):
        output = self.base / "semantic_source"
        with self.assertRaises(_InterruptedTraining):
            self._run(_arguments(output), interrupt_epoch=1)
        snapshot = output / "PACE_last.pth"
        saved = torch.load(snapshot, map_location="cpu", weights_only=False)
        self.assertEqual(saved["train_config"]["lambda_seg"], 1.)
        self.assertEqual(saved["train_config"]["audit_tau"], .5)
        for name, value in (
            ("lambda_seg", 2.), ("use_synthetic_quality_degradation", True),
            ("audit_tau", .7), ("use_ema", False), ("use_amp", True),
            ("seed", 1), ("quality_apply_probability", .4),
            ("decision_fusion", "off"),
        ):
            with self.subTest(option=name):
                with self.assertRaisesRegex(ValueError, "training configuration.*" + name):
                    self._run(_arguments(self.base / ("changed_" + name),
                                         resume_checkpoint=str(snapshot), **{name: value}))
                self.assertEqual(self.inferred, [])
                self.assertNotIn("open_test", self.events)

    def test_resume_allows_new_output_manifest_location_and_inference_options(self):
        output = self.base / "portable_source"
        with self.assertRaises(_InterruptedTraining):
            self._run(_arguments(output), interrupt_epoch=99)
        # The factory provides the same normalized pair contents under a new
        # manifest location. Pair-content hashing, rather than path identity,
        # protects the training split.
        self._run(_arguments(
            self.base / "portable_resumed", resume_checkpoint=str(output / "PACE_last.pth"),
            train_manifest="relocated_but_identical_pairs.json",
            test_manifest="different_final_inference_pairs.json", encoder_pretrained=False,
            batch_test=1, test_num_workers=0, save_predictions=True, ema_eval=False,
            prefetch_factor=4, master_port=32100, local_rank=3,
        ))
        self.assertEqual(len(self.inferred), 1)
        self.assertEqual(self.inferred[0]["weights_source"], "online")
        self.assertEqual(self.events, ["open_train", "open_train", "open_test"])


if __name__ == "__main__":
    unittest.main()
