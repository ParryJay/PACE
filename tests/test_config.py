"""CLI protocol and launch checks that require only the standard library."""
from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import config
import main


class ConfigurationTests(unittest.TestCase):
    def arguments(self, *options, dataset="FAHSYSU"):
        return config.parse_args([
            "--dataset", dataset, "--dataset_root", str(PACKAGE_ROOT / "example_data"),
            *options,
        ])

    def assert_invalid(self, *options):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.arguments(*options)
        self.assertEqual(error.exception.code, 2)

    def test_effective_reference_training_defaults_are_preserved(self):
        # These are the effective PACE defaults after the old entry points'
        # normalization and PACE overrides, including auxiliary loss settings.
        expected = {
            "seed": 3407, "epochs": 100, "img_size": 512,
            "batch_train": 6, "batch_test": 8, "num_workers": 6,
            "test_num_workers": 1, "prefetch_factor": 2,
            "persistent_workers": False, "pin_memory": True,
            "lr": 1e-4, "lr_backbone": 1e-5, "weight_decay": 1e-3,
            "backbone_layer_decay": .85, "fusion_lr_multiplier": 1.5,
            "max_grad_norm": 1., "lr_scheduler": "warmup_cosine",
            "warmup_epochs": 5., "fusion_warmup_epochs": 1.,
            "min_lr_ratio": .05, "lr_decay": .985,
            "freeze_backbone_epochs": 0, "finetune_mode": "all",
            "use_amp": True, "amp_dtype": "bfloat16", "amp_init_scale": 4096.,
            "amp_min_scale": 1., "amp_growth_interval": 2000,
            "amp_max_consecutive_overflows": 16,
            "use_ema": True, "ema_decay": .999,
            "ema_update_after_steps": 100, "ema_eval": True,
            "encoder_name": "convnextv2_base", "encoder_pretrained": True,
            "freeze_backbone": False, "d_model": 256, "token_dim": None,
            "num_tokens": 128, "preliminary_hidden_channels": 128,
            "fusion_num_heads": 4, "rho_min": .2, "rho_max": .85,
            "alpha_min": .05, "alpha_max": .35, "transport_epsilon": .05,
            "transport_fgw_iterations": 5, "transport_sinkhorn_iterations": 50,
            "audit_tau": .5, "node_spatial_strength": 8.,
            "lambda_f": 1., "lambda_s": .5, "lambda_b": .5,
            "theta_p": 1., "theta_q": 1., "theta_b": 1., "theta_d": 1.,
            "match_use_abs_pos": False, "opinion_grad": "detach",
            "l2_kv": "self", "decision_fusion": "add", "force_t": None,
            "seg_loss_mode": "foreground", "ce_loss_weight": .5,
            "dice_loss_weight": .5, "foreground_ce_weight": 1.5,
            "foreground_tversky_alpha": .4, "foreground_tversky_beta": .6,
            "lambda_seg": 1., "lambda_dec": .5, "lambda_q": .2,
            "lambda_boundary": .2, "lambda_quality": .1, "lambda_e": .2,
            "lambda_match": 0., "aux_warmup_epochs": 5.,
            "max_auto_pos_weight": 20., "boundary_pos_weight": None,
            "use_spatial_augmentation": True,
            "use_synthetic_quality_degradation": True,
            "quality_degradation_pair_policy": "single_modality",
            "quality_nbi_degrade_probability": .5,
            "quality_curriculum_epochs": 10., "quality_apply_probability": .8,
            "quality_min_severity": .2, "quality_max_severity": .7,
            "quality_local_probability": .8, "quality_max_operations": 2,
            "quality_seed": 3407, "weights_source": "auto", "weights_strict": True,
            "ddp_find_unused_parameters": False,
        }
        for dataset in ("FAHSYSU", "PICCOLO"):
            actual = self.arguments(dataset=dataset)
            for name, value in expected.items():
                with self.subTest(dataset=dataset, option=name):
                    self.assertEqual(getattr(actual, name), value)
            for operation in config.QUALITY_OPERATIONS:
                self.assertTrue(getattr(actual, "quality_" + operation))
                self.assertEqual(getattr(actual, "quality_" + operation + "_probability"), 1.)

    def test_fixed_training_protocol_rejects_early_or_late_test_schedules(self):
        for options in (("--epochs", "99"), ("--epochs", "101"),
                        ("--test_step", "5"), ("--test_vis",),
                        ("--best_metric", "mIoU"), ("--eval_aux_shuffle",)):
            with self.subTest(options=options):
                self.assert_invalid(*options)

    def test_standalone_inference_requires_final_weight_argument(self):
        self.assert_invalid("--inference_only")
        self.assert_invalid("--inference_only", "--resume", "last.pth")
        self.assert_invalid("--inference_only", "--weights", "final.pth", "--smoke_steps", "1")
        for alias in ("--inference_only", "--inference-only", "--eval_only", "--test-only"):
            args = self.arguments(alias, "--weights", "final.pth")
            self.assertTrue(args.inference_only)
            self.assertEqual(args.resume_ckpt, "final.pth")

    def test_boolean_aliases_and_weight_loading_modes(self):
        args = self.arguments("--encoder-pretrained", "false", "--no-use-amp",
                              "--save-predictions", "true", "--resume", "last.pth")
        self.assertFalse(args.encoder_pretrained)
        self.assertFalse(args.use_amp)
        self.assertTrue(args.save_predictions)
        self.assertEqual(args.resume_checkpoint, "last.pth")
        self.assertIsNone(args.resume_ckpt)
        self.assert_invalid("--weights", "warm.pth", "--resume", "last.pth")

    def test_default_output_is_contained_in_standalone_directory(self):
        args = self.arguments()
        self.assertEqual(Path(args.output_dir).parent, PACKAGE_ROOT / "runs")
        self.assertTrue(Path(args.output_dir).name.startswith("FAHSYSU_PACE_"))
        self.assertEqual(args.checkpoint_dir, args.output_dir)

    def test_invalid_numerical_settings_are_rejected(self):
        for options in (("--d_model", "255"), ("--quality_max_severity", "2"),
                        ("--lr", "nan"), ("--seed", "-1"),
                        ("--num_workers", "-1"), ("--force_t", "2"),
                        ("--quality_min_severity", ".9", "--quality_max_severity", ".7")):
            with self.subTest(options=options):
                self.assert_invalid(*options)

    def test_help_does_not_require_training_dependencies(self):
        result = subprocess.run(
            [sys.executable, "-B", str(PACKAGE_ROOT / "main.py"), "--help"],
            cwd=PACKAGE_ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("FAHSYSU", result.stdout)
        self.assertIn("PICCOLO", result.stdout)
        self.assertIn("100 epochs", result.stdout)

    def test_cpu_launch_and_zero_process_rejection(self):
        with patch.dict(os.environ, {}, clear=True):
            args = self.arguments("--devices", "cpu", "--dist_backend", "gloo")
            self.assertFalse(main._resolve_launch(config.build_parser(), args))
            self.assertTrue(args.cpu)
            self.assertEqual(args.gpus, 1)
            self.assertEqual(args.world_size, 1)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stderr(io.StringIO()):
            args = self.arguments("--devices", "cpu", "--dist_backend", "gloo", "--gpus", "0")
            with self.assertRaises(SystemExit):
                main._resolve_launch(config.build_parser(), args)

    def test_gpu_visibility_environment_is_respected_before_torch_import(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,5"}, clear=True):
            args = self.arguments()
            self.assertFalse(main._resolve_launch(config.build_parser(), args))
            self.assertEqual(args.devices, "2,5")
            self.assertEqual(args.gpus, 2)
            self.assertEqual(args.world_size, 2)

    def test_torchrun_resolution_and_incomplete_environment(self):
        environment = {"RANK": "1", "WORLD_SIZE": "2", "LOCAL_RANK": "1", "LOCAL_WORLD_SIZE": "2"}
        with patch.dict(os.environ, environment, clear=True):
            args = self.arguments("--devices", "cpu", "--dist_backend", "gloo")
            self.assertTrue(main._resolve_launch(config.build_parser(), args))
            self.assertEqual((args.rank, args.local_rank, args.world_size, args.gpus), (1, 1, 2, 2))
            self.assertEqual(args.launcher, "torchrun")
        with patch.dict(os.environ, {"RANK": "1"}, clear=True), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main._resolve_launch(config.build_parser(), self.arguments())


if __name__ == "__main__":
    unittest.main()
