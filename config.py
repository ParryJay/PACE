"""Command-line configuration for the shared PACE pipeline.

This module uses only the Python standard library, so ``--help`` does not
require PyTorch or access any dataset.
"""
from __future__ import annotations

import argparse
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo


QUALITY_OPERATIONS = (
    "gaussian_blur", "motion_blur", "defocus_blur", "occlusion",
    "overexposure", "low_illumination", "specular_highlight", "black_border",
    "color_corruption",
)


def _options(name: str, *aliases: str) -> tuple[str, ...]:
    options = []
    for name in (name, *aliases):
        option = name if name.startswith("-") else "--" + name
        options.append(option)
        if option.startswith("--"):
            options.append(option.replace("_", "-"))
    return tuple(dict.fromkeys(options))


def _boolean(value: object) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true/false, yes/no, or 1/0")


def _bool(group, name, default, help, *aliases):
    group.add_argument(*_options(name, *aliases), dest=name, type=_boolean,
                       nargs="?", const=True, default=default, metavar="BOOL", help=help)
    group.add_argument(*_options("no_" + name, "no-" + name.replace("_", "-")),
                       dest=name, action="store_false", default=argparse.SUPPRESS,
                       help=argparse.SUPPRESS)


def _arg(group, name, default, type, help, *aliases, **kwargs):
    group.add_argument(*_options(name, *aliases), dest=name, default=default,
                       type=type, help=help, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Train PACE for 100 epochs, save the final checkpoint, then "
                     "run test inference once. No test metrics or visualizations."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    data = parser.add_argument_group("dataset and fixed training protocol")
    data.add_argument(*_options("dataset"), required=True, choices=("FAHSYSU", "PICCOLO"),
                      help="FAHSYSU uses the original SYSU pair format.")
    data.add_argument(*_options("dataset_root"), required=True, help="Dataset root directory.")
    _arg(data, "train_manifest", None, str, "Optional training JSON/JSONL manifest.")
    _arg(data, "test_manifest", None, str, "Optional test JSON/JSONL manifest; read only after training.")
    _arg(data, "img_size", 512, int, "Reference input size; must be 512.")
    _arg(data, "epochs", 100, int, "Training duration; must be 100.")
    _arg(data, "batch_train", 6, int, "Training batch size per process.")
    _arg(data, "batch_test", 8, int, "Final inference batch size per process.")
    _arg(data, "smoke_steps", 0, int, "Debug training for this many steps; never runs test or saves final weights.")

    runtime = parser.add_argument_group("reproducibility and loading")
    _arg(runtime, "seed", 3407, int, "Shared initialization seed.")
    _bool(runtime, "deterministic", False, "Use deterministic PyTorch behavior where available.")
    _arg(runtime, "num_workers", 6, int, "Training loader workers per process.")
    _arg(runtime, "test_num_workers", 1, int, "Final inference loader workers per process.")
    _arg(runtime, "prefetch_factor", 2, int, "Training prefetch factor when workers are enabled.")
    _bool(runtime, "persistent_workers", False, "Keep training workers alive across epochs.")
    _bool(runtime, "pin_memory", True, "Use pinned loader memory.")

    optim = parser.add_argument_group("optimization")
    for name, default, help in (
        ("lr", 1e-4, "Base learning rate for projections and heads."),
        ("lr_backbone", 1e-5, "Top backbone stage learning rate."),
        ("weight_decay", 1e-3, "AdamW decay for matrix/kernel parameters."),
        ("backbone_layer_decay", .85, "Backbone layer learning-rate decay."),
        ("fusion_lr_multiplier", 1.5, "Learning-rate multiplier for fusion parameters."),
        ("max_grad_norm", 1., "Global gradient clipping norm."),
        ("warmup_epochs", 5., "Head/backbone learning-rate warmup epochs."),
        ("fusion_warmup_epochs", 1., "Fusion learning-rate warmup epochs."),
        ("min_lr_ratio", .05, "Cosine final/base learning-rate ratio."),
        ("lr_decay", .985, "Per-epoch exponential scheduler multiplier."),
        ("amp_init_scale", 4096., "Initial FP16 GradScaler scale."),
        ("amp_min_scale", 1., "Minimum recoverable FP16 GradScaler scale."),
        ("ema_decay", .999, "Exponential moving average decay."),
    ):
        _arg(optim, name, default, float, help)
    _arg(optim, "lr_scheduler", "warmup_cosine", str, "Learning-rate scheduler.",
         choices=("warmup_cosine", "exponential"))
    _arg(optim, "freeze_backbone_epochs", 0, int, "Initial epochs with zero backbone/projection learning rate.")
    _arg(optim, "finetune_mode", "all", str, "Optimizer parameter policy.", choices=("all", "fusion_only"))
    _bool(optim, "use_amp", True, "Enable CUDA automatic mixed precision.", "amp")
    _arg(optim, "amp_dtype", "bfloat16", str, "CUDA autocast data type.", choices=("float16", "bfloat16"))
    _arg(optim, "amp_growth_interval", 2000, int, "Successful FP16 updates before scale growth.")
    _arg(optim, "amp_max_consecutive_overflows", 16, int, "Maximum recoverable consecutive FP16 overflows.")
    _bool(optim, "use_ema", True, "Maintain a training weight exponential moving average.")
    _arg(optim, "ema_update_after_steps", 100, int, "Optimizer steps before EMA updates begin.")
    _bool(optim, "ema_eval", True, "Use final-epoch EMA weights for the one final inference pass.")

    model = parser.add_argument_group("model, graph matching, and audit")
    _arg(model, "encoder_name", "convnextv2_base", str, "timm encoder name.")
    _bool(model, "encoder_pretrained", True, "Initialize the encoder from pretrained timm weights.")
    _bool(model, "freeze_backbone", False, "Permanently disable backbone gradients.")
    _arg(model, "d_model", 256, int, "Pyramid and decoder channel dimension.")
    _arg(model, "token_dim", None, int, "Graph token dimension; defaults to d_model.")
    _arg(model, "num_tokens", 128, int, "Number of graph nodes per modality.", "N", "num_nodes")
    _arg(model, "preliminary_hidden_channels", 128, int, "Graph Q/B/V head hidden channels.")
    _arg(model, "fusion_num_heads", 4, int, "Attention head count.")
    for name, default, help in (
        ("rho_min", .2, "Minimum partial transport mass."),
        ("rho_max", .85, "Maximum partial transport mass."),
        ("alpha_min", .05, "Minimum graph structure mixing weight."),
        ("alpha_max", .35, "Maximum graph structure mixing weight."),
        ("transport_epsilon", .05, "Entropic transport regularization."),
        ("audit_tau", .5, "Maximum node uncertainty for correspondence audit."),
        ("node_spatial_strength", 8., "Initial learned positional assignment penalty."),
        ("lambda_f", 1., "Semantic feature cosine matching cost weight."),
        ("lambda_s", .5, "Semantic role matching cost weight."),
        ("lambda_b", .5, "Semantic boundary matching cost weight."),
    ):
        _arg(model, name, default, float, help)
    for name in ("p", "q", "b", "d"):
        _arg(model, "theta_" + name, 1., float, "Graph relative-" + name + " cost weight.")
    _arg(model, "transport_fgw_iterations", 5, int, "Fixed outer FGW iterations.")
    _arg(model, "transport_sinkhorn_iterations", 50, int, "Fixed inner Sinkhorn iterations.")
    _bool(model, "match_use_abs_pos", False, "Use absolute positions in semantic graph matching.")
    _arg(model, "opinion_grad", "detach", str, "Evidence opinion gradient policy.", choices=("detach", "through"))
    _arg(model, "l2_kv", "self", str, "F2 window attention key/value modality.", choices=("self", "cross"))
    _arg(model, "decision_fusion", "add", str, "Decision fusion rule.", choices=("add", "off"))
    _arg(model, "force_t", None, float, "Optional fixed directional reliability override.")

    losses = parser.add_argument_group("training losses")
    _arg(losses, "seg_loss_mode", "foreground", str, "Segmentation training objective.", choices=("foreground", "legacy"))
    for name, default, help in (
        ("ce_loss_weight", .5, "Cross-entropy component weight."),
        ("dice_loss_weight", .5, "Training Dice/Tversky component weight."),
        ("foreground_ce_weight", 1.5, "Foreground class weight in cross entropy."),
        ("foreground_tversky_alpha", .4, "False-positive Tversky coefficient."),
        ("foreground_tversky_beta", .6, "False-negative Tversky coefficient."),
        ("lambda_seg", 1., "Final segmentation training loss weight."),
        ("lambda_dec", .5, "Decoder segmentation training loss weight."),
        ("lambda_q", .2, "Coarse lesion supervision weight."),
        ("lambda_boundary", .2, "Boundary supervision weight."),
        ("lambda_quality", .1, "Detached correctness-quality supervision weight."),
        ("lambda_e", .2, "Beta evidential supervision weight."),
        ("lambda_match", 0., "Optional own-GT transport disagreement loss weight."),
        ("aux_warmup_epochs", 5., "Auxiliary training-loss warmup epochs."),
        ("max_auto_pos_weight", 20., "Maximum automatic boundary positive weight."),
    ):
        _arg(losses, name, default, float, help)
    _arg(losses, "boundary_pos_weight", None, float, "Optional manual boundary BCE positive weight.")

    quality = parser.add_argument_group("training augmentation")
    _bool(quality, "use_spatial_augmentation", True, "Apply paired image/mask square symmetries.")
    _bool(quality, "use_synthetic_quality_degradation", True, "Enable synthetic training image degradations.")
    _arg(quality, "quality_degradation_pair_policy", "single_modality", str,
         "Which modalities may be degraded.", choices=("single_modality", "independent"))
    for name, default, help in (
        ("quality_nbi_degrade_probability", .5, "Conditional NBI degradation selection probability."),
        ("quality_curriculum_epochs", 10., "Degradation strength/probability ramp duration."),
        ("quality_apply_probability", .8, "Per-selected-modality degradation probability."),
        ("quality_min_severity", .2, "Minimum degradation severity."),
        ("quality_max_severity", .7, "Maximum degradation severity before curriculum scaling."),
        ("quality_local_probability", .8, "Local versus global degradation probability."),
    ):
        _arg(quality, name, default, float, help)
    _arg(quality, "quality_seed", None, int, "Degradation seed; defaults to the run seed.")
    _arg(quality, "quality_max_operations", 2, int, "Maximum simultaneous degradation operations.")
    for operation in QUALITY_OPERATIONS:
        _bool(quality, "quality_" + operation, True, "Enable training " + operation.replace("_", " ") + ".")
        _arg(quality, "quality_" + operation + "_probability", 1., float, "Sampling probability for " + operation.replace("_", " ") + ".")

    checkpoint = parser.add_argument_group("checkpoint and final inference")
    _arg(checkpoint, "output_dir", None, str, "Output directory; defaults to runs/<dataset>_PACE_<timestamp> beside this file.")
    loading = checkpoint.add_mutually_exclusive_group()
    _arg(loading, "resume_checkpoint", None, str, "Resume complete training state, without accessing the test set.", "resume")
    _arg(loading, "resume_ckpt", None, str, "Warm-start model weights; final checkpoint for standalone inference.", "weights")
    _bool(checkpoint, "weights_strict", True, "Require exact parameter names for weight loading.")
    _arg(checkpoint, "weights_source", "auto", str, "Weight source for warm-start loading and final inference.", choices=("auto", "online", "ema"))
    _arg(checkpoint, "checkpoint_interval", 0, int, "Extra epoch checkpoint interval; 0 keeps only last and final.")
    _bool(checkpoint, "inference_only", False, "Infer from a final 100-epoch checkpoint without training.", "eval_only", "test_only")
    _bool(checkpoint, "save_predictions", False, "Save raw prediction arrays as NPZ; never generate visualization images.")

    distributed = parser.add_argument_group("distributed launch")
    _arg(distributed, "devices", None, str, "Visible GPU IDs, e.g. 0,1; cpu forces CPU. Defaults to CUDA_VISIBLE_DEVICES, then 0.")
    _arg(distributed, "nodes", 1, int, "Number of nodes for the built-in launcher.", "-n")
    _arg(distributed, "gpus", None, int, "Processes per node; inferred from devices.", "-g")
    _arg(distributed, "nr", 0, int, "Zero-based node rank.", "node_rank")
    _arg(distributed, "master_addr", os.environ.get("MASTER_ADDR", "127.0.0.1"), str, "Distributed rendezvous address.")
    _arg(distributed, "master_port", 29500, int, "Distributed rendezvous port.")
    _arg(distributed, "dist_backend", "nccl", str, "Distributed backend; use gloo for CPU.", choices=("nccl", "gloo"))
    _arg(distributed, "dist_timeout_minutes", 30, int, "Distributed initialization timeout in minutes.")
    _bool(distributed, "ddp_find_unused_parameters", False, "Enable DDP unused-parameter traversal.")
    _arg(distributed, "local_rank", None, int, argparse.SUPPRESS)
    return parser


def normalize_and_validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    if args.epochs != 100:
        parser.error("--epochs must be 100: test inference is allowed only after all 100 training epochs.")
    if args.img_size != 512:
        parser.error("--img_size must be 512 to preserve the reference preprocessing.")
    if not args.dataset_root.strip():
        parser.error("--dataset_root cannot be empty.")
    for name, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            parser.error(f"--{name} must be finite.")
    for name in ("batch_train", "batch_test", "prefetch_factor", "d_model", "num_tokens",
                 "preliminary_hidden_channels", "fusion_num_heads", "transport_fgw_iterations",
                 "transport_sinkhorn_iterations", "amp_growth_interval", "amp_max_consecutive_overflows",
                 "dist_timeout_minutes", "nodes"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive.")
    for name in ("num_workers", "test_num_workers", "freeze_backbone_epochs", "ema_update_after_steps",
                 "quality_max_operations", "smoke_steps", "checkpoint_interval"):
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be non-negative.")
    for name in ("lr", "lr_backbone", "fusion_lr_multiplier", "max_grad_norm", "lr_decay",
                 "amp_init_scale", "amp_min_scale", "transport_epsilon", "node_spatial_strength",
                 "foreground_ce_weight"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive.")
    for name in ("weight_decay", "warmup_epochs", "fusion_warmup_epochs", "aux_warmup_epochs",
                 "quality_curriculum_epochs", "ce_loss_weight", "dice_loss_weight", "foreground_tversky_alpha",
                 "foreground_tversky_beta", "lambda_seg", "lambda_dec", "lambda_q", "lambda_boundary",
                 "lambda_quality", "lambda_e", "lambda_match", "lambda_f", "lambda_s", "lambda_b",
                 "theta_p", "theta_q", "theta_b", "theta_d"):
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be non-negative.")
    for name in ("min_lr_ratio", "quality_nbi_degrade_probability", "quality_apply_probability",
                 "quality_min_severity", "quality_max_severity", "quality_local_probability",
                 *("quality_" + op + "_probability" for op in QUALITY_OPERATIONS)):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name} must be in [0,1].")
    if not 0 < args.backbone_layer_decay <= 1 or not 0 < args.ema_decay < 1:
        parser.error("Require backbone_layer_decay in (0,1] and ema_decay in (0,1).")
    if not 0 < args.rho_min < args.rho_max < 1:
        parser.error("Require 0 < rho_min < rho_max < 1.")
    if not 0 <= args.alpha_min < args.alpha_max <= 1:
        parser.error("Require 0 <= alpha_min < alpha_max <= 1.")
    if not 0 < args.audit_tau < 1 or (args.force_t is not None and not 0 <= args.force_t <= 1):
        parser.error("Require audit_tau in (0,1) and force_t in [0,1].")
    if args.quality_min_severity > args.quality_max_severity:
        parser.error("quality_min_severity cannot exceed quality_max_severity.")
    if args.amp_min_scale > args.amp_init_scale:
        parser.error("amp_min_scale cannot exceed amp_init_scale.")
    if args.ce_loss_weight + args.dice_loss_weight <= 0:
        parser.error("At least one segmentation component must have positive weight.")
    if args.foreground_tversky_alpha + args.foreground_tversky_beta <= 0:
        parser.error("Tversky alpha and beta must have a positive sum.")
    if args.max_auto_pos_weight < 1 or (args.boundary_pos_weight is not None and args.boundary_pos_weight <= 0):
        parser.error("max_auto_pos_weight must be >= 1; boundary_pos_weight must be positive.")
    if args.token_dim is not None and args.token_dim < 1:
        parser.error("--token_dim must be positive.")
    if args.d_model % args.fusion_num_heads or (args.token_dim or args.d_model) % args.fusion_num_heads:
        parser.error("d_model and token_dim must be divisible by fusion_num_heads.")
    if args.freeze_backbone_epochs and args.lr_scheduler != "warmup_cosine":
        parser.error("Initial backbone freezing requires warmup_cosine.")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be in [0,2**32-1].")
    if args.inference_only and (not args.resume_ckpt or args.resume_checkpoint or args.smoke_steps):
        parser.error("--inference_only requires --weights and cannot use --resume or --smoke_steps.")
    if not args.inference_only and args.weights_source == "ema" and not args.use_ema:
        parser.error("--weights_source ema requires --use_ema for final inference after training.")
    if args.devices and args.devices.strip().lower() == "cpu" and args.dist_backend != "gloo":
        parser.error("--devices cpu requires --dist_backend gloo.")
    args.quality_seed = args.seed if args.quality_seed is None else args.quality_seed
    args.dataset_root = os.fspath(Path(args.dataset_root).expanduser().resolve())
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    output = Path(args.output_dir).expanduser() if args.output_dir else Path(__file__).resolve().parent / "runs" / f"{args.dataset}_PACE_{stamp}"
    args.output_dir = os.fspath(output.resolve())
    args.checkpoint_dir = args.output_dir
    args.method = "PACE"
    args.num_classes = 2
    return args


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    return normalize_and_validate_args(parser, parser.parse_args(argv))
