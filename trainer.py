"""Shared PACE training: 100 epochs, then inference from the final checkpoint.

Only training images and labels are opened before all 100 epochs finish.
Checkpoint retention follows epoch progress, independently of test predictions.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

if __package__:
    from .augmentation import _build_quality_degrader_from_args, _image_to_unit_interval
    from .data import PairedDataset
    from .model import build_model
else:
    from augmentation import _build_quality_degrader_from_args, _image_to_unit_interval
    from data import PairedDataset
    from model import build_model


MODALITIES = ("nbi", "wli")
TOTAL_EPOCHS = 100
CHECKPOINT_FORMAT = "PACE_opensource_v1"


def _effective_train_config(args: Any) -> dict[str, Any]:
    """Resolve training semantics independently of paths, launch and inference.

    Group membership follows the public CLI so new loss/model/augmentation
    options are checked automatically. Runtime I/O settings can change on
    resume; the dataset and optimizer schedule are validated separately.
    """
    if __package__:
        from .config import build_parser
    else:
        from config import build_parser
    parser = build_parser()
    groups = {"optimization", "model, graph matching, and audit",
              "training losses", "training augmentation"}
    names = {action.dest for group in parser._action_groups if group.title in groups
             for action in group._group_actions}
    names.update(("img_size", "epochs", "batch_train", "seed", "deterministic"))
    # Pretrained initialization is overwritten by resume; EMA selection affects
    # only the one final inference pass, not training updates.
    names.difference_update(("encoder_pretrained", "ema_eval"))
    resolved = {name: getattr(args, name, parser.get_default(name)) for name in sorted(names)}
    resolved["quality_seed"] = resolved["seed"] if resolved["quality_seed"] is None else resolved["quality_seed"]
    resolved["token_dim"] = resolved["d_model"] if resolved["token_dim"] is None else resolved["token_dim"]
    # Programmatic callers receive the same effective defaults as CLI callers.
    for name, value in resolved.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    return _jsonable(resolved)



def normalize_binary_mask(mask: torch.Tensor) -> torch.Tensor:
    """Convert bool, 0/1, or 0/255 masks to float32 ``[B,1,H,W]``."""

    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"mask must be a torch.Tensor, got {type(mask).__name__}.")
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(
            "Binary mask must have shape [B,H,W] or [B,1,H,W], got "
            f"{tuple(mask.shape)}."
        )
    return (mask.detach().float() > 0.5).to(dtype=torch.float32)



@torch.no_grad()
def build_morphological_boundary(
    mask: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """Build a binary inner/outer boundary band using a PyTorch gradient."""

    radius = int(radius)
    if radius < 1:
        raise ValueError(f"Boundary radius must be >= 1, got {radius}.")
    binary_mask = normalize_binary_mask(mask)
    kernel_size = 2 * radius + 1
    # Explicit zero padding makes erosion treat pixels outside the image as
    # background, including when a lesion touches an image edge.
    padded = F.pad(binary_mask, (radius,) * 4, mode="constant", value=0.0)
    dilated = F.max_pool2d(padded, kernel_size=kernel_size, stride=1)
    eroded = -F.max_pool2d(-padded, kernel_size=kernel_size, stride=1)
    return (dilated - eroded).clamp_(0.0, 1.0)



class _MulticlassDiceLoss(torch.nn.Module):
    """Soft Dice loss averaged over all segmentation classes."""

    def __init__(self, num_classes=2, smooth=1e-5):
        super().__init__()
        self.num_classes = int(num_classes)
        self.smooth = float(smooth)

    def forward(self, logits, target):
        if target.ndim == 4 and target.shape[1] == 1:
            target = target[:, 0]
        if target.ndim != 3:
            raise ValueError("Dice target must have shape [B,H,W] or [B,1,H,W].")

        with torch.autocast(device_type=logits.device.type, enabled=False):
            probabilities = torch.softmax(logits.float(), dim=1)
            target_one_hot = F.one_hot(
                target.long(), num_classes=self.num_classes
            ).permute(0, 3, 1, 2).float()
            dims = (0, 2, 3)
            intersection = (probabilities * target_one_hot).sum(dim=dims)
            denominator = probabilities.square().sum(dim=dims)
            denominator = denominator + target_one_hot.square().sum(dim=dims)
            dice = (2.0 * intersection + self.smooth) / (
                denominator + self.smooth
            )
            return 1.0 - dice.mean()



class _ForegroundTverskyLoss(torch.nn.Module):
    """Per-sample foreground Tversky loss for small-lesion segmentation."""

    def __init__(self, alpha: float = 0.30, beta: float = 0.70, smooth: float = 1e-5):
        super().__init__()
        if alpha < 0 or beta < 0 or alpha + beta <= 0:
            raise ValueError("Tversky alpha/beta must be non-negative with positive sum.")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 4 and target.shape[1] == 1:
            target = target[:, 0]
        if logits.ndim != 4 or logits.shape[1] != 2 or target.ndim != 3:
            raise ValueError("Foreground Tversky expects logits [B,2,H,W], target [B,H,W].")
        with torch.autocast(device_type=logits.device.type, enabled=False):
            probability = torch.softmax(logits.float(), dim=1)[:, 1]
            target_float = target.float()
            dims = (1, 2)
            true_positive = (probability * target_float).sum(dim=dims)
            false_positive = (probability * (1.0 - target_float)).sum(dim=dims)
            false_negative = ((1.0 - probability) * target_float).sum(dim=dims)
            score = (true_positive + self.smooth) / (
                true_positive
                + self.alpha * false_positive
                + self.beta * false_negative
                + self.smooth
            )
            return (1.0 - score).mean()



def _seed_data_worker(worker_id: int) -> None:
    """Seed Python/NumPy from PyTorch's deterministic worker seed."""

    del worker_id
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)



class _ModelEMA:
    """Device-local weight EMA used only for final-checkpoint inference."""

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float,
        update_after_steps: int,
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0,1).")
        if update_after_steps < 0:
            raise ValueError("EMA update_after_steps must be non-negative.")
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)
        self.decay = float(decay)
        self.update_after_steps = int(update_after_steps)
        self.updates = 0
        self.initialized = False

    @torch.no_grad()
    def update(self, source: torch.nn.Module) -> None:
        self.updates += 1
        if self.updates < self.update_after_steps:
            return
        source_state = source.state_dict()
        target_state = self.model.state_dict()
        if not self.initialized:
            for name, target in target_state.items():
                target.copy_(source_state[name].detach())
            self.initialized = True
            return
        # A short bias correction prevents an excessively stale average early on.
        effective_decay = min(
            self.decay, (1.0 + self.updates) / (10.0 + self.updates)
        )
        for name, target in target_state.items():
            source_value = source_state[name].detach()
            if target.is_floating_point():
                target.lerp_(source_value, 1.0 - effective_decay)
            else:
                target.copy_(source_value)

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "decay": self.decay,
            "update_after_steps": self.update_after_steps,
            "updates": self.updates,
            "initialized": self.initialized,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        load_result = self.model.load_state_dict(state["model"], strict=True)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(
                "EMA resume requires an exact model state: "
                f"missing={load_result.missing_keys}, "
                f"unexpected={load_result.unexpected_keys}."
            )
        self.decay = float(state.get("decay", self.decay))
        self.update_after_steps = int(
            state.get("update_after_steps", self.update_after_steps)
        )
        if not 0.0 < self.decay < 1.0 or self.update_after_steps < 0:
            raise ValueError("Checkpoint contains an invalid EMA configuration.")
        self.updates = int(state.get("updates", 0))
        self.initialized = bool(state.get("initialized", True))



def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()



class _EpochProgress:
    """Overwrite rank-zero terminal progress; keep redirected logs concise."""

    def __init__(self, enabled: bool):
        self.stream = sys.stdout
        self.enabled = enabled and self.stream.isatty()
        self.width = 0

    def __enter__(self):
        return self

    def update(self, text: str):
        if not self.enabled:
            return
        available = max(1, shutil.get_terminal_size((120, 20)).columns - 1)
        text = text[:available]
        padded_width = min(available, max(self.width, len(text)))
        self.stream.write("\r" + text.ljust(padded_width))
        self.stream.flush()
        self.width = len(text)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enabled and self.width:
            # Leave failures on a fresh line; successful epochs replace the
            # transient progress with the persistent epoch summary below.
            self.stream.write("\n" if exc_type is not None else "\r" + " " * self.width + "\r")
            self.stream.flush()
        return False



def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model



def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return os.fspath(value)
    return value



def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)



def node_targets(graph: Mapping[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    """Own-modality GT averages; no correspondence labels are used."""
    assignment = graph["assignment"].float()
    soft_target = F.interpolate(target.float(), size=assignment.shape[-2:], mode="area")
    mass = graph["mass"].float()
    return (assignment * soft_target).flatten(2).sum(-1) / mass.clamp_min(1.0e-8)



def evidential_loss(
    graph: Mapping[str, torch.Tensor], target: torch.Tensor, epoch_progress: float,
) -> torch.Tensor:
    """Expected Beta BCE and a ten-epoch annealed target-removed Beta KL."""
    with torch.autocast(device_type=target.device.type, enabled=False):
        evidence = graph["evidence"].float()
        label = node_targets(graph, target)
        positive, negative = evidence.unbind(-1)
        alpha_positive, alpha_negative = positive + 1.0, negative + 1.0
        total = alpha_positive + alpha_negative
        expected_bce = (
            label * (torch.digamma(total) - torch.digamma(alpha_positive))
            + (1.0 - label) * (torch.digamma(total) - torch.digamma(alpha_negative))
        )
        # Soft-label extension of removing the evidence for the target class.
        removed_positive = 1.0 + (1.0 - label) * positive
        removed_negative = 1.0 + label * negative
        removed_total = removed_positive + removed_negative
        kl = (
            torch.lgamma(removed_total)
            - torch.lgamma(removed_positive) - torch.lgamma(removed_negative)
            + (removed_positive - 1.0) * torch.digamma(removed_positive)
            + (removed_negative - 1.0) * torch.digamma(removed_negative)
            - (removed_total - 2.0) * torch.digamma(removed_total)
        )
        anneal = 0.1 * min(1.0, max(0.0, float(epoch_progress)) / 10.0)
        mass = graph["mass"].float()
        return ((expected_bce + anneal * kl) * mass).sum(-1).div(
            mass.sum(-1).clamp_min(1.0e-8)
        ).mean()



class Trainer:
    """Train both datasets with the same model and optimization policy."""

    def __init__(self, gpu: int, rank: int, world_size: int, args: Any):
        self.gpu, self.rank, self.world_size = int(gpu), int(rank), int(world_size)
        self.args = args
        self.train_config = _effective_train_config(args)
        self.seed = int(getattr(args, "seed", 3407))
        random.seed(self.seed)
        np.random.seed(self.seed % (2**32))
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        if int(getattr(args, "img_size", 512)) != 512:
            raise ValueError("PACE uses the reference 512 x 512 input pipeline.")
        if int(args.epochs) != TOTAL_EPOCHS:
            raise ValueError("PACE training must complete exactly 100 epochs before testing.")
        has_weights = bool(getattr(args, "resume_checkpoint", None) or getattr(args, "resume_ckpt", None))
        self.model = build_model(args, pretrained=False) if has_weights else build_model(args)
        self.seg_loss_mode = str(getattr(args, "seg_loss_mode", "foreground"))
        self.foreground_ce_weight = float(getattr(args, "foreground_ce_weight", 1.5))
        self.loss_ce = torch.nn.CrossEntropyLoss()
        self.loss_dice = _MulticlassDiceLoss(num_classes=2)
        self.loss_tversky = _ForegroundTverskyLoss(
            alpha=float(getattr(args, "foreground_tversky_alpha", 0.4)),
            beta=float(getattr(args, "foreground_tversky_beta", 0.6)),
        )
        self.ce_loss_weight = float(getattr(args, "ce_loss_weight", 0.5))
        self.dice_loss_weight = float(getattr(args, "dice_loss_weight", 0.5))
        self.dataset_train = PairedDataset(
            args.dataset, args.dataset_root, getattr(args, "train_manifest", None),
            split="train", img_size=512,
        )
        self.output_dir = Path(args.output_dir)
        if self.rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        records = json.dumps(self.dataset_train.pairs, sort_keys=True, ensure_ascii=False)
        self.training_data = {
            "dataset": args.dataset,
            "root": os.fspath(Path(args.dataset_root).expanduser().resolve()),
            "pair_count": len(self.dataset_train),
            "pairs_sha256": hashlib.sha256(records.encode("utf-8")).hexdigest(),
        }
        self.degrader = None
        if bool(getattr(args, "use_synthetic_quality_degradation", True)):
            quality_seed = getattr(args, "quality_seed", self.seed)
            self.quality_seed = self.seed if quality_seed is None else int(quality_seed)
            self.degrader = _build_quality_degrader_from_args(
                args, seed=self.quality_seed + 104729 * self.rank
            )
        self._quality_base_probability = self.degrader.apply_probability if self.degrader else 0.0
        self._quality_base_severity = self.degrader.max_severity if self.degrader else 0.0
        self.current_epoch = 0


    def _segmentation_modality_loss(self, logits: torch.Tensor, target: torch.Tensor):
        if self.seg_loss_mode == "legacy":
            loss_ce = self.loss_ce(logits, target)
            loss_overlap = self.loss_dice(logits, target)
        else:
            class_weight = logits.new_tensor([1.0, self.foreground_ce_weight])
            with torch.autocast(device_type=logits.device.type, enabled=False):
                loss_ce = F.cross_entropy(logits.float(), target.long(), weight=class_weight.float())
            loss_overlap = self.loss_tversky(logits, target)
        return self.ce_loss_weight * loss_ce + self.dice_loss_weight * loss_overlap, loss_ce, loss_overlap


    def compute_loss(
        self, outputs: Mapping[str, Any], mask_nbi: torch.Tensor,
        mask_wli: torch.Tensor, epoch_progress: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        progress = float(self.current_epoch if epoch_progress is None else epoch_progress)
        targets = {"nbi": normalize_binary_mask(mask_nbi), "wli": normalize_binary_mask(mask_wli)}
        components: dict[str, torch.Tensor] = {}
        weights = {
            "seg": float(getattr(self.args, "lambda_seg", 1.0)), "dec": float(getattr(self.args, "lambda_dec", 0.5)),
            "q": float(getattr(self.args, "lambda_q", 0.2)),
            "boundary": float(getattr(self.args, "lambda_boundary", 0.2)),
            "quality": float(getattr(self.args, "lambda_quality", 0.1)),
            "edl": float(getattr(self.args, "lambda_e", 0.2)),
            "match": float(getattr(self.args, "lambda_match", 0.0)),
        }
        for modality in MODALITIES:
            target = targets[modality]
            graph = outputs["graph_" + modality]
            logits = outputs["logits_final_" + modality]
            final, ce, overlap = self._segmentation_modality_loss(logits, target[:, 0].long())
            components["seg_" + modality] = final
            components["ce_" + modality] = ce.detach()
            components["overlap_" + modality] = overlap.detach()
            if getattr(self.args, "decision_fusion", "add") == "add":
                dec, _, _ = self._segmentation_modality_loss(outputs["logits_dec_" + modality], target[:, 0].long())
                components["dec_" + modality] = dec
            else:
                components["dec_" + modality] = final.new_zeros(())
            with torch.autocast(device_type=target.device.type, enabled=False):
                q_logits = outputs["q_logits_" + modality].float()
                b_logits = outputs["b_logits_" + modality].float()
                v_logits = outputs["v_logits_" + modality].float()
                downsampled = F.interpolate(target.float(), size=q_logits.shape[-2:], mode="area")
                boundary = build_morphological_boundary(target, radius=2)
                boundary = F.adaptive_max_pool2d(boundary, b_logits.shape[-2:])
                explicit_weight = getattr(self.args, "boundary_pos_weight", None)
                if explicit_weight is None:
                    positives = boundary.sum()
                    pos_weight = ((boundary.numel() - positives) / positives.clamp_min(1.0)).clamp(
                        min=1.0, max=float(getattr(self.args, "max_auto_pos_weight", 20.0))
                    ).detach()
                else:
                    pos_weight = b_logits.new_tensor(float(explicit_weight))
                components["q_" + modality] = F.binary_cross_entropy_with_logits(q_logits, downsampled)
                components["boundary_" + modality] = F.binary_cross_entropy_with_logits(b_logits, boundary, pos_weight=pos_weight)
                quality_target = (1.0 - (torch.sigmoid(q_logits) - downsampled).abs()).detach()
                components["quality_" + modality] = F.binary_cross_entropy_with_logits(v_logits, quality_target)
                components["edl_" + modality] = evidential_loss(graph, target, progress)
        labels_nbi = node_targets(outputs["graph_nbi"], targets["nbi"])
        labels_wli = node_targets(outputs["graph_wli"], targets["wli"])
        with torch.autocast(device_type=mask_nbi.device.type, enabled=False):
            plan = outputs["Pi"].float()
            components["match"] = (
                (plan * (labels_nbi[:, :, None] - labels_wli[:, None, :]).abs()).sum((1, 2))
                / outputs["rho"].float().reshape(-1).clamp_min(1.0e-8)
            ).mean()
        # Keep both decoder objectives active while Q/B/V and node evidence
        # learn their meaning. Previously the CLI warm-up had no effect here.
        warmup = float(getattr(self.args, "aux_warmup_epochs", 5.0))
        aux_scale = min(1.0, max(0.0, progress) / warmup) if warmup > 0 else 1.0
        primary = sum(
            weights[name] * sum(components[name + "_" + modality] for modality in MODALITIES)
            for name in ("seg", "dec")
        )
        auxiliary = sum(
            weights[name] * sum(components[name + "_" + modality] for modality in MODALITIES)
            for name in ("q", "boundary", "quality", "edl")
        ) + weights["match"] * components["match"]
        total = primary + aux_scale * auxiliary
        components["aux_scale"] = total.new_tensor(aux_scale)
        components["total"] = total
        return total, {key: value.detach() for key, value in components.items()}


    def _optimizer(self, model: torch.nn.Module, steps_per_epoch: int):
        args = self.args
        grouped: dict[tuple[str, float, float], list[torch.nn.Parameter]] = {}
        fusion_prefixes = ("pair_estimator.", "global_fusion.", "node_fusion.", "window_fusion.", "fusion.", "tspr")
        fusion_prefixes = tuple(getattr(model, "scope_state_prefixes", fusion_prefixes))
        included: set[int] = set()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            no_decay = parameter.ndim <= 1 or name.endswith(".bias") or any(
                marker in name for marker in ("logit_scale", "temperature", "layer_scale", "position_prototypes")
            )
            decay = 0.0 if no_decay else float(args.weight_decay)
            if name.startswith("encoder.backbone."):
                relative = name.removeprefix("encoder.backbone.")
                layer = 0
                for stage in range(4):
                    if f"stages_{stage}." in relative or f"stages.{stage}." in relative:
                        layer = stage + 1
                        break
                component = "backbone"
                learning_rate = float(args.lr_backbone) * float(args.backbone_layer_decay) ** (4 - layer)
            elif name.startswith("encoder.projections."):
                component, learning_rate = "projections", float(args.lr)
            elif name.startswith(fusion_prefixes):
                component, learning_rate = "fusion", float(args.lr) * float(args.fusion_lr_multiplier)
            else:
                component, learning_rate = "heads", float(args.lr)
            if args.finetune_mode == "fusion_only" and component != "fusion":
                learning_rate = 0.0
            if id(parameter) in included:
                raise RuntimeError(f"Duplicate optimizer parameter {name}.")
            included.add(id(parameter))
            grouped.setdefault((component, learning_rate, decay), []).append(parameter)
        if included != {id(parameter) for parameter in model.parameters() if parameter.requires_grad}:
            raise RuntimeError("Optimizer groups must cover trainable parameters exactly once.")
        optimizer = torch.optim.AdamW([
            {"params": parameters, "name": component, "lr": rate, "weight_decay": decay}
            for (component, rate, decay), parameters in grouped.items()
        ])
        total_steps = max(1, int(args.epochs) * steps_per_epoch)
        frozen_steps = int(args.freeze_backbone_epochs) * steps_per_epoch
        warmup_steps = round(float(args.warmup_epochs) * steps_per_epoch)
        fusion_warmup = round(float(args.fusion_warmup_epochs) * steps_per_epoch)

        def schedule(step: int, component: str) -> float:
            if component in {"backbone", "projections"}:
                if step < frozen_steps:
                    return 0.0
                local_step, duration, warmup = step - frozen_steps, max(1, total_steps - frozen_steps), warmup_steps
            else:
                local_step, duration = step, total_steps
                warmup = fusion_warmup if component == "fusion" else warmup_steps
            if warmup > 0 and local_step < warmup:
                return max(1.0e-3, (local_step + 1) / warmup)
            progress = min(1.0, max(0.0, (local_step - warmup) / max(1, duration - warmup)))
            minimum = float(args.min_lr_ratio)
            return minimum + 0.5 * (1.0 - minimum) * (1.0 + math.cos(math.pi * progress))

        if args.lr_scheduler == "warmup_cosine":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [
                lambda step, component=group["name"]: schedule(step, component)
                for group in optimizer.param_groups
            ])
        else:
            if args.freeze_backbone_epochs:
                raise ValueError("Initial freezing requires warmup_cosine.")
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(args.lr_decay))
        config = {key: getattr(args, key) for key in (
            "lr", "lr_backbone", "weight_decay", "backbone_layer_decay", "fusion_lr_multiplier",
            "finetune_mode", "freeze_backbone_epochs", "lr_scheduler", "warmup_epochs",
            "fusion_warmup_epochs", "min_lr_ratio", "lr_decay", "epochs", "max_grad_norm",
        )}
        config["steps_per_epoch"] = steps_per_epoch
        config["training_runtime"] = {
            name: getattr(args, name, None)
            for name in (
                "batch_train", "seed", "deterministic", "use_amp", "amp_dtype",
                "use_ema", "ema_decay", "ema_update_after_steps",
            )
        }
        if self.rank == 0:
            for group in optimizer.param_groups:
                count = sum(parameter.numel() for parameter in group["params"])
                print(f"[PACE optimizer] {group['name']:<11} lr={group['lr']:.3e} wd={group['weight_decay']:.3e} params={count:,}")
        return optimizer, scheduler, config


    @torch.no_grad()
    def _training_prior(self, device: torch.device) -> float:
        """Scan exactly the original training pair list and resized GT pipeline."""
        foreground_pixels, total_pixels = 0, 0
        scan_error = None
        try:
            for index in range(self.rank, len(self.dataset_train.train_pairs), self.world_size):
                pair = self.dataset_train.train_pairs[index]
                for modality in MODALITIES:
                    label = self.dataset_train.read_label(pair, modality)
                    foreground_pixels += int(np.count_nonzero(label))
                    total_pixels += int(label.size)
        except Exception as error:
            scan_error = f"rank={self.rank}: {type(error).__name__}: {error}"
        counts = torch.tensor((foreground_pixels, total_pixels), device=device, dtype=torch.float64)
        if _distributed():
            errors = [None] * self.world_size
            dist.all_gather_object(errors, scan_error)
            scan_error = next((error for error in errors if error is not None), None)
            dist.all_reduce(counts)
        if scan_error is not None:
            raise RuntimeError(f"Training foreground prior scan failed: {scan_error}")
        if counts[1] <= 0:
            raise RuntimeError("The training GT prior cannot be estimated from an empty split.")
        probability = float((counts[0] / counts[1]).item())
        _unwrap(self.model).p_les.fill_(probability)
        if self.rank == 0:
            print(f"[PACE prior] merged training foreground pixel fraction={probability:.9f}")
            _write_json(self.output_dir / "training_prior.json", {
                "p_les": probability, "foreground_pixels": counts[0], "total_pixels": counts[1],
                "pair_count": len(self.dataset_train), "modalities": list(MODALITIES),
            })
        return probability


    def _augment_spatial(
        self, nbi: torch.Tensor, wli: torch.Tensor,
        masks: Mapping[str, torch.Tensor], epoch: int, step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Sample one exact image/mask transform per pair, without interpolation."""
        if not bool(getattr(self.args, "use_spatial_augmentation", True)):
            return nbi, wli, dict(masks)
        # Uniformly sample the eight square symmetries (including identity).
        # A local CPU RNG avoids device synchronization and is reproducible
        # after epoch-boundary resume independently of the degradation RNG.
        generator = torch.Generator().manual_seed(
            self.seed + 104729 * self.rank + 1000003 * epoch + 15485863 * (step + 1)
        )
        transforms = torch.randint(8, (nbi.shape[0],), generator=generator).tolist()

        def transform_batch(tensor: torch.Tensor) -> torch.Tensor:
            samples = []
            for sample, code in zip(tensor, transforms):
                if code >= 4:
                    sample = sample.flip(-1)
                samples.append(torch.rot90(sample, code % 4, dims=(-2, -1)))
            return torch.stack(samples)

        return transform_batch(nbi), transform_batch(wli), {
            modality: transform_batch(masks[modality]) for modality in MODALITIES
        }


    def _augment(self, nbi: torch.Tensor, wli: torch.Tensor, epoch: int, step: int):
        if self.degrader is None:
            return nbi, wli
        seed = self.quality_seed + 104729 * self.rank + 1000003 * epoch + 2 * step
        nbi_unit, wli_unit = _image_to_unit_interval(nbi), _image_to_unit_interval(wli)
        records = (self.degrader(nbi_unit, seed=seed), self.degrader(wli_unit, seed=seed + 1))
        if getattr(self.args, "quality_degradation_pair_policy", "single_modality") == "single_modality":
            generator = torch.Generator(device=nbi.device).manual_seed(seed + 2)
            choose_nbi = torch.rand((nbi.shape[0], 1, 1, 1), generator=generator, device=nbi.device) < float(
                getattr(self.args, "quality_nbi_degrade_probability", 0.5)
            )
            return torch.where(choose_nbi, records[0]["image"], nbi_unit), torch.where(~choose_nbi, records[1]["image"], wli_unit)
        return records[0]["image"], records[1]["image"]


    @staticmethod
    def _rng_state(generator: torch.Generator) -> dict[str, Any]:
        return {
            "python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "data_generator": generator.get_state(),
        }


    @staticmethod
    def _restore_rng(state: Mapping[str, Any], generator: torch.Generator):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        if state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
        generator.set_state(state["data_generator"].cpu())


    def _save_checkpoint(
        self, path, epoch, step, optimizer, scheduler, scaler, ema, config,
        generator, *, final=False,
    ):
        """Commit an epoch-boundary snapshot atomically; never inspect test data."""
        if final and epoch != TOTAL_EPOCHS:
            raise ValueError("A final checkpoint requires 100 completed training epochs.")
        local_rng = self._rng_state(generator)
        rng_states = [None] * self.world_size if _distributed() else [local_rng]
        if _distributed():
            dist.all_gather_object(rng_states, local_rng)
        error = None
        if self.rank == 0:
            try:
                payload = {
                    "format": CHECKPOINT_FORMAT,
                    "dataset": self.args.dataset,
                    "model": _unwrap(self.model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "ema": ema.state_dict() if ema else None,
                    "epoch": epoch, "completed_epochs": epoch,
                    "global_step": step, "final": bool(final),
                    "selection_policy": "fixed_epoch_100" if final else "epoch_progress",
                    "optimizer_config": config,
                    "training_data": self.training_data,
                    "train_config": self.train_config,
                    "args": dict(vars(self.args)),
                    "world_size": self.world_size,
                    "rng_states": rng_states,
                }
                path = Path(path)
                temporary = path.with_name(path.name + ".tmp")
                torch.save(payload, temporary)
                os.replace(temporary, path)
            except Exception as exception:
                error = f"{type(exception).__name__}: {exception}"
        if _distributed():
            status = [error]
            dist.broadcast_object_list(status, src=0)
            error = status[0]
        if error is not None:
            raise RuntimeError(f"PACE checkpoint write failed: {error}")

    def _iteration_log(self, epoch: int, step: int, global_step: int, values: Mapping[str, Any], device: torch.device):
        if self.rank == 0:
            row = dict(values, epoch=epoch, iteration=step + 1, global_step=global_step, rank=self.rank,
                peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
            with (self.output_dir / "train_iterations.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_jsonable(row), ensure_ascii=False, allow_nan=False) + "\n")


    def __call__(self, args: Any, pin_memory: bool = True):
        self.args = args
        if int(args.epochs) != TOTAL_EPOCHS:
            raise ValueError("PACE training requires exactly 100 epochs.")
        if bool(getattr(args, "inference_only", False)) or bool(getattr(args, "eval_only", False)):
            raise ValueError("Use the independent inference entry point for saved final weights.")
        deterministic = bool(getattr(args, "deterministic", False))
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
        torch.set_float32_matmul_precision("high")
        device = torch.device(f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(self.gpu)
        run_seed = self.seed + self.rank
        random.seed(run_seed)
        np.random.seed(run_seed % (2**32))
        torch.manual_seed(run_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(run_seed)
        self.model.to(device)
        generator = torch.Generator().manual_seed(run_seed)
        sampler = DistributedSampler(self.dataset_train, num_replicas=self.world_size, rank=self.rank, seed=self.seed)
        loader_options = {}
        if int(args.num_workers) > 0:
            loader_options.update(persistent_workers=bool(args.persistent_workers), prefetch_factor=int(args.prefetch_factor))
        loader = DataLoader(
            self.dataset_train, batch_size=int(args.batch_train), sampler=sampler,
            num_workers=int(args.num_workers), pin_memory=pin_memory, worker_init_fn=_seed_data_worker,
            generator=generator, **loader_options,
        )
        model = self.model
        optimizer, scheduler, config = self._optimizer(model, len(loader))
        amp_enabled = device.type == "cuda" and bool(args.use_amp)
        amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
        bf16_supported = torch.tensor(int(device.type == "cuda" and torch.cuda.is_bf16_supported()), device=device)
        if _distributed():
            dist.all_reduce(bf16_supported, op=dist.ReduceOp.MIN)
        bf16_available = bool(bf16_supported.item())
        if amp_enabled and amp_dtype == torch.bfloat16 and not bf16_available:
            raise RuntimeError("BF16 AMP requires BF16 support on every rank; select --amp_dtype float16.")
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16,
            init_scale=float(args.amp_init_scale), growth_interval=int(args.amp_growth_interval))
        full_resume = getattr(args, "resume_checkpoint", None)
        weights = getattr(args, "resume_ckpt", None)
        checkpoint = None
        if full_resume or weights:
            checkpoint = torch.load(full_resume or weights, map_location="cpu", weights_only=False)
            state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
            source = getattr(args, "weights_source", "auto")
            if weights and isinstance(checkpoint, Mapping) and source in {"auto", "ema"}:
                ema_state = checkpoint.get("ema")
                if isinstance(ema_state, Mapping) and ema_state.get("initialized", True):
                    state = ema_state["model"]
                elif source == "ema":
                    raise ValueError("--weights_source ema requires initialized EMA weights.")
            loaded = model.load_state_dict(state, strict=True if full_resume else bool(getattr(args, "weights_strict", True)))
            if not full_resume and "p_les" in loaded.missing_keys:
                self._training_prior(device)
        else:
            self._training_prior(device)
        ema = _ModelEMA(model, float(args.ema_decay), int(args.ema_update_after_steps)) if bool(args.use_ema) else None
        start_epoch, global_step = 1, 0
        if full_resume:
            if checkpoint.get("format") != CHECKPOINT_FORMAT:
                raise ValueError("Full resume requires a PACE_opensource_v1 checkpoint; use --weights for a model-only warm start.")
            if checkpoint.get("training_data") != self.training_data:
                raise ValueError("Full resume dataset, root or training pairs differ from checkpoint.")
            saved_epoch = int(checkpoint.get("completed_epochs", checkpoint["epoch"]))
            if saved_epoch != int(checkpoint["epoch"]) or not 0 <= saved_epoch <= TOTAL_EPOCHS:
                raise ValueError("Checkpoint must contain 0..100 fully completed epochs.")
            saved_config = checkpoint.get("train_config")
            if not isinstance(saved_config, Mapping):
                raise ValueError("Full resume requires a checkpoint with train_config; use --weights for a warm start.")
            if saved_config != self.train_config:
                changed = sorted(name for name in set(saved_config) | set(self.train_config)
                                 if saved_config.get(name) != self.train_config.get(name))
                raise ValueError("Full resume training configuration differs from checkpoint: " + ", ".join(changed))
            if checkpoint.get("optimizer_config") != config or checkpoint.get("world_size") != self.world_size:
                raise ValueError("Full resume optimizer/scheduler/world-size configuration differs from checkpoint.")
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            if ema is not None:
                if checkpoint.get("ema") is None:
                    raise ValueError("Full resume requested EMA but checkpoint has no EMA state.")
                ema.load_state_dict(checkpoint["ema"])
            start_epoch, global_step = int(checkpoint["epoch"]) + 1, int(checkpoint["global_step"])
            self._restore_rng(checkpoint["rng_states"][self.rank], generator)
        # Keep only the live model/optimizer state during training; a full
        # resume payload also contains a second copy of all parameter tensors.
        checkpoint, state, ema_state = None, None, None
        if self.world_size > 1:
            self.model = DDP(model, device_ids=[self.gpu] if device.type == "cuda" else None,
                find_unused_parameters=bool(getattr(args, "ddp_find_unused_parameters", False)))
        if self.rank == 0:
            _write_json(self.output_dir / "config.json", vars(args))
        smoke_steps = int(getattr(args, "smoke_steps", 0))
        smoke_history = []
        consecutive_overflows = 0
        for epoch in range(start_epoch, int(args.epochs) + 1):
            self.current_epoch = epoch
            sampler.set_epoch(epoch)
            self.model.train()
            if self.degrader is not None:
                duration = float(getattr(args, "quality_curriculum_epochs", 10.0))
                curriculum = min(1.0, epoch / duration) if duration > 0 else 1.0
                self.degrader.apply_probability = self._quality_base_probability * (0.25 + 0.75 * curriculum)
                self.degrader.max_severity = self.degrader.min_severity + curriculum * (self._quality_base_severity - self.degrader.min_severity)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            epoch_sums: dict[str, float] = {}
            epoch_steps = 0
            with _EpochProgress(self.rank == 0) as progress:
                for step, batch in enumerate(loader):
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    nbi = batch["nbi_img"].to(device=device, dtype=torch.float32, non_blocking=True)
                    wli = batch["wli_img"].to(device=device, dtype=torch.float32, non_blocking=True)
                    masks = {modality: normalize_binary_mask(batch[modality + "_lbl"]).to(device, non_blocking=True) for modality in MODALITIES}
                    nbi, wli, masks = self._augment_spatial(nbi, wli, masks, epoch, step)
                    nbi, wli = self._augment(nbi, wli, epoch, step)
                    optimizer.zero_grad(set_to_none=True)
                    prior_before = (model.audit.mu.clone(), model.audit.s0_sq.clone())
                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                        outputs = self.model(nbi, wli)
                        loss, losses = self.compute_loss(outputs, masks["nbi"], masks["wli"], (epoch - 1) + (step + 1) / max(1, len(loader)))
                    finite = torch.isfinite(loss.detach()).to(dtype=torch.int32)
                    if _distributed():
                        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    recovered_bf16 = False
                    if not bool(finite.item()) and amp_enabled and amp_dtype == torch.float16 and bf16_available:
                        # Preserve the reference FP16-to-BF16 forward retry. A
                        # failed attempt must not update the audit prior twice.
                        del outputs, loss, losses
                        model.audit.mu.copy_(prior_before[0])
                        model.audit.s0_sq.copy_(prior_before[1])
                        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
                            outputs = self.model(nbi, wli)
                            loss, losses = self.compute_loss(outputs, masks["nbi"], masks["wli"], (epoch - 1) + (step + 1) / max(1, len(loader)))
                        finite = torch.isfinite(loss.detach()).to(dtype=torch.int32)
                        if _distributed():
                            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                        recovered_bf16 = bool(finite.item())
                    if not bool(finite.item()):
                        raise FloatingPointError(f"PACE produced non-finite loss at epoch={epoch} step={step}.")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
                    finite_gradient = torch.isfinite(gradient_norm).to(dtype=torch.int32)
                    if _distributed():
                        dist.all_reduce(finite_gradient, op=dist.ReduceOp.MIN)
                    if not bool(finite_gradient.item()):
                        optimizer.zero_grad(set_to_none=True)
                        if scaler.is_enabled():
                            consecutive_overflows += 1
                            minimum_scale = float(getattr(args, "amp_min_scale", 1.0))
                            previous_scale = scaler.get_scale()
                            scaler.update(new_scale=max(minimum_scale, previous_scale * 0.5))
                            model.audit.mu.copy_(prior_before[0])
                            model.audit.s0_sq.copy_(prior_before[1])
                            self._iteration_log(epoch, step, global_step, {
                                "skipped_amp_overflow": True, "iteration_seconds": time.perf_counter() - started,
                                "amp_scale": scaler.get_scale(), "grad_norm": float(gradient_norm),
                            }, device)
                            del outputs, loss, losses
                            if consecutive_overflows <= int(getattr(args, "amp_max_consecutive_overflows", 16)) and previous_scale > minimum_scale:
                                continue
                        raise FloatingPointError(f"PACE produced non-finite gradients at epoch={epoch} step={step}.")
                    consecutive_overflows = 0
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    succeeded = not scaler.is_enabled() or scaler.get_scale() >= scale_before
                    if succeeded:
                        global_step += 1
                        if args.lr_scheduler == "warmup_cosine":
                            scheduler.step()
                        if ema is not None:
                            ema.update(model)
                            # The empirical Bayes EMA is already the calibration
                            # estimate; weight EMA must not smooth those buffers again.
                            ema.model.audit.mu.copy_(model.audit.mu)
                            ema.model.audit.s0_sq.copy_(model.audit.s0_sq)
                            ema.model.p_les.copy_(model.p_les)
                    optimizer.zero_grad(set_to_none=True)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - started
                    values = {key: float(value) for key, value in losses.items()}
                    values.update(rho=float(outputs["rho"].mean()), alpha=float(outputs["alpha"].mean()),
                        t=0.5 * sum(float(outputs["audit_" + modality]["t"].mean()) for modality in MODALITIES),
                        fallback_rate=0.5 * sum(float(outputs["audit_" + modality]["fallback"].float().mean()) for modality in MODALITIES),
                        iteration_seconds=elapsed, grad_norm=float(gradient_norm), bf16_forward_recovery=float(recovered_bf16))
                    self._iteration_log(epoch, step, global_step, values, device)
                    for key, value in values.items():
                        epoch_sums[key] = epoch_sums.get(key, 0.0) + value
                    epoch_steps += 1
                    if smoke_steps:
                        smoke_history.append({"step": global_step, **values})
                    progress.update(
                        f"[PACE] epoch={epoch}/{args.epochs} step={step + 1}/{len(loader)} "
                        f"loss={values['total']:.4f} avg={epoch_sums['total'] / epoch_steps:.4f} "
                        f"t={values['t']:.3f} {elapsed:.2f}s"
                    )
                    del outputs, loss, losses
                    if smoke_steps and len(smoke_history) >= smoke_steps:
                        break
            if args.lr_scheduler == "exponential":
                scheduler.step()
            if epoch_steps == 0:
                raise FloatingPointError("PACE epoch completed without any finite optimizer updates.")
            keys = sorted(epoch_sums)
            statistics = torch.tensor([epoch_sums[key] for key in keys] + [epoch_steps], device=device, dtype=torch.float64)
            peak_memory = torch.tensor(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0, device=device, dtype=torch.float64)
            if _distributed():
                dist.all_reduce(statistics)
                dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
            epoch_log = {key: float(statistics[index] / statistics[-1]) for index, key in enumerate(keys)}
            epoch_log.update(epoch=epoch, global_step=global_step, mu=float(model.audit.mu), s0_sq=float(model.audit.s0_sq),
                peak_memory_bytes=int(peak_memory), p_les=float(model.p_les))
            if self.rank == 0:
                with (self.output_dir / "train_epochs.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(_jsonable(epoch_log), ensure_ascii=False, allow_nan=False) + "\n")
                print(
                    f"[PACE train] epoch={epoch}/{args.epochs} loss={epoch_log['total']:.6f} "
                    f"t={epoch_log['t']:.4f} rho={epoch_log['rho']:.4f} "
                    f"alpha={epoch_log['alpha']:.4f} {epoch_log['iteration_seconds']:.3f}s/step",
                    flush=True,
                )
            if smoke_steps and len(smoke_history) >= smoke_steps:
                smoke = {"steps": len(smoke_history), "initial_loss": smoke_history[0]["total"], "final_loss": smoke_history[-1]["total"],
                    "loss_decreased": smoke_history[-1]["total"] < smoke_history[0]["total"], "mean_iteration_seconds": float(np.mean([row["iteration_seconds"] for row in smoke_history])),
                    "peak_memory_bytes": int(peak_memory), "history": smoke_history}
                if self.rank == 0:
                    _write_json(self.output_dir / "smoke_training.json", smoke)
                return smoke
            self._save_checkpoint(
                self.output_dir / "PACE_last.pth", epoch, global_step,
                optimizer, scheduler, scaler, ema, config, generator,
            )
            interval = int(getattr(args, "checkpoint_interval", 0))
            if interval > 0 and epoch % interval == 0 and epoch < TOTAL_EPOCHS:
                self._save_checkpoint(
                    self.output_dir / f"PACE_epoch_{epoch:03d}.pth", epoch,
                    global_step, optimizer, scheduler, scaler, ema, config, generator,
                )

        if smoke_steps:
            raise RuntimeError("The smoke-step limit exceeded the remaining schedule; no test was run.")
        final_path = self.output_dir / "PACE_final.pth"
        self._save_checkpoint(
            final_path, TOTAL_EPOCHS, global_step, optimizer, scheduler, scaler,
            ema, config, generator, final=True,
        )
        # The test split is first opened by run_inference, after this committed
        # epoch-100 snapshot. Reload it to make the exact weight source explicit.
        del checkpoint
        del optimizer, scheduler, scaler
        if __package__:
            from .inference import load_final_checkpoint, select_checkpoint_state, run_inference
        else:
            from inference import load_final_checkpoint, select_checkpoint_state, run_inference
        final_checkpoint = load_final_checkpoint(final_path, args.dataset)
        state, source = select_checkpoint_state(
            final_checkpoint, getattr(args, "weights_source", "auto"),
            ema_eval=bool(getattr(args, "ema_eval", True)),
        )
        model.load_state_dict(state, strict=True)
        del state, final_checkpoint, ema
        model.eval()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if self.rank == 0:
            print(f"[PACE] Training completed: epoch=100; inference weights={source}, checkpoint={final_path}", flush=True)
        return run_inference(
            model, args, device, rank=self.rank, world_size=self.world_size,
            epoch=TOTAL_EPOCHS, weights_source=source,
            checkpoint_path=os.fspath(final_path),
        )

