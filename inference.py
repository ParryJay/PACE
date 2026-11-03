"""Final-checkpoint inference without ground truth, scores, or visualization."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

if __package__:
    from .data import PairedDataset
    from .model import build_model
else:
    from data import PairedDataset
    from model import build_model


def load_final_checkpoint(path: str | Path, dataset: str) -> Mapping[str, Any]:
    """Only checkpoints produced after the complete training schedule may test."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Inference requires a complete PACE_final.pth checkpoint.")
    if (
        checkpoint.get("format") != "PACE_opensource_v1"
        or checkpoint.get("epoch") != 100
        or checkpoint.get("completed_epochs") != 100
        or checkpoint.get("final") is not True
    ):
        raise ValueError("Test inference requires a final checkpoint saved after 100 completed epochs.")
    if checkpoint.get("dataset") != dataset:
        raise ValueError(
            f"Checkpoint dataset {checkpoint.get('dataset')!r} differs from --dataset {dataset!r}."
        )
    if not isinstance(checkpoint.get("args"), Mapping) or "model" not in checkpoint:
        raise ValueError("The final checkpoint is missing its model or configuration.")
    return checkpoint


def select_checkpoint_state(
    checkpoint: Mapping[str, Any], source: str = "auto", *, ema_eval: bool = True,
) -> tuple[Mapping[str, torch.Tensor], str]:
    """Choose the predetermined online/EMA weights from the same final epoch."""
    if source not in {"auto", "online", "ema"}:
        raise ValueError("weights_source must be auto, online, or ema.")
    ema = checkpoint.get("ema")
    available = isinstance(ema, Mapping) and bool(ema.get("initialized", False)) and "model" in ema
    if source == "ema" and not available:
        raise ValueError("--weights_source ema requires initialized EMA weights in the checkpoint.")
    if source == "ema" or (source == "auto" and ema_eval and available):
        return ema["model"], "ema"
    return checkpoint["model"], "online"


@torch.inference_mode()
def run_inference(
    model: torch.nn.Module,
    args: Any,
    device: torch.device,
    *,
    rank: int = 0,
    world_size: int = 1,
    epoch: int = 100,
    weights_source: str = "online",
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read the test split for the first time, then infer each pair exactly once.

    Ranks use disjoint indices without DistributedSampler padding. The model
    must be unwrapped from DDP because ranks may have different batch counts.
    Optional NPZ outputs contain numeric predictions, never rendered images.
    """
    if epoch != 100:
        raise ValueError("Test inference runs only after epoch 100.")
    if not 0 <= rank < world_size:
        raise ValueError("Invalid inference rank/world_size.")
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        raise ValueError("Inference requires an unwrapped model for unequal rank shards.")
    dataset = PairedDataset(
        args.dataset, args.dataset_root, getattr(args, "test_manifest", None),
        split="test", img_size=int(args.img_size),
    )
    loader = DataLoader(
        Subset(dataset, range(rank, len(dataset), world_size)),
        batch_size=int(args.batch_test), shuffle=False,
        num_workers=int(args.test_num_workers), pin_memory=bool(args.pin_memory),
        generator=torch.Generator().manual_seed(int(args.seed) + rank),
    )
    model.eval()
    amp_enabled = device.type == "cuda" and bool(args.use_amp)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    if amp_enabled and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 inference requires BF16 support; select --amp_dtype float16.")
    save = bool(getattr(args, "save_predictions", False))
    prediction_dir = Path(args.output_dir) / "predictions"
    if save:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for batch in loader:
        nbi = batch["nbi_img"].to(device=device, dtype=torch.float32, non_blocking=True)
        wli = batch["wli_img"].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(nbi, wli)
        # Producing a prediction is independent of whether it is written to disk.
        probabilities = {
            modality: outputs["logits_final_" + modality].float().softmax(dim=1)
            for modality in ("nbi", "wli")
        }
        masks = {modality: probability.argmax(dim=1) for modality, probability in probabilities.items()}
        if save:
            for offset, pair_index in enumerate(batch["pair_index"].tolist()):
                pair = dataset.pairs[pair_index]
                values = {
                    "pair_index": np.asarray(pair_index, dtype=np.int64),
                    "nbi_img": np.asarray(pair["nbi_img"]),
                    "wli_img": np.asarray(pair["wli_img"]),
                }
                for modality in ("nbi", "wli"):
                    values[modality + "_probability"] = probabilities[modality][offset, 1].cpu().numpy()
                    values[modality + "_mask"] = masks[modality][offset].to(torch.uint8).cpu().numpy()
                destination = prediction_dir / f"{pair_index:08d}.npz"
                temporary = destination.with_suffix(".npz.tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, **values)
                temporary.replace(destination)
        count += nbi.shape[0]
        del outputs, probabilities, masks
    total = torch.tensor(count, device=device, dtype=torch.int64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    if int(total) != len(dataset):
        raise RuntimeError(f"Inference processed {int(total)} pairs; expected {len(dataset)}.")
    result = {
        "dataset": args.dataset, "completed_epochs": epoch,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "weights_source": weights_source, "pair_count": int(total),
        "predictions_saved": save,
    }
    if rank == 0:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        temporary = output / "inference_run.json.tmp"
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "inference_run.json")
        print(f"[PACE inference] dataset={args.dataset} epoch=100 pairs={int(total)} source={weights_source}", flush=True)
    return result


def run_checkpoint_inference(
    args: Any, device: torch.device, *, rank: int = 0, world_size: int = 1,
) -> dict[str, Any]:
    """Standalone inference restores the architecture recorded at training time."""
    checkpoint_path = args.resume_ckpt
    checkpoint = load_final_checkpoint(checkpoint_path, args.dataset)
    state, source = select_checkpoint_state(
        checkpoint, getattr(args, "weights_source", "auto"),
        ema_eval=bool(checkpoint["args"].get("ema_eval", True)),
    )
    model = build_model(checkpoint["args"], pretrained=False).to(device)
    model.load_state_dict(state, strict=True)
    del checkpoint, state
    return run_inference(
        model, args, device, rank=rank, world_size=world_size,
        epoch=100, weights_source=source, checkpoint_path=checkpoint_path,
    )
