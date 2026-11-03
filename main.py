"""Unified PACE entry point and single-/multi-process launcher."""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
from typing import Sequence
import warnings

if __package__:
    from .config import build_parser, normalize_and_validate_args
else:
    from config import build_parser, normalize_and_validate_args


def _resolve_launch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> bool:
    """Resolve visibility before importing PyTorch; return whether torchrun owns launch."""
    rank_present = any(name in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK")) or args.local_rank is not None
    if rank_present:
        if not all(name in os.environ for name in ("RANK", "WORLD_SIZE")) or ("LOCAL_RANK" not in os.environ and args.local_rank is None):
            parser.error("RANK, WORLD_SIZE and LOCAL_RANK must be supplied together by torchrun.")
        try:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank)))
            args.gpus = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        except ValueError:
            parser.error("torchrun rank and world size values must be integers.")
        if args.world_size < 1 or args.gpus < 1 or not 0 <= args.rank < args.world_size or not 0 <= args.local_rank < args.gpus:
            parser.error("Invalid torchrun rank or world size.")
        args.cpu = bool(args.devices and args.devices.strip().lower() == "cpu")
        if args.cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        elif args.devices is not None:
            warnings.warn("Under torchrun, set CUDA_VISIBLE_DEVICES before launch; --devices is ignored.", stacklevel=2)
        args.devices = "cpu" if args.cpu else os.environ.get("CUDA_VISIBLE_DEVICES")
        args.nr = int(os.environ.get("GROUP_RANK", str(args.rank // args.gpus)))
        args.nodes = int(os.environ.get("GROUP_WORLD_SIZE", str(math.ceil(args.world_size / args.gpus))))
        args.launcher = "torchrun"
        return True

    raw_devices = args.devices if args.devices is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    args.cpu = raw_devices.strip().lower() == "cpu" or not raw_devices.strip()
    if args.cpu:
        if args.dist_backend != "gloo":
            parser.error("CPU execution requires --dist_backend gloo.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        args.devices = "cpu"
        args.gpus = 1 if args.gpus is None else args.gpus
    else:
        tokens = tuple(token.strip() for token in raw_devices.split(","))
        if not tokens or any(not token for token in tokens) or len(set(tokens)) != len(tokens):
            parser.error("--devices must contain distinct comma-separated GPU IDs.")
        args.devices = ",".join(tokens)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
        args.gpus = len(tokens) if args.gpus is None else args.gpus
        if args.gpus > len(tokens):
            parser.error("--gpus exceeds the number of visible devices.")
    if args.gpus < 1 or not 0 <= args.nr < args.nodes:
        parser.error("Require gpus >= 1 and 0 <= node rank < nodes.")
    if not 1 <= args.master_port <= 65535:
        parser.error("--master_port must be in [1,65535].")
    if args.nodes > 1 and args.master_addr in {"localhost", "127.0.0.1", "::1"}:
        parser.error("Multi-node launch needs a non-loopback --master_addr.")
    args.launcher, args.rank, args.local_rank = "builtin", None, None
    args.world_size = args.nodes * args.gpus
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    return False


def _seed_everything(seed: int, deterministic: bool) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def _worker(local_rank: int, args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist

    rank = int(args.rank) if args.launcher == "torchrun" else int(args.nr) * int(args.gpus) + local_rank
    args.rank, args.local_rank = rank, local_rank
    use_cuda = not args.cpu and torch.cuda.is_available()
    if args.dist_backend == "nccl" and not use_cuda:
        raise RuntimeError("NCCL requires CUDA; use --devices cpu --dist_backend gloo for CPU execution.")
    if use_cuda:
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(f"Local rank {local_rank} exceeds visible CUDA devices.")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    initialized = False
    try:
        if args.world_size > 1:
            dist.init_process_group(backend=args.dist_backend, init_method="env://",
                                    world_size=args.world_size, rank=rank,
                                    timeout=timedelta(minutes=args.dist_timeout_minutes))
            initialized = True
            shared_output = [args.output_dir if rank == 0 else None]
            dist.broadcast_object_list(shared_output, src=0)
            args.output_dir = str(shared_output[0])
            args.checkpoint_dir = args.output_dir
        _seed_everything(args.seed, args.deterministic)
        if rank == 0:
            output = Path(args.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            temporary = output / ".config.tmp"
            temporary.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, output / "config.json")
            print(f"[PACE run] dataset={args.dataset} epochs=100 world_size={args.world_size} output={args.output_dir}", flush=True)
            print("[PACE protocol] No test access during training; final checkpoint inference after epoch 100 only.", flush=True)
        if initialized:
            dist.barrier()
        if args.inference_only:
            if __package__:
                from .inference import run_checkpoint_inference
            else:
                from inference import run_checkpoint_inference
            run_checkpoint_inference(args, device, rank=rank, world_size=args.world_size)
        else:
            if __package__:
                from .trainer import Trainer
            else:
                from trainer import Trainer
            trainer = Trainer(gpu=local_rank, rank=rank, world_size=args.world_size, args=args)
            trainer(args, pin_memory=args.pin_memory)
    finally:
        if initialized and dist.is_initialized():
            dist.destroy_process_group()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = normalize_and_validate_args(parser, parser.parse_args(argv))
    torchrun = _resolve_launch(parser, args)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if torchrun:
        _worker(int(args.local_rank), args)
    elif args.gpus == 1:
        _worker(0, args)
    else:
        import torch.multiprocessing as mp
        mp.spawn(_worker, args=(args,), nprocs=args.gpus, join=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
