<div align="center">

# PACE: Partial-Correspondence Auditing and Calibrated Evidential Fusion for Unregistered WLI/NBI Lesion Segmentation

**Pengyu Jie, Wanquan Liu<sup>†</sup>, Rui He, Yihui Wen, Pengcheng Li, Weiping Wen, Deyu Meng, Junwei Han, Chenqiang Gao<sup>†</sup>**

<sup>†</sup> Corresponding authors

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TMI-blue)](#citation)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-TBD-lightgrey)](#license)

English | [简体中文](README_zh-CN.md)

</div>

This repository contains the official PyTorch implementation of **PACE**, a fusion framework for lesion segmentation from **unregistered** white-light imaging (WLI) and narrow-band imaging (NBI) endoscopic image pairs.

<!-- Add the framework figure (Fig. 2 of the paper) as assets/framework.png, then uncomment:
<p align="center"><img src="assets/framework.png" width="90%" alt="Overview of PACE"></p>
-->

## Table of Contents

- [Overview](#overview)
- [Main Results](#main-results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Key Hyperparameters](#key-hyperparameters)
- [Running Tests](#running-tests)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Contact](#contact)

## Overview

WLI and NBI are acquired sequentially by switching the light source, so viewpoint change, tissue deformation and occlusion leave every image pair unregistered and only partially overlapping. Fusion must rely on an *estimated* cross-modal correspondence. Its errors are hard to catch: a mismatched message comes from another real region and is as confident as a correct one, so uncertainty-aware fusion cannot detect it, and fusing it can cause negative transfer.

PACE **audits** the correspondence of each image pair instead of assuming it is correct, and calibrates fusion with the audited reliability:

| Component | What it does |
| --- | --- |
| **Lesion-Centric Evidential Graph (LCEG)** | Represents each view as a graph of soft, lesion-guided regions. Edges depend only on relative distances (invariant to in-plane translation/rotation), and each node carries a subjective-logic opinion (lesion belief, background belief, uncertainty). |
| **Adaptive Partial Fused Gromov–Wasserstein (PFGW) transport** | Matches the two graphs with a pair-specific transport budget ρ and structure weight α. Unmatched mass is absorbed by dummy nodes, so regions without a counterpart enter fusion as *unknown* rather than as evidence. |
| **Correspondence audit** | On confidently co-observed nodes, measures how far cross-modal agreement exceeds chance (Cohen's κ) and turns it into an upper bound on the pair's contamination rate, η ≤ 1 − κ. No correspondence ground truth is needed, and the audit has no learnable parameters. |
| **Audit-calibrated fusion** | The reliability *t* = 1 − η̂⁺ drives **Shared–Private–Reject (TSPR)** routing at three feature levels and a **robust decision-level likelihood** based on Huber's contamination model. |

**Theoretical guarantees.** If contaminated messages agree with the target only at chance, chance-corrected agreement gives a *sharp* upper bound on the contamination rate. Overestimating this rate only moves fusion toward single-modality prediction, so audit-calibrated decision-level fusion is asymptotically no worse than single-modality prediction.

## Main Results

Mean ± std over four runs. All methods use the same patient-level splits, input resolution (512 × 512) and evaluation protocol. See Table I of the paper for all 11 baselines.

| Dataset | Method | mIoU (%) | DSC (%) | PA (%) | Recall (%) | Precision (%) |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| PICCOLO (Paired) | ADFNet (MIA 2026) | 80.52 ± 0.81 | 82.48 ± 0.30 | 94.67 ± 0.43 | 84.90 ± 0.40 | 80.20 ± 0.44 |
| | RGSF (arXiv 2026) | 83.54 ± 0.31 | 84.78 ± 0.36 | 94.49 ± 0.26 | 89.85 ± 0.75 | 80.25 ± 0.24 |
| | **PACE (Ours)** | **87.08 ± 0.13** | **89.22 ± 0.13** | **95.88 ± 0.16** | **93.87 ± 0.44** | **85.01 ± 0.47** |
| FAHSYSU (Paired) | ADFNet (MIA 2026) | 79.72 ± 0.27 | 80.72 ± 0.19 | 92.96 ± 0.33 | 80.79 ± 0.28 | 80.66 ± 0.26 |
| | RGSF (arXiv 2026) | 81.08 ± 0.23 | 82.37 ± 0.17 | 93.40 ± 0.30 | 83.68 ± 0.25 | 81.11 ± 0.22 |
| | **PACE (Ours)** | **82.36 ± 0.09** | **83.44 ± 0.24** | **93.88 ± 0.11** | **85.72 ± 0.35** | **81.27 ± 0.29** |

PACE also stays stable when only the auxiliary view is perturbed (translation 20%, rotation 30°, scaling ±30%, 50% field-of-view crop): on FAHSYSU, mean DSC drops by at most 0.045 pp at the severe level.

## Repository Structure

```text
PACE/
├── main.py              # Entry point: single-process / multi-GPU (DDP) / torchrun launcher
├── config.py            # All command-line arguments, defaults and validation
├── data.py              # Paired WLI/NBI manifests and preprocessing (FAHSYSU / PICCOLO)
├── model.py             # PACE model: LCEG, PFGW solver, audit, TSPR routing, decision fusion
├── trainer.py           # Training loop, losses, optimizer groups, EMA, checkpointing
├── augmentation.py      # Synthetic image-quality degradation for training
├── inference.py         # Final-checkpoint validation and label-free inference
├── requirements.txt     # pip dependencies
├── requirement.yaml     # conda environment
└── tests/               # Regression tests with tiny models and temporary data
```

## Installation

Tested on Linux with Python 3.11 and NVIDIA GPUs (the paper's experiments used 4 × RTX 3090).

**Option A: conda**

```bash
git clone https://github.com/ParryJay/PACE.git
cd PACE
conda env create -f requirement.yaml
conda activate mmmv-pace
```

**Option B: pip**

```bash
git clone https://github.com/ParryJay/PACE.git
cd PACE
conda create -n pace python=3.11 -y && conda activate pace
pip install -r requirements.txt
```

Check the installation:

```bash
python -B main.py --help
```

The default backbone is `convnextv2_base` from [timm](https://github.com/huggingface/pytorch-image-models). With `--encoder_pretrained true` (the default), the ImageNet-pretrained weights are downloaded on first use. Offline machines can pre-populate the timm/Hugging Face cache or train from scratch with `--encoder_pretrained false`. Resuming from a full checkpoint and standalone inference never download backbone weights.

## Data Preparation

Each sample is a pair of separately acquired, **unregistered** WLI and NBI images of the same lesion from one patient, with one binary mask per modality. The code uses your fixed train/test pair lists and does **not** re-split the data. Splits should be made at patient level.

| Dataset | Pairs (train / test) | Availability |
| --- | --- | --- |
| FAHSYSU (Paired) | 14,529 (11,512 / 3,017) | In-house dataset, The First Affiliated Hospital of Sun Yat-sen University. Please contact the corresponding authors about access. |
| PICCOLO (Paired) | 7,362 (5,529 / 1,833) | Public dataset: Sánchez-Peralta *et al.*, *Applied Sciences*, 2020 ([doi:10.3390/app10238501](https://doi.org/10.3390/app10238501)). |

### FAHSYSU layout

```text
FAHSYSU_ROOT/
├── pairs_train.json
├── pairs_test.json
├── nbi/
│   ├── img/
│   └── lbl/
└── wli/
    ├── img/
    └── lbl/
```

Manifest (JSON array or `.jsonl`):

```json
[
  {"id": "000001", "nbi_img": "nbi.jpg", "nbi_lbl": "nbi.png", "wli_img": "wli.jpg", "wli_lbl": "wli.png"}
]
```

### PICCOLO layout

```text
PICCOLO_ROOT/
├── train.jsonl
├── test.jsonl
├── img/
│   ├── NBI/
│   └── WLI/
└── lbl/
```

Manifest (one pair per line):

```json
{"id": "001", "nbi": "nbi.png", "wli": "wli.png"}
```

### Notes

- Field aliases: `nbi` / `nbi_img` for NBI images and `wli` / `wli_img` / `wht_img` for WLI images. If `nbi_lbl` / `wli_lbl` are omitted, the label file name defaults to the image file name.
- Both datasets accept JSON arrays and `.jsonl`. Use `--train_manifest` / `--test_manifest` to point to manifests stored elsewhere. Relative manifest paths are resolved against `--dataset_root`. Relative image and label names are resolved against the modality directories above, and absolute paths are also accepted.
- Test manifests only need the two image fields. Label fields in a test manifest are ignored.
- Preprocessing: images are resized to 512 × 512 with bilinear interpolation. Masks are read, resized bilinearly, converted to grayscale and binarized at a threshold of 100.

## Training

Run all commands from the repository root. Batch sizes are **per process (per GPU)**.

**Single GPU**

```bash
# FAHSYSU
python -B main.py --dataset FAHSYSU --dataset_root /path/to/FAHSYSU_ROOT --devices 0

# PICCOLO
python -B main.py --dataset PICCOLO --dataset_root /path/to/PICCOLO_ROOT --devices 0
```

**Multiple GPUs**

```bash
# Built-in launcher (spawns one process per GPU)
python -B main.py --dataset PICCOLO --dataset_root /path/to/PICCOLO_ROOT --devices 0,1,2,3

# Or torchrun
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  main.py --dataset PICCOLO --dataset_root /path/to/PICCOLO_ROOT
```

**Training protocol**

- Training always runs for **100 epochs** (`--epochs` must be 100) at **512 × 512** (`--img_size` must be 512).
- The test manifest is never opened during training. There is no mid-training evaluation and no "best checkpoint" selection.
- After epoch 100, `PACE_final.pth` is saved and reloaded, and a single inference pass over the test set is run. By default the EMA weights are used. Pass `--ema_eval false` or `--weights_source online` to use the online weights instead.
- Mixed precision (`bfloat16`) is on by default on CUDA. Use `--amp_dtype float16` on GPUs without bf16 support.
- For a quick CPU check: `--devices cpu --dist_backend gloo --smoke_steps 2`. `--smoke_steps` stops training early and never runs testing or writes a final checkpoint.

**Outputs** (default `runs/<dataset>_PACE_<timestamp>/`, override with `--output_dir`)

| File | Description |
| --- | --- |
| `PACE_last.pth` | Updated after each epoch. Holds model, optimizer, scheduler, scaler, EMA, epoch and per-rank RNG states for resuming. |
| `PACE_epoch_XXX.pth` | Extra periodic checkpoints, written only when `--checkpoint_interval > 0`. |
| `PACE_final.pth` | Final weights after 100 epochs (marked `final=true`), used for inference. |
| `config.json` | Full resolved configuration. |
| `training_prior.json` | Foreground prior estimated from the training set (the lesion prior *p*<sub>les</sub>). |
| `train_iterations.jsonl`, `train_epochs.jsonl` | Training logs. |

**Resume training**

```bash
python -B main.py --dataset PICCOLO --dataset_root /path/to/PICCOLO_ROOT \
  --resume /path/to/run/PACE_last.pth --output_dir /path/to/run --devices 0
```

Keep the hyperparameters, dataset root, manifests and number of processes the same as the original run. `--weights` loads model weights only (warm start). It does not restore the training state.

## Inference

Run label-free inference from a finished 100-epoch checkpoint:

```bash
python -B main.py --dataset PICCOLO --dataset_root /path/to/PICCOLO_ROOT \
  --inference_only --weights /path/to/run/PACE_final.pth \
  --devices 0 --save_predictions
```

- Only a `PACE_final.pth` produced by this code is accepted. Intermediate checkpoints and checkpoints from a different dataset are rejected.
- The model architecture and algorithm parameters are restored from the checkpoint. You only need to specify the data location, devices and output options.
- `--weights_source auto|online|ema` selects the weights. `auto` follows the `ema_eval` setting stored in the checkpoint.
- Run information is written to `inference_run.json`. With `--save_predictions`, each pair is saved as `predictions/XXXXXXXX.npz`:

| Key | Type | Description |
| --- | --- | --- |
| `pair_index`, `nbi_img`, `wli_img` | – | Pair index and image file names |
| `nbi_probability`, `wli_probability` | `float32`, 512 × 512 | Foreground probability |
| `nbi_mask`, `wli_mask` | `uint8`, 512 × 512 | Binary prediction (0/1) |

With multiple processes, test samples are sharded without padding or duplication.

## Evaluation

The inference script outputs raw predictions only. It does not compute metrics or produce visualizations. To reproduce the paper's numbers, compare each modality's prediction with its own ground-truth mask (no registration or mask warping) at a threshold of 0.5, and report:

- **mIoU**: IoU averaged over foreground and background
- **DSC** = 2 · Precision · Recall / (Precision + Recall)
- **PA** = (TP + TN) / (TP + TN + FP + FN)
- **Recall** = TP / (TP + FN), **Precision** = TP / (TP + FP)

In the paper, each pair has equal weight, including pairs that share a source image.

## Key Hyperparameters

Paper symbols and the corresponding command-line arguments (defaults shown). Run `python -B main.py --help` for the full list.

| Paper | Argument | Default | Meaning |
| --- | --- | --- | --- |
| *N* | `--num_tokens` | 128 | Graph nodes per modality |
| [ρ<sub>min</sub>, ρ<sub>max</sub>] | `--rho_min`, `--rho_max` | 0.2, 0.85 | Range of the pair-specific transport budget |
| [α<sub>min</sub>, α<sub>max</sub>] | `--alpha_min`, `--alpha_max` | 0.05, 0.35 | Range of the FGW structure weight |
| λ<sub>f</sub>, λ<sub>s</sub>, λ<sub>b</sub> | `--lambda_f`, `--lambda_s`, `--lambda_b` | 1.0, 0.5, 0.5 | Semantic cost weights (Eq. 6) |
| θ<sub>p</sub>, θ<sub>q</sub>, θ<sub>b</sub>, θ<sub>d</sub> | `--theta_p/q/b/d` | 1.0 | Graph edge weights (Eq. 4) |
| ε | `--transport_epsilon` | 0.05 | Entropic regularization |
| – | `--transport_fgw_iterations`, `--transport_sinkhorn_iterations` | 5, 50 | Outer FGW / inner partial-Sinkhorn iterations |
| τ | `--audit_tau` | 0.5 | Uncertainty threshold of the auditable set *K* |
| – | `--decision_fusion` | `add` | Decision-level fusion (`off` disables it) |
| – | `--force_t` | – | Fix reliability *t* (e.g. `0` or `1` for ablations) |
| λ<sub>dec</sub> | `--lambda_dec` | 0.5 | Decoder segmentation loss weight |
| *L*<sub>Q</sub>, *L*<sub>B</sub>, *L*<sub>V</sub>, *L*<sub>EDL</sub> | `--lambda_q`, `--lambda_boundary`, `--lambda_quality`, `--lambda_e` | 0.2, 0.2, 0.1, 0.2 | Auxiliary loss weights (Eq. 16) |
| – | `--aux_warmup_epochs` | 5 | Linear ramp of λ<sub>aux</sub> |
| – | `--encoder_name` | `convnextv2_base` | timm backbone |
| – | `--lr`, `--lr_backbone` | 1e-4, 1e-5 | Learning rates (heads / backbone) |
| – | `--batch_train`, `--batch_test` | 6, 8 | Per-GPU batch sizes |
| – | `--seed` | 3407 | Random seed |

Boolean flags accept `--flag true/false` and `--no-flag`. Underscores and hyphens in option names are interchangeable.

## Running Tests

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests -p 'test_*.py'
```

The tests use tiny stand-in models and temporary data to check the training, resume and final-inference workflow and the label-free inference protocol. They do not replace a full 100-epoch run on real data.

## Citation

If you find this work useful, please cite:

```bibtex
@article{jie2026pace,
  title   = {{PACE}: Partial-Correspondence Auditing and Calibrated Evidential Fusion for Unregistered {WLI/NBI} Lesion Segmentation},
  author  = {Jie, Pengyu and others},
  journal = {<manuscript>},
  year    = {2026}
}
```

Related work from our group:

```bibtex
@article{jie2026rgsf,
  title   = {Registration-Grounded Spectral Fusion for Unregistered {WLI/NBI} Endoscopic Lesion Segmentation},
  author  = {Jie, Pengyu and others},
  journal = {arXiv preprint arXiv:2607.26395},
  year    = {2026}
}
```
```bibtex
@article{jiepy_eClinicalMedicine,
  author    = {He, Rui and Jie, Pengyu and Hou, Weijian and Long, Yudong and Zhou, Guanqun and Wu, Shumei and Liu, Wanquan and Lei, Wenbin and Wen, Weiping and Wen, Yihui},
  title     = {Real-time artificial intelligence-assisted detection and segmentation of nasopharyngeal carcinoma using multimodal endoscopic data: a multi-center, prospective study},
  journal   = {eClinicalMedicine},
  year      = {2025},
  month     = {Mar},
  day       = {01},
  volume    = {81},
  issn      = {2589-5370}
}
```
```bibtex
@article{jiepy_AoM,
author = {Rui He and Pengyu Jie and Zhangfeng Wang and Yaqiong Lu and Huanhuan Lv and Zixuan Huang and Wendong Liu and Yongquan Wang and Wanquan Liu and Wenbin Lei and Weiping Wen and Yihui Wen},
title = {Artificial intelligence-assisted real-time nasopharyngeal cancer diagnostic model enhances rhinologist performance: a prospective multi-reader study},
journal = {Annals of Medicine},
volume = {58},
number = {1},
pages = {2620335},
year = {2026},
}
```

```bibtex
@article{jiepy_TCSVT,
  author  = {Jie, Pengyu and Liu, Wanquan and Gao, Chenqiang and Wen, Yihui and He, Rui and Wen, Weiping and Li, Pengcheng and Zhang, Jintao and Meng, Deyu},
  journal = {IEEE Transactions on Circuits and Systems for Video Technology},
  title   = {A Point-Neighborhood Learning Framework for Nasal Endoscopic Image Segmentation},
  year    = {2025},
  volume  = {},
  number  = {},
  pages   = {1-1}
}
```
```bibtex
@misc{jie2025diffusionguidedmaskconsistentpairedmixing,
      title={Diffusion-Guided Mask-Consistent Paired Mixing for Endoscopic Image Segmentation}, 
      author={Pengyu Jie and Wanquan Liu and Rui He and Yihui Wen and Deyu Meng and Chenqiang Gao},
      year={2025},
      eprint={2511.03219},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.03219}, 
}
```
## Acknowledgements

This implementation builds on [timm](https://github.com/huggingface/pytorch-image-models) (ConvNeXt V2 backbone). We thank the authors of the [PICCOLO dataset](https://doi.org/10.3390/app10238501) and the clinicians who verified the WLI/NBI pairings.

## License

The license has not been specified yet. Please contact the authors before using this code beyond academic research.

## Contact

- Pengyu Jie: jiepy_wy@163.com


Questions and bug reports are welcome via [GitHub Issues](https://github.com/ParryJay/PACE/issues).
