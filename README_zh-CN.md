<div align="center">

# PACE：面向非配准 WLI/NBI 病灶分割的部分对应审计与校准证据融合

**Pengyu Jie, Wanquan Liu<sup>†</sup>, Rui He, Yihui Wen, Pengcheng Li, Weiping Wen, Deyu Meng, Junwei Han, Chenqiang Gao<sup>†</sup>**

<sup>†</sup> 通讯作者

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TMI-blue)](#引用)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-TBD-lightgrey)](#许可证)

[English](README.md) | 简体中文

</div>

本仓库是 **PACE** 的官方 PyTorch 实现。PACE 用于融合**非配准**的白光成像（WLI）与窄带成像（NBI）内镜图像对，完成病灶分割。

<!-- 将论文 Fig. 2 框架图放到 assets/framework.png 后取消注释：
<p align="center"><img src="assets/framework.png" width="90%" alt="PACE 框架"></p>
-->

## 目录

- [方法概述](#方法概述)
- [主要结果](#主要结果)
- [代码结构](#代码结构)
- [环境安装](#环境安装)
- [数据准备](#数据准备)
- [训练](#训练)
- [推理](#推理)
- [评估](#评估)
- [关键超参数](#关键超参数)
- [运行测试](#运行测试)
- [引用](#引用)
- [致谢](#致谢)
- [许可证](#许可证)
- [联系方式](#联系方式)

## 方法概述

WLI 与 NBI 通过切换光源依次采集，期间的视角变化、组织形变和遮挡使每个图像对都无法配准，且只部分重叠。融合只能依赖*估计出的*跨模态对应关系，而对应错误很难被发现：误匹配的消息来自另一个真实区域，置信度与正确匹配一样高，基于不确定性的融合无法识别，融合后还可能产生负迁移。

PACE 不假设对应关系正确，而是对每个图像对的对应关系进行**审计**，再用审计得到的可靠度校准融合：

| 模块 | 作用 |
| --- | --- |
| **病灶中心证据图（LCEG）** | 将每个视图表示为由病灶引导的软区域构成的图。边只依赖相对距离（对平面内平移、旋转不变），每个节点带有主观逻辑意见（病灶信念、背景信念、不确定度）。 |
| **自适应部分融合 Gromov–Wasserstein（PFGW）传输** | 以逐对预测的传输预算 ρ 和结构权重 α 匹配两张图。未匹配质量由虚节点吸收，没有对应区域的部分以"未知"而非证据的形式进入融合。 |
| **对应审计** | 在双模态均置信观测的节点上，衡量跨模态一致性超出随机水平的程度（Cohen's κ），并转化为该图像对污染率的上界 η ≤ 1 − κ。无需对应关系真值，审计本身没有可学习参数。 |
| **审计校准融合** | 可靠度 *t* = 1 − η̂⁺ 同时控制三个特征层级上的 **Shared–Private–Reject（TSPR）** 路由，以及基于 Huber 污染模型的**鲁棒决策级似然**。 |

**理论保证。** 若被污染的消息与目标仅在随机水平上一致，经随机校正的一致性可给出污染率的*紧*上界。高估污染率只会让融合退向单模态预测，因此审计校准的决策级融合渐近地不差于单模态预测。

## 主要结果

四次运行的均值 ± 标准差。所有方法使用相同的患者级划分、输入分辨率（512 × 512）和评估协议。全部 11 个对比方法见论文 Table I。

| 数据集 | 方法 | mIoU (%) | DSC (%) | PA (%) | Recall (%) | Precision (%) |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| PICCOLO (Paired) | ADFNet (MIA 2026) | 80.52 ± 0.81 | 82.48 ± 0.30 | 94.67 ± 0.43 | 84.90 ± 0.40 | 80.20 ± 0.44 |
| | RGSF (arXiv 2026) | 83.54 ± 0.31 | 84.78 ± 0.36 | 94.49 ± 0.26 | 89.85 ± 0.75 | 80.25 ± 0.24 |
| | **PACE（本文）** | **87.08 ± 0.13** | **89.22 ± 0.13** | **95.88 ± 0.16** | **93.87 ± 0.44** | **85.01 ± 0.47** |
| FAHSYSU (Paired) | ADFNet (MIA 2026) | 79.72 ± 0.27 | 80.72 ± 0.19 | 92.96 ± 0.33 | 80.79 ± 0.28 | 80.66 ± 0.26 |
| | RGSF (arXiv 2026) | 81.08 ± 0.23 | 82.37 ± 0.17 | 93.40 ± 0.30 | 83.68 ± 0.25 | 81.11 ± 0.22 |
| | **PACE（本文）** | **82.36 ± 0.09** | **83.44 ± 0.24** | **93.88 ± 0.11** | **85.72 ± 0.35** | **81.27 ± 0.29** |

只扰动辅助视图时（平移 20%、旋转 30°、缩放 ±30%、视野裁剪至 50%），PACE 依然稳定：在 FAHSYSU 上，严重扰动下平均 DSC 最多只下降 0.045 pp。

## 代码结构

```text
PACE/
├── main.py              # 主入口：单进程 / 多卡 DDP / torchrun 启动
├── config.py            # 全部命令行参数、默认值与校验
├── data.py              # WLI/NBI 配对清单读取与预处理（FAHSYSU / PICCOLO）
├── model.py             # PACE 模型：LCEG、PFGW 求解、审计、TSPR 路由、决策融合
├── trainer.py           # 训练循环、损失、优化器分组、EMA、断点保存
├── augmentation.py      # 训练用合成图像质量退化增强
├── inference.py         # 最终权重校验与无标签推理
├── requirements.txt     # pip 依赖
├── requirement.yaml     # conda 环境
└── tests/               # 使用小模型与临时数据的回归测试
```

## 环境安装

已在 Linux、Python 3.11 和 NVIDIA GPU 上测试（论文实验使用 4 × RTX 3090）。

**方式 A：conda**

```bash
git clone https://github.com/ParryJay/PACE.git
cd PACE
conda env create -f requirement.yaml
conda activate mmmv-pace
```

**方式 B：pip**

```bash
git clone https://github.com/ParryJay/PACE.git
cd PACE
conda create -n pace python=3.11 -y && conda activate pace
pip install -r requirements.txt
```

检查安装：

```bash
python -B main.py --help
```

默认骨干网络为 [timm](https://github.com/huggingface/pytorch-image-models) 中的 `convnextv2_base`。默认 `--encoder_pretrained true`，首次运行时会下载 ImageNet 预训练权重。离线环境可提前准备 timm / Hugging Face 缓存，或用 `--encoder_pretrained false` 从随机初始化训练。从完整断点恢复训练和独立推理都不会下载骨干权重。

## 数据准备

每个样本是同一患者同一病灶分别采集、**未配准**的 WLI 与 NBI 图像，每个模态各有一张二值掩膜。代码直接使用你提供的固定训练 / 测试配对清单，**不会**重新划分数据。数据划分应在患者级别完成。

| 数据集 | 配对数（训练 / 测试） | 获取方式 |
| --- | --- | --- |
| FAHSYSU (Paired) | 14,529（11,512 / 3,017） | 中山大学附属第一医院院内数据集，如需获取请联系通讯作者。 |
| PICCOLO (Paired) | 7,362（5,529 / 1,833） | 公开数据集：Sánchez-Peralta 等，*Applied Sciences*，2020（[doi:10.3390/app10238501](https://doi.org/10.3390/app10238501)）。 |

### FAHSYSU 目录结构

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

清单格式（JSON 数组或 `.jsonl`）：

```json
[
  {"id": "000001", "nbi_img": "nbi.jpg", "nbi_lbl": "nbi.png", "wli_img": "wli.jpg", "wli_lbl": "wli.png"}
]
```

### PICCOLO 目录结构

```text
PICCOLO_ROOT/
├── train.jsonl
├── test.jsonl
├── img/
│   ├── NBI/
│   └── WLI/
└── lbl/
```

清单格式（每行一个配对）：

```json
{"id": "001", "nbi": "nbi.png", "wli": "wli.png"}
```

### 说明

- 字段别名：NBI 图像可用 `nbi` / `nbi_img`，WLI 图像可用 `wli` / `wli_img` / `wht_img`。省略 `nbi_lbl` / `wli_lbl` 时，标签文件名默认与图像文件名相同。
- 两个数据集都支持 JSON 数组和 `.jsonl`。清单放在其他位置时，用 `--train_manifest` / `--test_manifest` 指定。相对清单路径基于 `--dataset_root` 解析；相对的图像、标签文件名基于上面各模态目录解析，也支持绝对路径。
- 测试清单只需两个图像字段，其中的标签字段会被忽略。
- 预处理：图像双线性缩放到 512 × 512；标签读取后双线性缩放、转灰度，再以阈值 100 二值化。

## 训练

以下命令均在仓库根目录执行。batch size 均为**每个进程（每张 GPU）**的大小。

**单卡**

```bash
# FAHSYSU
python -B main.py --dataset FAHSYSU --dataset_root /path/to/FAHSYSU_ROOT --devices 0

# PICCOLO
python -B main.py --dataset PICCOLO --dataset_root /path/to/PICCOLO_ROOT --devices 0
```

**多卡**

```bash
# 内置启动器（每张 GPU 一个进程）
python -B main.py --dataset PICCOLO --dataset_root /path/to/PICCOLO_ROOT --devices 0,1,2,3

# 或使用 torchrun
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  main.py --dataset PICCOLO --dataset_root /path/to/PICCOLO_ROOT
```

**训练协议**

- 固定训练 **100 个 epoch**（`--epochs` 只能为 100），输入分辨率固定 **512 × 512**（`--img_size` 只能为 512）。
- 训练期间不会打开测试清单，没有中途测试，也不挑选"最佳权重"。
- 第 100 个 epoch 结束后保存并重新加载 `PACE_final.pth`，对测试集做一次推理。默认使用 EMA 权重；如需 online 权重，请使用 `--ema_eval false` 或 `--weights_source online`。
- CUDA 上默认开启 `bfloat16` 混合精度。不支持 bf16 的 GPU 可用 `--amp_dtype float16`。
- CPU 快速自检：`--devices cpu --dist_backend gloo --smoke_steps 2`。`--smoke_steps` 会提前结束训练，不做测试，也不生成最终权重。

**输出文件**（默认在 `runs/<dataset>_PACE_<timestamp>/`，可用 `--output_dir` 指定）

| 文件 | 说明 |
| --- | --- |
| `PACE_last.pth` | 每个 epoch 结束后更新，包含模型、优化器、调度器、scaler、EMA、epoch 及各 rank 随机状态，用于断点续训。 |
| `PACE_epoch_XXX.pth` | 仅在 `--checkpoint_interval > 0` 时按周期额外保存。 |
| `PACE_final.pth` | 训练满 100 epoch 后的最终权重（标记 `final=true`），用于推理。 |
| `config.json` | 完整的解析后配置。 |
| `training_prior.json` | 训练集前景先验（病灶先验 *p*<sub>les</sub>）。 |
| `train_iterations.jsonl`、`train_epochs.jsonl` | 训练日志。 |

**断点续训**

```bash
python -B main.py --dataset PICCOLO --dataset_root /path/to/PICCOLO_ROOT \
  --resume /path/to/run/PACE_last.pth --output_dir /path/to/run --devices 0
```

续训时请保持超参数、数据根目录、配对清单和进程数与原始运行一致。`--weights` 只加载模型权重（热启动），不恢复训练状态。

## 推理

用训练完成的 100 epoch 权重做无标签推理：

```bash
python -B main.py --dataset PICCOLO --dataset_root /path/to/PICCOLO_ROOT \
  --inference_only --weights /path/to/run/PACE_final.pth \
  --devices 0 --save_predictions
```

- 只接受本代码生成的 `PACE_final.pth`，中途断点或数据集不匹配的权重会被拒绝。
- 模型结构和算法参数从权重文件中恢复，运行时只需指定数据位置、设备和输出选项。
- `--weights_source auto|online|ema` 选择权重来源，`auto` 沿用权重文件中保存的 `ema_eval` 设置。
- 运行信息写入 `inference_run.json`。开启 `--save_predictions` 后，每个图像对保存为 `predictions/XXXXXXXX.npz`：

| 键 | 类型 | 说明 |
| --- | --- | --- |
| `pair_index`、`nbi_img`、`wli_img` | – | 配对序号与图像文件名 |
| `nbi_probability`、`wli_probability` | `float32`，512 × 512 | 前景概率 |
| `nbi_mask`、`wli_mask` | `uint8`，512 × 512 | 二值预测（0/1） |

多进程推理按不重复的索引分片，不补齐、不重复样本。

## 评估

推理脚本只输出原始预测，不计算指标，也不生成可视化。复现论文结果时，将各模态的预测分别与该模态自身的真值掩膜比较（不做配准或掩膜形变），阈值取 0.5，报告以下指标：

- **mIoU**：前景与背景两类 IoU 的平均
- **DSC** = 2 · Precision · Recall / (Precision + Recall)
- **PA** = (TP + TN) / (TP + TN + FP + FN)
- **Recall** = TP / (TP + FN)，**Precision** = TP / (TP + FP)

论文中每个图像对权重相同，包括共享同一源图像的图像对。

## 关键超参数

论文符号与命令行参数的对应关系及默认值。完整参数见 `python -B main.py --help`。

| 论文符号 | 参数 | 默认值 | 含义 |
| --- | --- | --- | --- |
| *N* | `--num_tokens` | 128 | 每个模态的图节点数 |
| [ρ<sub>min</sub>, ρ<sub>max</sub>] | `--rho_min`、`--rho_max` | 0.2、0.85 | 逐对传输预算范围 |
| [α<sub>min</sub>, α<sub>max</sub>] | `--alpha_min`、`--alpha_max` | 0.05、0.35 | FGW 结构项权重范围 |
| λ<sub>f</sub>、λ<sub>s</sub>、λ<sub>b</sub> | `--lambda_f`、`--lambda_s`、`--lambda_b` | 1.0、0.5、0.5 | 语义代价权重（式 6） |
| θ<sub>p</sub>、θ<sub>q</sub>、θ<sub>b</sub>、θ<sub>d</sub> | `--theta_p/q/b/d` | 1.0 | 图边权重（式 4） |
| ε | `--transport_epsilon` | 0.05 | 熵正则系数 |
| – | `--transport_fgw_iterations`、`--transport_sinkhorn_iterations` | 5、50 | 外层 FGW / 内层部分 Sinkhorn 迭代次数 |
| τ | `--audit_tau` | 0.5 | 可审计集合 *K* 的不确定度阈值 |
| – | `--decision_fusion` | `add` | 决策级融合（`off` 为关闭） |
| – | `--force_t` | – | 固定可靠度 *t*（消融时可设为 `0` 或 `1`） |
| λ<sub>dec</sub> | `--lambda_dec` | 0.5 | 解码器分割损失权重 |
| *L*<sub>Q</sub>、*L*<sub>B</sub>、*L*<sub>V</sub>、*L*<sub>EDL</sub> | `--lambda_q`、`--lambda_boundary`、`--lambda_quality`、`--lambda_e` | 0.2、0.2、0.1、0.2 | 辅助损失权重（式 16） |
| – | `--aux_warmup_epochs` | 5 | λ<sub>aux</sub> 线性升温的 epoch 数 |
| – | `--encoder_name` | `convnextv2_base` | timm 骨干网络 |
| – | `--lr`、`--lr_backbone` | 1e-4、1e-5 | 学习率（头部 / 骨干） |
| – | `--batch_train`、`--batch_test` | 6、8 | 每卡 batch size |
| – | `--seed` | 3407 | 随机种子 |

布尔参数支持 `--flag true/false` 和 `--no-flag` 两种写法，参数名中的下划线与连字符可互换。

## 运行测试

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests -p 'test_*.py'
```

测试使用小型替身模型和临时数据，检查训练、续训、最终推理的流程以及无标签推理协议，不能替代在真实数据上完整训练 100 epoch。

## 引用

如果本工作对你有帮助，请引用：

```bibtex
@article{jie2026pace,
  title   = {{PACE}: Partial-Correspondence Auditing and Calibrated Evidential Fusion for Unregistered {WLI/NBI} Lesion Segmentation},
  author  = {Jie, Pengyu and Liu, Wanquan and He, Rui and Wen, Yihui and Li, Pengcheng and Wen, Weiping and Meng, Deyu and Han, Junwei and Gao, Chenqiang},
  journal = {<manuscript>},
  year    = {2026}
}
```

本组相关工作：

```bibtex
@article{jie2026rgsf,
  title   = {Registration-Grounded Spectral Fusion for Unregistered {WLI/NBI} Endoscopic Lesion Segmentation},
  author  = {Jie, Pengyu and others},
  journal = {arXiv preprint arXiv:2607.26395},
  year    = {2026}
}
```

## 致谢

本实现基于 [timm](https://github.com/huggingface/pytorch-image-models)（ConvNeXt V2 骨干）。感谢 [PICCOLO 数据集](https://doi.org/10.3390/app10238501)的作者，以及参与 WLI/NBI 配对核验的临床医生。

## 许可证

许可证尚未确定。如需将本代码用于学术研究以外的用途，请先联系作者。

## 联系方式

- Wanquan Liu：liuwq63@mail.sysu.edu.cn
- Chenqiang Gao：gaochq6@mail.sysu.edu.cn

欢迎通过 [GitHub Issues](https://github.com/ParryJay/PACE/issues) 提问或报告问题。
