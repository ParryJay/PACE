# PACE：FAHSYSU / PICCOLO 统一训练与推理

本目录整合原 `main/SYSU/main_PACE.py` 和 `main/PICCOLO/main_PACE.py` 对应的训练、模型及数据读取代码。`FAHSYSU` 对应原 SYSU 数据集。可以单独复制本目录使用，无需导入原项目的 `main`、`Trainer`、`model`、`dataset` 或 `lib`。

## 实验流程

```text
仅加载训练清单及训练图像、标签
    → 完整训练 100 epoch（无中途测试）
    → 保存第 100 epoch 的 PACE_final.pth
    → 从该文件重新加载 online / EMA 权重
    → 首次打开测试清单，执行一次完整测试推理
```

- `--epochs` 固定为 `100`，其他值会报错。训练期间不创建测试 Dataset、不读取测试清单、不进行测试推理。
- 每个 epoch 结束保存 `PACE_last.pth`，仅用于断点续训。可额外按固定 epoch 周期保存断点。权重保存与测试结果无关，没有 best checkpoint 选择。
- 测试只读取 NBI/WLI 配对图像，不要求、不读取测试标签；不计算 IoU、Dice、边界距离、ECE 等评价指标，不生成任何可视化图片或图表。
- 保留原训练中的 Dice/Tversky **损失函数**、图匹配和审计计算，它们属于训练目标或模型算法。
- 默认保留原设置，最终推理优先使用第 100 epoch checkpoint 内已初始化的 EMA 权重；没有 EMA 时使用 online 权重。可在训练开始前通过 `--ema_eval false` 或 `--weights_source online` 明确选择 online 权重。

## 文件

| 文件 | 职责 |
| --- | --- |
| `main.py` | 主入口、单进程 / DDP / torchrun 启动 |
| `config.py` | 统一超参数、默认值和参数校验 |
| `data.py` | FAHSYSU / PICCOLO 配对清单和数据预处理 |
| `model.py` | 完整 PACE 模型和统一构建函数 |
| `trainer.py` | 训练、损失、优化器、EMA、断点保存与恢复 |
| `augmentation.py` | 原有训练图像质量退化增强 |
| `inference.py` | 第 100 epoch 最终权重校验及纯推理 |
| `requirements.txt` | 运行依赖 |
| `tests/` | 使用小型模型和临时数据的流程回归检查 |

## 环境

建议使用 Linux、Python 3.11 和支持所选 PyTorch 版本的 NVIDIA GPU：

```bash
pip install -r test/opensource/requirements.txt
python -B test/opensource/main.py --help
```

将本目录单独复制后，也可以在复制出的目录中直接运行 `python -B main.py ...`。包入口可使用 `cd test && python -B -m opensource.main ...`。

默认骨干网络为 `convnextv2_base`。首次使用 `--encoder_pretrained true` 需要下载 timm 预训练权重，或提前准备对应缓存。无网络时使用 `--encoder_pretrained false` 从随机初始化训练，或通过 `--weights` 加载已有模型权重。完整 checkpoint 的恢复和独立推理不下载骨干预训练权重。

保留原模型、双路 RGB 输入、512×512 预处理、训练损失、增强、AdamW 参数分组、学习率计划、AMP 和 EMA。图像仍使用 OpenCV 双线性缩放；训练标签仍沿用“彩色读取 → 双线性缩放 → 灰度 → 大于 100 二值化”，没有更改标签处理语义。

## 数据准备

使用已准备好的训练 / 测试配对划分，不会重新随机划分数据。训练清单包含两路图像和标签；测试清单只需两路图像字段。路径、清单均通过参数传入，没有个人机器的硬编码路径。

### FAHSYSU

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

训练 JSON 为对象数组，例如：

```json
[
  {"id": "000001", "nbi_img": "nbi.jpg", "nbi_lbl": "nbi.png", "wli_img": "wli.jpg", "wli_lbl": "wli.png"}
]
```

原项目的 FAHSYSU 清单位于仓库 `dataset/SYSU/pairs_train.json` 和 `dataset/SYSU/pairs_test.json`。可将现有清单放入数据根目录，或通过 `--train_manifest`、`--test_manifest` 指定其**绝对路径**，无需修改原文件。

### PICCOLO

```text
PICCOLO_ROOT/
├── train.jsonl
├── test.jsonl
├── img/
│   ├── NBI/
│   └── WLI/
└── lbl/
```

JSONL 每行一个配对对象，例如：

```json
{"id": "001", "nbi": "nbi.png", "wli": "wli.png"}
```

兼容 `nbi_img`、`wli_img` / `wht_img` 字段。训练时缺省标签文件名与图像文件名相同，可通过 `nbi_lbl`、`wli_lbl` 显式指定。

两种数据集均支持 JSON 数组或 `.jsonl` 清单。相对清单路径相对于 `--dataset_root` 解析；相对图像、标签文件名分别相对于上面的模态目录解析，也支持绝对文件路径。测试清单即使保留标签字段，也会被忽略。

## 训练

以下命令从仓库根目录执行。默认单卡；batch size 均为**每个进程 / GPU**的大小。

```bash
python -B test/opensource/main.py \
  --dataset FAHSYSU --dataset_root /path/to/FAHSYSU_ROOT \
  --devices 0

python -B test/opensource/main.py \
  --dataset PICCOLO --dataset_root /path/to/PICCOLO_ROOT \
  --devices 0
```

默认输出到本目录下 `runs/<dataset>_PACE_<timestamp>/`，可用 `--output_dir` 指定运行目录。主要参数：

| 参数 | 默认值 / 含义 |
| --- | --- |
| `--dataset` | 必填：`FAHSYSU` 或 `PICCOLO` |
| `--dataset_root` | 必填：对应数据根目录 |
| `--train_manifest` / `--test_manifest` | 可选：覆盖默认清单位置 |
| `--epochs` / `--img_size` | 固定 `100` / `512` |
| `--batch_train` / `--batch_test` | `6` / `8` |
| `--lr` / `--lr_backbone` | `1e-4` / `1e-5` |
| `--num_workers` / `--test_num_workers` | `6` / `1` |
| `--use_amp` / `--amp_dtype` | `true` / `bfloat16`，仅 CUDA 启用 |
| `--use_ema` / `--ema_eval` | `true` / `true` |
| `--checkpoint_interval` | `0`，大于 0 时额外按固定周期留存断点 |
| `--save_predictions` | `false`，开启后保存数值预测 `.npz` |
| `--seed` | `3407` |

布尔参数支持 `--flag false` 和 `--no-flag`，下划线参数也支持连字符形式。完整模型和训练参数见 `--help`。

多卡启动：

```bash
python -B test/opensource/main.py \
  --dataset PICCOLO --dataset_root /path/to/PICCOLO_ROOT \
  --devices 0,1,2,3

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  test/opensource/main.py \
  --dataset PICCOLO --dataset_root /path/to/PICCOLO_ROOT
```

CPU 开发检查可用 `--devices cpu --dist_backend gloo`；完整默认模型的 100 epoch 训练建议在 GPU 上执行。FP16 GPU 可指定 `--amp_dtype float16`。

## 保存和恢复

- `PACE_last.pth`：每个完整 epoch 后更新，可恢复模型、优化器、调度器、scaler、EMA、epoch 及各 rank 随机状态。
- `PACE_epoch_050.pth` 等：仅在指定 `--checkpoint_interval` 后保存，命名按 epoch 进度。
- `PACE_final.pth`：训练完成 100 epoch 后保存，记录 `final=true` 和固定 epoch 保存策略；最终推理从此文件重载权重。
- `config.json`、`training_prior.json`、`train_iterations.jsonl`、`train_epochs.jsonl`：配置、训练集前景先验和训练日志。

恢复训练：

```bash
python -B test/opensource/main.py \
  --dataset PICCOLO --dataset_root /path/to/PICCOLO_ROOT \
  --resume /path/to/run/PACE_last.pth --devices 0
```

恢复时需保持原训练超参数、数据根目录、配对清单和进程数一致；从已完成的 epoch 继续，直到第 100 epoch，再进行测试。可以通过 `--output_dir` 使用原输出目录。原项目 checkpoint 可使用 `--weights` 作为权重初始化；它不会被当作本目录的完整训练断点，也不会触发测试基线或沿用其 best 记录。

## 独立推理与输出

已经完成训练后，可以独立运行最终权重的推理，无需训练清单或训练标签：

```bash
python -B test/opensource/main.py \
  --dataset PICCOLO --dataset_root /path/to/PICCOLO_ROOT \
  --inference_only --weights /path/to/run/PACE_final.pth \
  --devices 0 --save_predictions
```

独立推理只接受本目录生成且确认完成 100 epoch 的 final checkpoint，拒绝中途断点和不匹配的数据集。模型结构和算法参数从 checkpoint 恢复；运行时只需指定数据位置、设备和输出选项。`--weights_source auto|online|ema` 控制加载来源，`auto` 遵循 checkpoint 中保存的 `ema_eval` 设定。

默认只执行推理并记录 `inference_run.json` 中的样本数量、数据集和权重来源。开启 `--save_predictions` 后，每对图像写入 `predictions/00000000.npz` 等文件，包含：

- `pair_index`、`nbi_img`、`wli_img`：配对位置及图像文件名。
- `nbi_probability`、`wli_probability`：512×512 前景概率，float32。
- `nbi_mask`、`wli_mask`：512×512 预测类别，uint8，取值 0/1。

这些文件是数值数组，不生成 PNG、热力图或叠加图；`inference_run.json` 仅记录运行信息，不含评价指标。多进程推理按不重复的索引分片，不补齐或重复测试样本。

## 开发检查

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover \
  -s test/opensource/tests -p 'test_*.py'
```

回归检查使用临时数据和小型替身模型，验证训练 / 续训 / 最终推理的时序与无标签推理协议，不代表已运行真实数据上的完整 100 epoch 训练。`--smoke_steps` 仅用于开发调试：提前退出训练、不测试、不生成 `PACE_final.pth`。
