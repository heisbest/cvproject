# 动物图像识别（Mask-First + 多骨干融合）

先定位（xywh）→ 可选 mask → **三骨干分类器 + 不确定性融合**。  
课程评测以 **Animal-90 分类准确率** 为主；多动物定位与 mask-first 为扩展创新点。

详细数据集下载见 **[DATASETS.md](./DATASETS.md)**。

---

## 快速开始

```bash
pip install -r requirements.txt

# 下载 ImageNet 预训练骨干（EfficientNet 等，建议开代理）
export http_proxy="http://127.0.0.1:7893"
export https_proxy="http://127.0.0.1:7893"
python scripts/download_pretrained_weights.py

# 只训分类（冲 Animal-90 准确率，推荐）
python scripts/train.py --data-path ./data/animal-90 --no-loc \
  --backbones efficientnet_b3 --cls-epochs 50 --split-seed 42

# 评测
python scripts/evaluate.py --weights-dir ./weights/animal-90 \
  --eval-path ./data/animal-90 --split-seed 42

# 推理：三模型融合（默认）
python scripts/infer.py --weights-dir ./weights/animal-90 --image test.jpg --output out.jpg

# 推理：仅 EfficientNet-B3
python scripts/infer.py --weights-dir ./weights/animal-90 --image test.jpg \
  --backbones efficientnet_b3
```

---

## 权重目录

`--data-path ./data/animal-90` → 权重默认保存在 `./weights/animal-90/`（文件夹名取自数据集目录名）。

```
weights/animal-90/
├── class_names.json                      # 90 类名称列表
├── localizer_best.pth                    # 自训练定位器（可选）
├── classifier_resnet_cbam_best.pth       # ResNet50+CBAM
├── classifier_efficientnet_b3_best.pth   # EfficientNet-B3
├── classifier_convnext_t_best.pth        # ConvNeXt-Tiny
├── classifier_cbam_best.pth              # 旧版单 ResNet 权重（兼容）
└── train_state.json                      # 训练配置与 split seed 记录
```

---

## 使用权重进行推理 / 评测

### 分类器模式

| 模式 | 条件 | 行为 |
|------|------|------|
| **单模型** | `--backbones <一个>` | 只加载指定 checkpoint |
| **多模型融合** | 不传 `--backbones` | 加载目录中**所有已存在**的 checkpoint，按不确定性融合 |
| **指定子集融合** | `--backbones A B` | 只融合列出的模型 |

可选 backbone 名称：

| 名称 | 模型 | 权重文件 |
|------|------|----------|
| `resnet_cbam` | ResNet50 + CBAM | `classifier_resnet_cbam_best.pth` |
| `efficientnet_b3` | EfficientNet-B3 | `classifier_efficientnet_b3_best.pth` |
| `convnext_t` | ConvNeXt-Tiny | `classifier_convnext_t_best.pth` |

### 推理示例

```bash
# 三模型不确定性融合（默认）
python scripts/infer.py --weights-dir ./weights/animal-90 --image cat.jpg

# 只用 ResNet+CBAM
python scripts/infer.py --weights-dir ./weights/animal-90 --image cat.jpg \
  --backbones resnet_cbam

# 融合 EfficientNet + ConvNeXt（不用 ResNet）
python scripts/infer.py --weights-dir ./weights/animal-90 --image cat.jpg \
  --backbones efficientnet_b3 convnext_t

# 禁用 LocateAnything，仅用自训练 localizer
python scripts/infer.py --weights-dir ./weights/animal-90 --image cat.jpg --no-locateanything
```

### 评测示例

```bash
# 各单模型准确率 + 融合准确率
python scripts/evaluate.py --weights-dir ./weights/animal-90 \
  --eval-path ./data/animal-90 --split-seed 42

# 只评 EfficientNet
python scripts/evaluate.py --weights-dir ./weights/animal-90 \
  --eval-path ./data/animal-90 --backbones efficientnet_b3 --split-seed 42
```

### Web Demo

```bash
python web_demo.py --weights-dir ./weights/animal-90 --port 7860
python web_demo.py --weights-dir ./weights/animal-90 --backbones efficientnet_b3
```

---

## `scripts/train.py` 参数说明

### 数据与路径

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-path` | （必填） | 训练数据根目录，如 `./data/animal-90` |
| `--dataset-type` | `auto` | 强制数据集类型：`animal90` / `multianimal` / `serengeti` / `auto` |
| `--weights-dir` | `weights/<数据集名>/` | 自定义权重保存目录 |
| `--eval-path` | 同 `data-path` | 训练结束后评测的数据路径 |
| `--eval-only` | 关 | 只评测，不训练 |

### 训练阶段控制

| 参数 | 说明 |
|------|------|
| `--no-loc` | **跳过定位器训练**。冲 Animal-90 分类准确率时建议加上 |
| `--no-cls` | 跳过分类器训练 |
| `--train-only resnet` | 等价于 `--no-loc`，只训分类 |
| `--train-only maskformer` | 等价于 `--no-cls`，只训定位 |
| `--resume` | 从 `weights-dir` 已有 checkpoint 继续训练（**仍会按 split-seed 重新划分 train/val**） |

### 分类骨干 `--backbones`

指定要训练的分类模型（可多个，空格分隔）：

```bash
--backbones resnet_cbam                    # 只训 ResNet+CBAM
--backbones efficientnet_b3                # 只训 EfficientNet-B3
--backbones resnet_cbam efficientnet_b3 convnext_t   # 训三个再融合
```

### 划分与增强

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--split-seed` | `random` | train/val **分层划分**随机种子。`random` = 每次训练新种子；固定评测可设 `42` |
| `--val-ratio` | `0.15` | 验证集比例 |
| `--augment-level` | `strong` | 数据增强强度：`none` / `standard` / `strong`（推荐 strong） |

`strong` 增强包含：RandomResizedCrop、翻转、**旋转**、Affine 几何变换、ColorJitter（亮度/对比度/**饱和度**/色相）、模糊、RandomErasing、TrivialAugmentWide 等。

### 分类训练超参

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cls-epochs` | `50` | 分类训练 epoch |
| `--cls-lr` | `3e-4` | 分类头学习率（骨干为 0.25×） |
| `--weight-decay` | `1e-4` | AdamW 权重衰减（L2 正则） |
| `--label-smoothing` | `0.1` | 标签平滑 |
| `--grad-clip` | `1.0` | 梯度裁剪 max norm |
| `--unfreeze-epoch` | `5` | 第 N epoch 解冻全骨干 |
| `--cls-use-mask` | 关 | 训练分类时使用 bbox mask（默认**全图**，更适合 Animal-90） |
| `--no-amp` | 关 | 关闭混合精度 AMP |

### 定位训练（创新 / 扩展）

| 参数 | 说明 |
|------|------|
| `--loc-data-paths` | 多动物定位数据，如 `./data/animal-90-multianimal ./data/serengeti-multianimal` |
| `--loc-eval-path` | 多动物定位评测集 |
| `--loc-epochs` | 定位 epoch 数（默认 20） |
| `--loc-lr` | 定位学习率 |

多动物数据下载：

```bash
python scripts/download_localization_data.py
```

---

## 其他脚本

| 脚本 | 用途 |
|------|------|
| `scripts/download_pretrained_weights.py` | 下载 ResNet / EfficientNet-B3 / ConvNeXt ImageNet 预训练权重 |
| `scripts/download_localization_data.py` | 下载 Serengeti 多动物子集 + 检查 multianimal 数据 |
| `scripts/generate_datasets.py` | 从 Animal-90 合成 multianimal / occlusion 数据 |
| `scripts/evaluate.py` | 独立评测脚本 |
| `scripts/infer.py` | 单张图推理 + 可视化 |
| `web_demo.py` | 网页上传识别 |

---

## 方法概览

1. **定位**：LocateAnything（零样本多框，可选）或自训练 `BboxLocalizer`（单框回退）
2. **分类**：ResNet50+CBAM / EfficientNet-B3 / ConvNeXt-T 三骨干
3. **融合**：Softmax + 预测熵加权的不确定性融合（非简单平均）
4. **Animal-90 准确率优化**：强增强 + AdamW + 余弦退火 + 标签平滑 + 分层划分
5. **扩展**：多动物定位数据、mask-first 流水线（ `--cls-use-mask` ）

---

## 项目结构

```
models/          CBAM、三骨干分类器、Localizer、Pipeline、Ensemble
data/            数据集、划分、transforms
scripts/         train / evaluate / infer / 下载脚本
web_demo.py      网页 Demo
weights/         按数据集名保存 checkpoint
configs/         默认超参
DATASETS.md      数据集下载
docs/            汇报 PPT 等
```

---

## 常见问题

**Q: 训练 EfficientNet 时下载权重失败？**  
A: 设置代理后运行 `python scripts/download_pretrained_weights.py --only efficientnet_b3`。

**Q: 只想用课程提供的权重做预测，不训练？**  
A: 把权重放到 `weights/animal-90/`，运行 `scripts/infer.py` 或 `web_demo.py`，用 `--backbones` 选择模型。

**Q: `--split-seed random` 和固定 `42` 的区别？**  
A: `random` 每次训练重新划分 train/val，泛化实验用；固定种子便于和评测/报告对齐。

**Q: 为什么 `--no-loc` 还能推理出框？**  
A: 推理时可选用 LocateAnything 或已训好的 `localizer_best.pth`；`--no-loc` 只影响**训练**是否训定位器。
