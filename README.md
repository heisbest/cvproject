# 动物图像识别（Mask-First + CBAM）

先定位（xywh）→ mask 抑制背景 → ResNet50+CBAM 分类。支持 **Animal-90** 与 **Snapshot Serengeti** 两个现成数据集。

详细下载地址见 **[DATASETS.md](./DATASETS.md)**。

## 数据集下载（摘要）

| 数据集 | 下载 |
|--------|------|
| **Animal-90** | [Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals) · [Hugging Face](https://huggingface.co/datasets/lucabaggi/animal-wildlife) |
| **Snapshot Serengeti** | [LILA BC](https://lila.science/datasets/snapshot-serengeti/) · [Dryad](https://doi.org/10.5061/dryad.5pt92) |

## 权重目录规则

`--data-path ./data/animal-90` → 权重保存在 `./weights/animal-90/`

```
weights/animal-90/
├── class_names.json
├── localizer_best.pth
├── classifier_cbam_best.pth
└── train_state.json
```

## 命令

```bash
pip install -r requirements.txt

# 训练（指定数据集路径）
python scripts/train.py --data-path ./data/animal-90

# 在已有权重上继续训练
python scripts/train.py --data-path ./data/animal-90 --resume

# 只评测，不训练（指定评测集）
python scripts/evaluate.py --weights-dir ./weights/animal-90 --eval-path ./data/animal-90

# 命令行推理
python scripts/infer.py --weights-dir ./weights/animal-90 --image test.jpg --output out.jpg

# 网页上传识别（可视化框 + 概率）
python web_demo.py --weights-dir ./weights/animal-90
# 或
python web_demo.py --data-path ./data/animal-90 --port 7860
```

### Serengeti 示例

```bash
python scripts/train.py --data-path ./data/serengeti --dataset-type serengeti
python scripts/evaluate.py --data-path ./data/serengeti --eval-path ./data/serengeti
python web_demo.py --data-path ./data/serengeti
```

## 项目结构

```
models/          CBAM、ResNet50-CBAM、Localizer、Pipeline
data/            数据集加载与类型检测
scripts/         train / evaluate / infer
web_demo.py      网页上传识别（单文件）
weights/         按数据集名称保存权重
DATASETS.md      数据集下载详细说明
```

## 方法要点（汇报用）

1. 传统全图分类在多动物场景泛化差 → **先定位再分类**
2. 定位输出 **xywh**，优先 **LocateAnything**，回退自训练 Localizer
3. **CBAM** 插入 ResNet Bottleneck，通道+空间注意力抑制背景
4. 定位与分类 **分开反向传播、分开算指标**
5. 不再使用 copy-paste 合成数据；改用 Animal-90 + Serengeti 真实数据集
