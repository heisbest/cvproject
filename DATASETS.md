# 数据集下载地址

本项目支持 **ImageFolder 格式**（Animal-90）和 **Snapshot Serengeti**（带 bbox 标注的相机陷阱数据）。

---

## 1. Animal-90（90 类动物，约 5400 张）

单动物、每类约 60 张，适合分类基线实验。

| 来源 | 链接 | 说明 |
|------|------|------|
| **Kaggle（官方）** | https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals | 需 Kaggle 账号，下载 zip |
| **Hugging Face** | https://huggingface.co/datasets/lucabaggi/animal-wildlife | 已含 80/20 划分，可用 `datasets` 库加载 |
| **TIB 元数据** | https://doi.org/10.57702/en8usfd1 | 数据集 DOI 索引页 |

### Kaggle CLI 下载

```bash
pip install kaggle
# 配置 ~/.kaggle/kaggle.json 后：
kaggle datasets download -d iamsouravbanerjee/animal-image-dataset-90-different-animals
unzip animal-image-dataset-90-different-animals.zip -d ./data/animal-90
```

解压后目录结构应为：

```
data/animal-90/
├── antelope/
│   ├── xxx.jpg
├── crab/
└── ...（90 个类别文件夹）
```

### Hugging Face 下载

```python
from datasets import load_dataset
ds = load_dataset("lucabaggi/animal-wildlife")
# 导出为 ImageFolder 或直接在代码里使用
```

---

## 2. Snapshot Serengeti（塞伦盖蒂相机陷阱，多动物 + bbox）

真实 savanna 背景，含多种哺乳动物，带 **边界框标注**，适合「先定位再分类」流程。

| 来源 | 链接 | 说明 |
|------|------|------|
| **LILA BC（推荐入口）** | https://lila.science/datasets/snapshot-serengeti/ | 官方聚合页，含云存储与 metadata |
| **Zooniverse 项目页** | https://www.zooniverse.org/projects/zooniverse/snapshot-serengeti | 背景介绍 |
| **Dryad 元数据** | https://doi.org/10.5061/dryad.5pt92 | 共识分类 CSV/JSON（约 520MB） |
| **Nature Scientific Data 论文** | https://www.nature.com/articles/sdata201526 | 数据描述与字段说明 |

### 云存储（按季下载，无需巨型 zip）

- **GCP**: `gs://public-datasets-lila/snapshotserengeti-unzipped`
- **AWS**: `s3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/snapshotserengeti-unzipped`
- **Azure**: `https://lilawarehouse.blob.core.windows.net/lila-wildlife/snapshotserengeti-unzipped`

LILA 页面还提供：

- **Bounding boxes** metadata（JSON，用于定位训练）
- **Recommended train/val splits**
- 各 Season 图像与 metadata 分开下载

### 建议本地目录结构

```
data/serengeti/
├── images/                          # 相机陷阱图片（或按 Season 子目录）
│   └── S1/B04/B04_R1/xxx.JPG
└── snapshotserengeti_bbox.json      # LILA 提供的 bbox 标注 JSON（文件名可不同）
```

将 LILA 下载的 bbox JSON 放在根目录，程序会自动识别 `*bounding*box*.json` / `snapshot*.json` 等文件名。

> Serengeti 全量数据极大（数 TB）。课程实验可只下载 **1 个 Season + bbox metadata**，或 LILA 推荐的 train/val 子集。

---

## 权重保存规则

训练时指定 `--data-path`，权重自动保存到：

```
weights/<数据集文件夹名>/
├── class_names.json
├── localizer_best.pth
├── classifier_cbam_best.pth
└── train_state.json
```

例如 `--data-path ./data/animal-90` → `./weights/animal-90/`
