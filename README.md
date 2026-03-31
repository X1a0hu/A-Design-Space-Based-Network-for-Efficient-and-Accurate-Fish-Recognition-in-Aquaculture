# A Design Space-Based Network for Efficient and Accurate Fish Recognition in Aquaculture

Automated individual fish recognition in aquaculture environments using design-space-driven deep learning and large margin cosine loss.

## Overview

This project provides code for SeekNet, a fish individual recognition network family, and related training/evaluation utilities. The codebase includes:

- SeekNet model definitions with multiple compute budgets (`cfg/models/*.yaml`)
- Four stage construction types in SeekNet: `X`, `CX`, `CX2`, `CD`
- Individual recognition training with Large Margin Cosine Loss (LMCL)
- Evaluation metrics: Accuracy (`threshold=0.95`), Rank-1 Accuracy, TAR@FAR=1e-6
- Utility scripts for dataset conversion and preprocessing
- Video pipeline entry in `main.py` (depends on `segment` and `generator` modules)

## Paper

Our paper is currently under review at Ecological Informatics. The full BibTeX citation will be updated here immediately upon acceptance.

## Project Structure

```
.
|- cfg/
|  |- datasets/
|  \- models/
|- identify/
|  |- seeknet.py
|  |- layer.py
|  |- model.py
|  |- trainer.py
|  |- estimator.py
|  |- losses.py
|  |- dataset.py
|  |- sampler.py
|  |- networks.py
|  \- regsenet.py
|- utils/
|- main.py
|- keypoint_test.py
|- regnet.py
|- resnet.py
|- resnext.py
|- vgg.py
```

## Features

- **SeekNet architecture family**: architecture definitions provided by `cfg/models/seeknet_*.yaml`
- **Multiple stage construction types**: `X`, `CX`, `CX2`, `CD` (implemented in `identify/seeknet.py`)
- **LMCL training objective**: `identify/losses.py` (`LargeMarginCosineLoss`)
- **SE attention blocks**: implemented in `identify/layer.py` (`SEBlock`)
- **Train/eval pipeline**: `SHOUModel` + `Trainer` + `Estimator`
- **Utility scripts**: data split/rename/convert, JSON-to-YOLO keypoint conversion, video-to-images extraction

## Getting Started

### Prerequisites

- Python
- PyTorch

### Installation

1. Clone repository:

```bash
git clone <https://github.com/X1a0hu/A-Design-Space-Based-Network-for-Efficient-and-Accurate-Fish-Recognition-in-Aquaculture.git>
cd <A-Design-Space-Based-Network-for-Efficient-and-Accurate-Fish-Recognition-in-Aquaculture>
```

2. Create and activate environment (optional):

```bash
conda create -n seeknet python=3.10 -y
conda activate seeknet
```

3. Install PyTorch:

```bash
pip install torch torchvision torchaudio
```

4. Install other dependencies used in code:

```bash
pip install ultralytics
pip install scikit-learn
pip install opencv-python
pip install Pillow
pip install tqdm
pip install pyyaml
pip install matplotlib
pip install numpy
pip install torchsummary
pip install fvcore
```

## Data Preparation

### Step 1: Download the Dataset

Dataset download link:

>https://zenodo.org/records/18523242

### Step 2: Organize the Data

`identify/dataset.py` expects dataset structure defined by YAML (`path`, `train`, `val`) and label folders fixed to `labels/train`, `labels/val`:

```
<dataset_root>/
|- images/
|  |- train/
|  |  |- xxx.jpg
|  |  \- ...
|  \- val/
|     |- xxx.jpg
|     \- ...
\- labels/
   |- train/
   |  |- xxx.txt
   |  \- ...
   \- val/
      |- xxx.txt
      \- ...
```

**Label format:** each `.txt` contains one integer class id.

## Model Zoo

SeekNet variants are defined in `cfg/models/`.

| Model Variant | Config File | FLOPs (approx.) | Embed Dim | Total Blocks | Margin (m) | Scale (s) |
|:---|:---|:---|:---|:---|:---|:---|
| **SeekNet-400MF** | `seeknet_400mf.yaml` | 0.42 G | 512 | 8 | 0.35 | 32 |
| **SeekNet-800MF** | `seeknet_800mf.yaml` | 0.90 G | 512 | 12 | 0.35 | 32 |
| **SeekNet-1.5GF** | `seeknet_1_5gf.yaml` | 1.54 G | 512 | 9 | 0.35 | 32 |
| **SeekNet-3.4GF** | `seeknet_3_4gf.yaml` | 3.49 G | 512 | 17 | 0.35 | 32 |
| **SeekNet-4.7GF** | `seeknet_4_7gf.yaml` | 4.78 G | 512 | 15 | 0.35 | 32 |
| **SeekNet-7.3GF** | `seeknet_7_3gf.yaml` | 7.31 G | 512 | 17 | 0.35 | 32 |

Each variant supports four stage types: `X`, `CX`, `CX2`, `CD`.

## Training

### Run Training

```bash
python train.py
```

### Training Outputs

Training output directory is auto-created under:

```
runs/identify/seek/
|- train/
|  |- weights/
|  |  |- best.pt
|  |  \- last.pt
|  \- results.csv
|- train1/
|- train2/
\- ...
```

### Training Hyperparameters

| Parameter | Default | Description |
|:---|:---|:---|
| `stage_type` | `"X"` | Stage construction type (`X`, `CX`, `CX2`, `CD`) |
| `embed_dim` | `512` | Embedding dimension |
| `m` | `0.35` | LMCL margin |
| `s` | `32` | LMCL scale |
| `epochs` | `100` | Training epochs |
| `batch` | `32` | Batch size |
| `workers` | `1` | DataLoader workers |
| `lr0` | `1e-2` | Initial learning rate |
| `lrf` | `1e-2` | Final LR factor (`eta_min = lr0 * lrf`) |
| `weight_decay` | `1e-4` | SGD weight decay |
| `momentum` | `0.9` | SGD momentum |
| `interval` | `5` | Checkpoint interval |

## Key Dependencies

| Package | Purpose |
|:---|:---|
| `torch` | Deep learning training/inference |
| `torchvision` | Image transforms and model backbones |
| `ultralytics` | YOLO-related scripts (`keypoint_test.py`) |
| `scikit-learn` | ROC computation (`roc_curve`) |
| `opencv-python` | Video/image IO and drawing |
| `Pillow` | Image loading |
| `pyyaml` | YAML config parsing |
| `tqdm` | Training progress bar |
| `matplotlib` | Plotting and color utilities |
| `numpy` | Numeric operations |
| `fvcore` | FLOPs analysis |
| `torchsummary` | Model summary |

## Citation

Our paper is currently under review at Ecological Informatics. The full BibTeX citation will be updated here immediately upon acceptance.

For now, if you use this code, please refer to this repository or cite the paper title:
[A Design Space-Based Network for Efficient and Accurate Fish Recognition in Aquaculture]

## Contact

For questions or issues:

- GitHub issue:https://github.com/X1a0hu/A-Design-Space-Based-Network-for-Efficient-and-Accurate-Fish-Recognition-in-Aquaculture/issues
- Email:2241125@st.shou.edu.cn
