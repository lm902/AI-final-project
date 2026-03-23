# Military Camouflage Object Detection via Cross-Domain Transfer Learning

**COMP 9130 — Applied Artificial Intelligence | Capstone Project**  

| Member | Role |  
|---|---|   
| Binger Yu | Data & Preprocessing Lead |
| Yansong Jia | Model & Training Lead |
| Sepehr Mansouri | Evaluation & Integration Lead |   

---

## Project Overview

This project investigates whether models pretrained on **natural animal camouflage** (COD10K) can transfer to detect **military personnel in camouflage** (ACD1K). We fine-tune SINet-V2 across three experimental conditions and quantify the cross-domain transfer gap.

**Research Gap:** Existing camouflage detection research focuses almost entirely on animal camouflage. The cross-domain transfer from animal → military camouflage is largely untested.

---

## Datasets

| Dataset | Images | Split | Domain | Source |
|---|---|---|---|---|
| COD10K | 10,000 | 6K/4K train/test | Natural animal camouflage | [Kaggle](https://www.kaggle.com/datasets/ivanomelchenkoim11/cod10k-dataset) |
| ACD1K | 1,078 | 748/330 train/test | Military personnel | [Kaggle](https://www.kaggle.com/datasets/aalihhiader/military-camouflage-soldiers-dataset-mcs1k) |
| CAMO | 1,250 | 1K/250 train/test | Mixed natural + artificial | [Official site](https://sites.google.com/view/ltnghia/research/camo) |

> **Data is not included in this repository.** See setup instructions below.

---

## Repository Structure

```
AI-final-project/
├── data/                    # Datasets (not tracked by Git)
│   ├── DATA_DOWNLOAD_INSTRUCTIONS.txt
│   ├── COD10K-v3/
│   ├── dataset-splitM/      # ACD1K
│   └── CAMO-V.1.0-CVIU2019/
├── notebooks/
│   ├── 01_EDA_Binger.ipynb
│   ├── 02_Training_Yansong.ipynb
│   └── 03_Evaluation_Sepehr.ipynb
├── src/
│   ├── dataset.py           # Data loading & preprocessing pipeline
│   ├── model.py             # SINet-V2 architecture
│   ├── train.py             # Training loop
│   └── evaluate.py          # Evaluation metrics
├── outputs/
│   └── figures/             # EDA and result figures (not tracked)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Experimental Conditions

| Condition | Training Data | Test Data | Purpose |
|---|---|---|---|
| 1 — Baseline | COD10K only | ACD1K test | Zero-shot transfer |
| 2 — Target only | ACD1K only | ACD1K test | Upper bound |
| 3 — Joint | COD10K + CAMO + ACD1K | ACD1K test | Cross-domain benefit |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/bing-er/AI-final-project.git
cd AI-final-project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download datasets

- **COD10K**: Download from [Kaggle](https://www.kaggle.com/datasets/ivanomelchenkoim11/cod10k-dataset), extract to `data/COD10K-v3/`
- **ACD1K**: Download from [Kaggle](https://www.kaggle.com/datasets/aalihhiader/military-camouflage-soldiers-dataset-mcs1k), extract to `data/dataset-splitM/`
- **CAMO**: Download from the [official project page](https://sites.google.com/view/ltnghia/research/camo), extract to `data/CAMO-V.1.0-CVIU2019/`

### 4. Verify dataset loading

```bash
python src/dataset.py data/
```

All three conditions should print ✅.

---

## Usage

### Run EDA (Binger)

Open `notebooks/01_EDA_Binger.ipynb` in Google Colab or JupyterLab.

### Run Training (Yansong)

```bash
# Condition 1 — COD10K baseline
python src/train.py --condition cod10k --epochs 100 --batch_size 16

# Condition 2 — ACD1K only
python src/train.py --condition acd1k --epochs 100 --batch_size 16

# Condition 3 — Joint training
python src/train.py --condition joint --epochs 100 --batch_size 16
```

### Run Evaluation (Sepehr)

```bash
python src/evaluate.py --weights outputs/best_model.pth --condition joint
```

---

## Normalization Constants

Computed from EDA on training sets:

| Dataset | Mean (R, G, B) | Std (R, G, B) |
|---|---|---|
| COD10K | (0.407, 0.424, 0.340) | (0.208, 0.198, 0.194) |
| ACD1K | (0.411, 0.405, 0.327) | (0.196, 0.196, 0.184) |
| CAMO | (0.479, 0.461, 0.360) | (0.197, 0.195, 0.185) |
| Joint | (0.432, 0.430, 0.342) | (0.200, 0.196, 0.188) |

---

## Model

**SINet-V2** (Search-and-Identification Network v2)  
- Backbone: ResNet-50 pretrained on ImageNet  
- Input resolution: 352×352  
- Reference: Fan et al., IEEE TPAMI 2022

---

## Evaluation Metrics

- **mAP@50** — primary detection metric
- **S-measure (Sα)** — structural similarity
- **E-measure (Eφ)** — perceptual alignment
- **MAE** — mean absolute error on predicted masks
- **F-measure (Fβ)** — precision-recall balance

---

## References

1. Fan et al., "Camouflaged Object Detection," CVPR 2020.
2. Haider et al., "Identification of Camouflage Military Individuals with DFAN and SINetV2," *Scientific Reports*, 2025.
3. Le et al., "Anabranch Network for Camouflaged Object Segmentation," *CVIU*, 2019.

---

## License

For academic use only. Datasets are subject to their respective licenses.
