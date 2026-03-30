# Military Camouflage Object Detection via Cross-Domain Transfer Learning

**COMP 9130 — Applied Artificial Intelligence | Capstone Project**
**British Columbia Institute of Technology — MSc Applied Computing**

---

## Team

| Member | Role | Specific Responsibilities |
|---|---|---|
| Binger Yu | Data & Preprocessing Lead + Experiment 3 Lead | - Generated and version-controlled final test set hold-out index files. Conducted full EDA (`01_EDA_Binger.ipynb`). <br/>- Designed and implemented the complete preprocessing pipeline (`src/dataset.py`): mask binarization, Albumentations augmentation, and `WeightedRandomSampler` for ACD1K oversampling. <br/>- Implemented `src/generate_splits.py` for reproducible splits with fixed random seed across all three experiments. Implemented dataloaders for all three experimental conditions. <br/>- Led Experiment 3 (joint training): hyperparameter tuning, full final training, and final test set evaluation. <br/>- Maintains GitHub repository structure, `README.md`, `requirements.txt`, `.gitignore`, and dataset download instructions. <br/>- Contributes to proposal and report (Background, Dataset, Timeline, Team Roles, Conclusion, Contributions, Acknowledgements). |
| Yansong Jia | Methodology Lead + Experiment 1 Lead | - Owns the overall methodology: defines and documents the experimental design, fixed hyperparameters, augmentation policy, and evaluation protocol. Revises methodology if issues arise during implementation. <br/>- Tracks project progress across all three experiments; facilitates cross-member communication and knowledge transfer. <br/>- Leads Experiment 1: implements and trains SINetV2 on ACD1K (CNN baseline), performs hyperparameter tuning, and evaluates on the final test set. <br/>- Contributes to proposal and report (Background, Research Gap, Methodology, Discussion — Ethical Considerations). |
| Sepehr Mansouri | Experiment 2 Lead + Evaluation Lead | - Leads Experiment 2: implements and trains the SegFormer transfer learning pipeline (COD10K pretraining → ACD1K fine-tuning), performs hyperparameter tuning for both stages, and evaluates on the final test set. <br/>- Develops and maintains all evaluation scripts: mIoU, F1/Dice, MAE, FPR on noise images, terrain-stratified breakdowns, and cross-experiment comparison visualisations. <br/>- Leads system integration and supports debugging across all three pipelines. <br/>- Contributes to proposal and report (Success Criteria, Experiments & Results, Discussion — Analysis of Results and Failure Cases & Limitations). |

---

## Project Overview

This project investigates whether cross-domain transfer learning from **natural animal camouflage** (COD10K) can improve pixel-level segmentation of **camouflaged military personnel** (ACD1K). We compare a CNN baseline (SINetV2) against a Transformer-based architecture (SegFormer-B2) across three experimental conditions and evaluate cross-environment generalisation across forest, desert/rocky, and snow terrains.

**Research Gap:** Existing camouflage detection research focuses predominantly on animal camouflage. The cross-domain transfer from animal → military camouflage is largely untested, and no published study has evaluated terrain-stratified generalisation on military-specific imagery.

---

## Datasets

| Dataset | Images | Split | Domain | Source |
|---|---|---|---|---|
| COD10K | 10,000 | 6,000 / 4,000 train/test | Natural animal camouflage | [Kaggle](https://www.kaggle.com/datasets/ivanomelchenkoim11/cod10k-dataset) |
| ACD1K | 1,078 | 748 / 330 train/test | Military personnel | [Kaggle](https://www.kaggle.com/datasets/aalihhiader/military-camouflage-soldiers-dataset-mcs1k) |
| CAMO | 1,250 | 1,000 / 250 train/test | Mixed natural + artificial | [Official site](https://sites.google.com/view/ltnghia/research/camo) |

> **Data is not included in this repository.** See setup instructions below.

---

## Experimental Design

| Experiment | Architecture | Training Data | Purpose |
|---|---|---|---|
| 1 — CNN Baseline | SINetV2 | ACD1K only | CNN baseline; quantifies performance with military-specific data only |
| 2 — Transfer Learning | SegFormer-B2 | COD10K → ACD1K (two-stage) | Tests cross-domain pretraining benefit: animal → military |
| 3 — Joint Training | SegFormer-B2 | COD10K + CAMO + ACD1K | Tests whether enriched joint distribution improves over sequential transfer |

All three experiments are evaluated on the **same 200-image held-out final test set** (100 ACD1K + 50 COD10K + 50 noise distractors) with terrain-stratified breakdowns (forest, desert/rocky, snow).

---

## Repository Structure

```
AI-final-project/
├── data/                          # Datasets (not tracked by Git)
│   ├── DATA_DOWNLOAD_INSTRUCTIONS.txt
│   ├── COD10K-v3/                 # COD10K dataset
│   ├── dataset-splitM/            # ACD1K dataset
│   └── CAMO-V.1.0-CVIU2019/       # CAMO dataset
├── notebooks/
│   ├── 01_EDA_Binger.ipynb        # Exploratory data analysis
│   ├── 02_train_exp3_Binger.ipynb # Experiment 3 joint training
│   └── 03_evaluate_Binger.ipynb   # Experiment 3 evaluation
├── src/
│   ├── __init__.py                # Package root
│   ├── dataset.py                 # Data loading & preprocessing pipeline
│   ├── evaluate.py                # Evaluation metrics
│   └── generate_splits.py         # Fixed split index generator
├── splits/                        # Version-controlled split index files
│   ├── acd1k_train.json
│   ├── acd1k_val.json
│   ├── final_test_acd1k.json
│   ├── final_test_cod10k.json
│   └── final_test_noise.json
├── outputs/
│   └── figures/                   # EDA and result figures (not tracked)
├── .gitignore
├── README.md
└── requirements.txt
```

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

See `data/DATA_DOWNLOAD_INSTRUCTIONS.txt` for detailed instructions.

### 4. Verify dataset loading

```bash
python src/dataset.py data/
```

All three dataset configurations should print ✅.

---

## Usage

### EDA (Binger)

```bash
# Open in Google Colab or JupyterLab
notebooks/01_EDA_Binger.ipynb
```

### Experiment 1 — SINetV2 CNN Baseline (Yansong)

```bash
# Train SINetV2 on ACD1K only
python src/train_exp1.py --epochs 100 --batch_size 16

# Evaluate on final test set
python src/evaluate.py --weights outputs/exp1_best.pth --experiment 1
```

### Experiment 2 — SegFormer Transfer Learning (Sepehr)

```bash
# Stage 1: Pretrain on COD10K
python src/train_exp2.py --stage 1 --epochs 50 --batch_size 16

# Stage 2: Fine-tune on ACD1K
python src/train_exp2.py --stage 2 --weights outputs/exp2_stage1_best.pth --epochs 50

# Evaluate on final test set
python src/evaluate.py --weights outputs/exp2_best.pth --experiment 2
```

### Experiment 3 — SegFormer Joint Training (Binger)

```bash
# Train on COD10K + CAMO + ACD1K jointly
# See notebooks/02_train_exp3_Binger.ipynb (Google Colab, A100 GPU)

# Evaluate on final test set
# See notebooks/03_evaluate_Binger.ipynb
```

---

## Normalization Constants

Computed from training sets (values normalized to [0, 1]):

| Dataset | Mean (R, G, B) | Std (R, G, B) |
|---|---|---|
| COD10K | (0.407, 0.424, 0.340) | (0.208, 0.198, 0.194) |
| ACD1K | (0.411, 0.405, 0.327) | (0.196, 0.196, 0.184) |
| CAMO | (0.479, 0.461, 0.360) | (0.197, 0.195, 0.185) |
| Joint | (0.432, 0.430, 0.342) | (0.200, 0.196, 0.188) |

---

## Models

### SINetV2 (Experiment 1)
- Two-stage CNN architecture with texture-enhanced modules and group-reversal attention
- Input resolution: 352×352
- Reference: Fan et al., IEEE TPAMI 2022

### SegFormer-B2 (Experiments 2 & 3)
- Hierarchical Vision Transformer encoder (MiT-B2) + lightweight all-MLP decoder
- Initialised from `nvidia/segformer-b2-finetuned-ade-512-512`
- Input resolution: 512×512
- Reference: Xie et al., NeurIPS 2021

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **mIoU** | Mean Intersection-over-Union — primary segmentation metric |
| **F1 / Dice** | Harmonic mean of precision and recall on foreground pixels |
| **MAE** | Mean Absolute Error between predicted and ground-truth masks |
| **FPR** | False Positive Rate on 50 noise distractor images |

Terrain-stratified breakdowns (forest, desert/rocky, snow) and cross-experiment comparison visualisations are generated by `src/evaluate.py`.

---

## References

1. Fan et al., "Camouflaged Object Detection," CVPR 2020.
2. Fan et al., "Concealed Object Detection," IEEE TPAMI 2022.
3. Xie et al., "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers," NeurIPS 2021.
4. Haider et al., "Identification of Camouflage Military Individuals with DFAN and SINetV2," *Scientific Reports*, 2025.
5. Le et al., "Anabranch Network for Camouflaged Object Segmentation," *CVIU*, 2019.

---

## License

For academic use only. Datasets are subject to their respective licenses.