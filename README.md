# LogSAD: Reimplementation & Demo (Group 6)

<p align="center">
  <p align="center">
    <a>Jinwoong Jung</a>&emsp;
    <a>Seungtaek Lee</a>&emsp;
    <a>Jeongsoh Hur</a>&emsp;
    <a>Fatima Tuz Zahra</a>
  </p>
  <p align="center">
    <i>Sungkyunkwan University</i><br>
  </p>
   <p align="center">
    <i>Deep Learning 1: Foundations and Image Processing(DSC3032)</i><br>
  </p>
</p>

## Project Overview

<div>
  <a href="https://arxiv.org/abs/2503.18325"><img src="https://img.shields.io/static/v1?label=Arxiv&message=LogSAD&color=red&logo=arxiv"></a> &ensp;
  <a href="https://github.com/zhang0jhon/LogSAD.git"><img src="https://img.shields.io/static/v1?label=Original%20Code&message=GitHub&color=black&logo=github"></a>
</div>

This repository is the reimplementation of 
### LogSAD: Towards Training-free Anomaly Detection with Vision and Language Foundation Models.

This repository is organized for reproducing the project results on three datasets and for running a short `juice_bottle` demo.

---

## 1. Environment Setup

Clone this repository and move into the project folder:

```bash
git clone https://github.com/JinWoong-Jung/DSC3032_LogSAD.git LogSAD
cd LogSAD
```

Create and activate a conda environment:

```bash
conda create -n logsad python=3.10 -y
conda activate logsad
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the SAM ViT-H checkpoint:

```bash
mkdir -p checkpoint
wget -P checkpoint/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Expected checkpoint path:

```text
checkpoint/sam_vit_h_4b8939.pth
```

---

## 2. Dataset Setup

Place datasets under `datasets/`:

```text
datasets/
├── MVTec_LOCO/
├── VisA/
└── MVTec_AD/
```

Dataset links:

- MVTec LOCO: https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad
- VisA: [amazon-science/spot-diff.git](https://github.com/amazon-science/spot-diff.git)
- MVTec AD: https://www.mvtec.com/research-teaching/datasets/mvtec-ad

The evaluation scripts assume the dataset folders are already prepared in this structure.

---

## 3. Run Evaluation

Run from the repository root:

```bash
cd LogSAD
conda activate logsad
```

### MVTec LOCO

```bash
bash run_loco_table9.sh
```

### VisA

```bash
bash run_visa_table11.sh
```

### MVTec AD

```bash
bash run_ad_table10.sh
```

Results are saved under:

```text
outputs/MVTec_LOCO/
outputs/VisA/
outputs/MVTec_AD/
```

---

## 4. Demo

The demo notebooks are in `demo/`.

```text
demo/
├── Group6-fulltraining.ipynb
├── Group6-demo.ipynb
├── statistic_scores_model_ensemble_few_shot_val.pkl
└── images/
    ├── shot/
    │   └── good/
    │       ├── 000.png
    │       ├── 001.png
    │       ├── 002.png
    │       └── 003.png
    └── test/
        ├── good/
        │   ├── 000.png
        │   ├── 001.png
        │   └── 002.png
        ├── logical_anomalies/
        │   ├── 000.png
        │   ├── 001.png
        │   └── 002.png
        └── structural_anomalies/
            ├── 000.png
            ├── 001.png
            └── 002.png
```

`Group6-demo.ipynb` is the main live demo notebook. It uses the `juice_bottle` category with a 4-shot, training-free setup:

- `demo/images/shot/good/`: normal reference shots used to build the memory bank
- `demo/images/test/good/`: normal test images
- `demo/images/test/logical_anomalies/`: logical anomaly examples
- `demo/images/test/structural_anomalies/`: structural anomaly examples

Recommended demo flow:

1. Run the setup cells first to load the model and build the memory bank.
2. Run the live section to select images, run inference, and show visual results.
3. Structural anomalies are shown with heatmaps; logical anomalies are shown with detector score diagnostics.

Open the notebook from the project folder:

```bash
cd LogSAD
conda activate logsad
jupyter notebook demo/Group6-demo.ipynb
```

---

## 5. Project Files

```text
LogSAD/
├── evaluation.py
├── evaluation_VisA.py
├── evaluation_MVTec.py
├── model_ensemble.py
├── model_ensemble_few_shot.py
├── model_ensemble_few_shot_visa.py
├── run_loco_table9.sh
├── run_visa_table11.sh
├── run_ad_table10.sh
├── datasets/
│   ├── MVTec_LOCO/
│   ├── VisA/
│   └── MVTec_AD/
├── checkpoint/
│   └── sam_vit_h_4b8939.pth
├── outputs/
└── demo/
    ├── Group6-fulltraining.ipynb
    ├── Group6-demo.ipynb
    ├── statistic_scores_model_ensemble_few_shot_val.pkl
    └── images/
        ├── shot/
        └── test/
```

---

## Acknowledgement

This project builds on LogSAD and related foundation-model tools including SAM, OpenCLIP, DINOv2, NACLIP, and anomalib.

## Citation

```bibtex
@inproceedings{zhang2025logsad,
  title={Towards Training-free Anomaly Detection with Vision and Language Foundation Models},
  author={Jinjin Zhang, Guodong Wang, Yizhou Jin, Di Huang},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025},
}
```
