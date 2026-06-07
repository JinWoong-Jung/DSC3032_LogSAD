# LogSAD — Reimplementation (Group 6)

Reimplementation of **LogSAD: Towards Training-free Anomaly Detection with Vision and Language Foundation Models**.

This repository is organized for reproducing the project results on three datasets and for running a short `juice_bottle` demo.

---

## 1. Environment Setup

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

The evaluation scripts assume the dataset folders are already prepared in this structure.

---

## 3. Run Evaluation

Run from the repository root:

```bash
cd /home/gaya6/LogSAD
conda activate logsad
```

### MVTec LOCO

```bash
bash run_loco_table9.sh
```

Results are saved under:

```text
outputs/MVTec_LOCO/
```

### VisA

```bash
bash run_visa_table11.sh
```

Results are saved under:

```text
outputs/VisA/
```

### MVTec AD

```bash
bash run_ad_table10.sh
```

Results are saved under:

```text
outputs/MVTec_AD/
```

---

## 4. Demo

The demo notebooks are in `demo/`.

```text
demo/
├── Group6-fulltraining.ipynb
└── Group6-demo.ipynb
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
cd /home/gaya6/LogSAD
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
├── checkpoint/
├── outputs/
└── demo/
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
