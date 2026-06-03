# LogSAD — Reimplementation (Group 6)

**Paper:** *Towards Training-free Anomaly Detection with Vision and Language Foundation Models*
(CVPR 2025) · [arXiv](https://arxiv.org/abs/2503.18325)

> **Introduction to Deep Learning — Final Project**
> Group 6 · GPU Server: aicoss220

---

## Overview

LogSAD is a **training-free** multi-modal framework for industrial anomaly detection that handles both **structural anomalies** (scratches, dents, contamination) and **logical anomalies** (wrong count, mismatched color/label, violated spatial constraints).

No gradient updates or fine-tuning are performed. The pipeline consists of three stages:

1. **Match-of-Thought (MoT)** — GPT-4V generates interests and compositional rules from a few normal images (one-time offline step).
2. **Multi-granularity Detectors** — CLIP + DINOv2 patch matching (structural) and SAM-based interest/composition matching (logical) run at inference time.
3. **Calibration & Fusion** — Anomaly scores from each detector are standardised and fused via max-pooling.

Foundation models used (all frozen, ~1.3B params total):

| Model | Variant | Role |
|---|---|---|
| CLIP | ViT-L/14 (DataComp-1B) | Patch + composition features |
| DINOv2 | ViT-L/14 | Patch features |
| SAM | ViT-H | Open-vocabulary segmentation |

---

## Installation

```bash
pip install -r requirements.txt
```

Download the SAM ViT-H checkpoint and place it in `checkpoint/`:

```bash
wget -P checkpoint/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

---

## Dataset Setup

### MVTec LOCO (primary benchmark)

```
datasets/MVTec_LOCO/          # local disk
├── breakfast_box/
├── juice_bottle/
├── pushpins/
├── screw_bag/
└── splicing_connectors/
```

### VisA (supplementary)

```bash
# Download & extract to shared storage
tar -xf VisA_20220922.tar -C /mnt/dsc3032-gaya-shared/group6/datasets/VisA/
```

The `datasets/VisA` entry is a symlink to shared storage:

```
datasets/VisA -> /mnt/dsc3032-gaya-shared/group6/datasets/VisA/
```

`visa_pytorch/` (anomalib-compatible split) is created automatically on first run.

### MVTec AD (optional)

```bash
cd /mnt/dsc3032-gaya-shared/group6/datasets
wget -O mvtec_anomaly_detection.tar.xz \
  "https://www.mydrive.ch/shares/150996/.../mvtec_anomaly_detection.tar.xz"
mkdir -p MVTec_AD && tar -xf mvtec_anomaly_detection.tar.xz -C MVTec_AD/
rm mvtec_anomaly_detection.tar.xz
```

---

## Storage Layout

Large files are stored on shared NFS to avoid filling the local `/home` partition:

```
/mnt/dsc3032-gaya-shared/group6/
├── datasets/
│   └── VisA/              # VisA raw images + visa_pytorch split
└── memory_bank/           # Pre-computed full-data coreset .pt files
    ├── mem_patch_feature_clip_*.pt      (~5–6 GB each)
    ├── mem_patch_feature_dinov2_*.pt    (~5–6 GB each)
    └── mem_instance_features_*.pt
```

Symlinks in the project root point to these locations:

```
LogSAD/memory_bank  ->  /mnt/dsc3032-gaya-shared/group6/memory_bank/
LogSAD/datasets/VisA -> /mnt/dsc3032-gaya-shared/group6/datasets/VisA/
```

---

## Evaluation

### MVTec LOCO — Full Reimplementation (Table 9)

Runs all 5 categories × {1-shot, 2-shot, 4-shot, full-data}. Results saved to `outputs/MVTec_LOCO/`.

```bash
bash run_loco_table9.sh
```

**Reproduced results** (`outputs/MVTec_LOCO/results.txt`):

| Protocol  | Breakfast Box | Juice Bottle | Pushpins | Screw Bag | Splicing Con. | Avg F1 | Avg AUROC |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1-shot    | 85.0 / 88.0 | 85.6 / 78.1 | 75.7 / 78.0 | 80.7 / 70.6 | 78.2 / 77.7 | 81.0 | 78.5 |
| 2-shot    | 88.0 / 91.5 | 85.7 / 77.5 | 77.8 / 81.1 | 83.0 / 80.5 | 78.2 / 79.8 | 82.6 | 82.1 |
| 4-shot    | 89.9 / 94.4 | 88.2 / 84.3 | 81.4 / 82.5 | 83.0 / 81.3 | 84.7 / 88.6 | 85.4 | 86.2 |
| full-data | 92.0 / 95.7 | 93.9 / 95.2 | 81.3 / 83.5 | 85.2 / 83.2 | 91.3 / 93.5 | 88.7 | 90.2 |

*(F1-max / AUROC)*

Per-category logs: `outputs/MVTec_LOCO/logs/{1-shot,2-shot,4-shot,full-data}/`

---

### MVTec LOCO — Single Category / Protocol

```bash
# Few-shot (k = 1 / 2 / 4)
python evaluation.py \
  --module_path model_ensemble_few_shot \
  --category juice_bottle \
  --dataset_path datasets/MVTec_LOCO \
  --k_shot 4

# Full-data: compute coreset first, then evaluate
python compute_coreset.py \
  --module_path model_ensemble \
  --category juice_bottle \
  --dataset_path datasets/MVTec_LOCO

python evaluation.py \
  --module_path model_ensemble \
  --category juice_bottle \
  --dataset_path datasets/MVTec_LOCO \
  --k_shot 4
```

Categories: `breakfast_box` · `juice_bottle` · `pushpins` · `screw_bag` · `splicing_connectors`

---

### VisA — Full Reimplementation (Table 4 / Table 11)

Runs all 12 categories × {1-shot, 2-shot, 4-shot}. Results saved to `outputs/VisA/`.

```bash
bash run_visa_table.sh
```

Categories: `candle` · `capsules` · `cashew` · `chewinggum` · `fryum` · `macaroni1` · `macaroni2` · `pcb1` · `pcb2` · `pcb3` · `pcb4` · `pipe_fryum`

### VisA — Single Category

```bash
python evaluation_VisA.py \
  --module_path model_ensemble_few_shot_visa \
  --category candle \
  --dataset_path datasets/VisA \
  --k_shot 4
```

---

## Demo Notebooks

Interactive demo for `juice_bottle`, 4-shot protocol. Run from `demo/` with the `logsad` conda environment.

```
demo/
├── 01_setup_and_results.ipynb   # Model loading, MoT proposals, memory bank, results vs paper
└── 02_live_inference.ipynb      # Live inference + anomaly map visualisation
```

**Recommended presentation workflow:**

```
Before presentation
├── Run 01 entirely (pre-computed outputs, ~3 min)
└── Run 02 cells 1–3 (model load + 4-shot setup, ~3 min)

During presentation (~1 min live)
└── Run 02 cells 4–7 (image selection → inference → visualisation)
```

To open on the GPU server via VS Code:

```bash
cd /home/gaya6/LogSAD/demo
jupyter notebook   # or open .ipynb directly in VS Code
```

---

## Project Structure

```
LogSAD/
├── evaluation.py                    # MVTec LOCO evaluation entry point
├── evaluation_VisA.py               # VisA evaluation entry point
├── compute_coreset.py               # Full-data memory bank computation
├── model_ensemble_few_shot.py       # LogSAD model (MVTec LOCO)
├── model_ensemble_few_shot_visa.py  # LogSAD model — VisA (structural only)
├── model_ensemble.py                # Full-data variant
├── run_loco_table9.sh               # One-command full LOCO reimplementation
├── run_visa_table.sh                # One-command full VisA reimplementation
├── prompt_ensemble.py               # CLIP text prompt utilities
├── datasets/
│   ├── MVTec_LOCO/                  # Local
│   └── VisA -> (NFS symlink)
├── memory_bank/ -> (NFS symlink)    # Pre-computed coreset .pt files
├── checkpoint/
│   └── sam_vit_h_4b8939.pth
├── outputs/
│   ├── MVTec_LOCO/results.txt
│   └── VisA/results.txt
└── demo/
    ├── 01_setup_and_results.ipynb
    └── 02_live_inference.ipynb
```

---

## Acknowledgement

We are grateful for the following projects used in LogSAD:
[SAM](https://github.com/facebookresearch/segment-anything) ·
[OpenCLIP](https://github.com/mlfoundations/open_clip) ·
[DINOv2](https://github.com/facebookresearch/dinov2) ·
[NACLIP](https://github.com/sinahmr/NACLIP) ·
[anomalib](https://github.com/openvinotoolkit/anomalib)

## Citation

```bibtex
@inproceedings{zhang2025logsad,
  title={Towards Training-free Anomaly Detection with Vision and Language Foundation Models},
  author={Jinjin Zhang, Guodong Wang, Yizhou Jin, Di Huang},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025},
}
```
