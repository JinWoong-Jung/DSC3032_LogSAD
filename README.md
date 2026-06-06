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

Place all datasets under the `datasets/` directory:

```
datasets/
├── MVTec_LOCO/
│   ├── breakfast_box/
│   ├── juice_bottle/
│   ├── pushpins/
│   ├── screw_bag/
│   └── splicing_connectors/
├── VisA/
│   ├── candle/
│   ├── capsules/
│   ├── ...
│   └── split_csv/
└── MVTec_AD/
    ├── bottle/
    ├── cable/
    └── ...
```

### MVTec LOCO

Download from the [MVTec LOCO website](https://www.mvtec.com/company/research/datasets/mvtec-loco).

### VisA

```bash
# Download and extract
wget -O VisA_20220922.tar "https://amazon.com/.../VisA_20220922.tar"
tar -xf VisA_20220922.tar -C datasets/VisA/
rm VisA_20220922.tar
```

`visa_pytorch/` (anomalib-compatible split) is created automatically on first run.

### MVTec AD

```bash
wget -O datasets/mvtec_anomaly_detection.tar.xz \
  "https://www.mydrive.ch/shares/150996/b52ecdcbf521176e9db9c731f2304b27/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz"
mkdir -p datasets/MVTec_AD
tar -xf datasets/mvtec_anomaly_detection.tar.xz -C datasets/MVTec_AD/
rm datasets/mvtec_anomaly_detection.tar.xz
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

### MVTec AD — Full Reimplementation (Table 10)

Runs all 15 categories × {1-shot, 2-shot, 4-shot}. Results saved to `outputs/MVTec_AD/`.

```bash
bash run_mvtec_table10.sh
```

Categories: `bottle` · `cable` · `capsule` · `carpet` · `grid` · `hazelnut` · `leather` · `metal_nut` · `pill` · `screw` · `tile` · `toothbrush` · `transistor` · `wood` · `zipper`

### MVTec AD — Single Category

```bash
python evaluation_MVTec.py \
  --module_path model_ensemble_few_shot_visa \
  --category bottle \
  --dataset_path datasets/MVTec_AD \
  --k_shot 4
```

---

## Demo Notebooks

Interactive demo for `juice_bottle`, 4-shot protocol. Run from `demo/` with the `logsad` conda environment.

```
demo/
├── Group6-fulltraining.ipynb   # Setup & evaluation (training-free equivalent of "full-training"):
│                               # frozen model loading, MoT proposals, memory bank, results vs paper
└── Group6-demo.ipynb           # Live inference + anomaly-map visualisation
```

> **Naming.** LogSAD is **training-free (Category D)** — there is no fine-tuning step, so
> `Group6-fulltraining.ipynb` is the training-free equivalent (setup + reproduced evaluation).
> Both notebooks ship with their cell outputs already populated as a fallback.

**Demo environment (presentation, room 31709):**

- **Primary** — VS Code **Remote-SSH from the classroom machine into the GPU server `aicoss220`**, then open the notebooks. Live inference needs a GPU, which the classroom machine lacks.
- **Secondary** — Google Colab (GPU runtime); `Group6-demo.ipynb` has a Colab bootstrap cell. Pre-warm the model load and pass phone auth before presenting.
- **Fallback** — a screen recording of the live run; the saved cell outputs also stand in if the kernel disconnects.

**Recommended presentation workflow:**

```
Before presentation
├── Run Group6-fulltraining.ipynb entirely (pre-computed outputs, ~3 min)
└── Run Group6-demo.ipynb sections 0–3 (model load + 4-shot memory bank, ~3–5 min) — keep kernel warm

During presentation (~1 min live)
└── Run Group6-demo.ipynb sections 4–7 (image selection → inference → visualisation)
    (marked "▶ LIVE FROM HERE" in the notebook)
```

To open on the GPU server via VS Code:

```bash
cd /home/gaya6/LogSAD/demo
jupyter notebook   # or open .ipynb directly in VS Code (Remote-SSH)
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
│   ├── MVTec_LOCO/
│   ├── VisA/
│   └── MVTec_AD/
├── memory_bank/                     # Pre-computed full-data coreset .pt files (~5–6 GB each)
├── checkpoint/
│   └── sam_vit_h_4b8939.pth
├── outputs/
│   ├── MVTec_LOCO/results.txt
│   ├── VisA/results.txt
│   └── MVTec_AD/results.txt
└── demo/
    ├── Group6-fulltraining.ipynb   # setup + reproduced evaluation (training-free)
    └── Group6-demo.ipynb           # live inference + visualisation
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
