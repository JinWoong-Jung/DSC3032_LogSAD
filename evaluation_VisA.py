# OWN CODE (Group 6): VisA evaluation entry point — not in the original LogSAD repo.
# Original repo only provides evaluation.py for MVTec LOCO; this file extends support
# to VisA (12 structural-anomaly categories) to reproduce Table 11 (CVPR 2025).
"""Evaluation script for VisA dataset (structural anomaly detection).

Adapted from evaluation.py (MVTec LOCO) with the following changes:
- Uses anomalib Visa datamodule instead of MVTecLoco
- Calls prepare_data() to create visa_pytorch/ split if not yet done
- Reports image-level F1-max + AUROC and pixel-level F1-max + AUROC
- No logical/structural sub-metric split (VisA has structural anomalies only)
"""

import argparse
import importlib

import numpy as np
import torch
import torch.nn.functional as F
import logging
from torch import nn

from anomalib.data import Visa
from anomalib.metrics.f1_max import F1Max
from anomalib.metrics.auroc import AUROC
from tabulate import tabulate

DEFAULT_K_SHOT = 4
VISA_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module_path", type=str, required=True)
    parser.add_argument("--class_name", default="MyModel", type=str)
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument(
        "--dataset_path",
        default="/home/gaya6/LogSAD/datasets/VisA",
        type=str,
        help="Root of the VisA dataset (must contain candle/, split_csv/, etc.)",
    )
    parser.add_argument("--category", type=str, required=True,
                        choices=VISA_CATEGORIES)
    parser.add_argument("--k_shot", default=DEFAULT_K_SHOT, type=int)
    parser.add_argument("--viz", action="store_true", default=False)
    return parser.parse_args()


def load_model(module_path, class_name, weights_path):
    model_class = getattr(importlib.import_module(module_path), class_name)
    model = model_class()
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
    return model


def run(module_path, class_name, weights_path, dataset_path, category, k_shot, viz):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = load_model(module_path, class_name, weights_path)
    model.to(device)

    # Visa.prepare_data() converts the raw VisA layout to visa_pytorch/ (MVTec format).
    # Safe to call repeatedly — skips conversion if visa_pytorch/<category>/ already exists.
    datamodule = Visa(root=dataset_path, eval_batch_size=1,
                      image_size=(448, 448), category=category)
    datamodule.prepare_data()
    datamodule.setup()

    if k_shot < 1:
        raise ValueError(f"k_shot must be >= 1, got {k_shot}")
    if k_shot > len(datamodule.train_data):
        raise ValueError(
            f"k_shot={k_shot} exceeds available train samples "
            f"({len(datamodule.train_data)})"
        )

    few_shot_indices = list(range(k_shot))
    model.set_viz(viz)

    # Metrics
    image_f1   = F1Max()
    image_auroc = AUROC()
    pixel_f1   = F1Max()
    pixel_auroc = AUROC()

    # Pass k-shot normal images to model
    setup_data = {
        "few_shot_samples": torch.stack(
            [datamodule.train_data[i]["image"] for i in few_shot_indices]
        ).to(device),
        "few_shot_samples_path": [
            datamodule.train_data[i]["image_path"] for i in few_shot_indices
        ],
        "dataset_category": category,
    }
    model.setup(setup_data)

    for data in datamodule.test_dataloader():
        with torch.no_grad():
            output = model(data["image"].to(device), data["image_path"])

        pred  = output["pred_score"].cpu()
        label = data["label"]

        image_f1.update(pred, label)
        image_auroc.update(pred, label)
        print(data["image_path"], pred.item())

        # Pixel-level metrics (only anomalous samples have non-zero masks)
        if "mask" in data and data["mask"] is not None:
            mask = data["mask"].squeeze(1)   # (B, H, W)
            amap = output["anomaly_map"]     # (feat_size, feat_size)

            # Upsample anomaly map to mask resolution
            amap_up = F.interpolate(
                amap.unsqueeze(0).unsqueeze(0).float(),
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=True,
            ).squeeze()

            pixel_f1.update(amap_up.flatten().cpu(), mask.flatten().long())
            pixel_auroc.update(amap_up.flatten().cpu(), mask.flatten().long())

    # Logging setup
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger("visa_eval")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
                            datefmt="%y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    table = [[
        category,
        str(k_shot),
        str(np.round(image_f1.compute().item()    * 100, decimals=2)),
        str(np.round(image_auroc.compute().item() * 100, decimals=2)),
        str(np.round(pixel_f1.compute().item()    * 100, decimals=2)),
        str(np.round(pixel_auroc.compute().item() * 100, decimals=2)),
    ]]

    results = tabulate(
        table,
        headers=["category", "K-shots",
                 "F1-Max(image)", "AUROC(image)",
                 "F1-Max(pixel)", "AUROC(pixel)"],
        tablefmt="pipe",
    )
    logger.info("\n%s", results)


if __name__ == "__main__":
    args = parse_args()
    run(args.module_path, args.class_name, args.weights_path,
        args.dataset_path, args.category, args.k_shot, args.viz)
