"""Few-shot anomaly detection model for VisA (structural anomalies only).

VisA contains only structural anomalies (scratches, dents, colour spots, missing parts),
so the composition/histogram matching logic specific to MVTec LOCO logical anomalies
is omitted. Detection relies on CLIP + DINOv2 PatchCore ensemble.

pred_score = max(s_clip, s_dinov2)  — raw patchcore score, no sigmoid calibration.
AUROC and F1Max are rank-based so absolute scale does not matter, and sigmoid
calibration causes saturation (all scores → 1.0) when k-shot variance is near zero.
"""

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import v2

import open_clip_local as open_clip
from dinov2.dinov2.hub.backbones import dinov2_vitl14


def _setup_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MyModel(nn.Module):
    """PatchCore ensemble (CLIP + DINOv2) for few-shot structural AD on VisA.

    pred_score = max(s_clip, s_dinov2)  — raw cosine-distance patchcore score.
    """

    def __init__(self) -> None:
        super().__init__()

        _setup_seed(42)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.transform = v2.Compose([
            v2.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

        # CLIP ViT-L/14 DataComp-1B
        self.model_clip, _, _ = open_clip.create_model_and_transforms(
            'hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K'
        )
        self.model_clip.eval()
        self.feature_list = [6, 12, 18, 24]
        self.vision_width = 1024

        # DINOv2 ViT-L/14
        self.model_dinov2 = dinov2_vitl14()
        self.model_dinov2.to(self.device)
        self.model_dinov2.eval()
        self.feature_list_dinov2 = [6, 12, 18, 24]
        self.vision_width_dinov2 = 1024

        self.feat_size = 64        # spatial resolution after upsampling
        self.ori_feat_size = 32    # native ViT-L/14 patch grid (448/14 ~= 32)
        self.align_corners = True
        self.antialias = True
        self.inter_mode = 'bilinear'

        # Populated by setup()
        self.mem_clip = None
        self.mem_dinov2 = None
        self.k_shot = 4
        self.class_name = ""
        self.visualization = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_viz(self, viz):
        self.visualization = viz

    def setup(self, data):
        few_shot_samples = data["few_shot_samples"]
        self.class_name = data["dataset_category"]
        self.k_shot = few_shot_samples.size(0)

        imgs = self.transform(few_shot_samples).to(self.device)
        imgs = F.interpolate(imgs, size=(448, 448),
                             mode=self.inter_mode, align_corners=self.align_corners,
                             antialias=self.antialias)

        self.mem_clip   = self._extract_clip(imgs)
        self.mem_dinov2 = self._extract_dinov2(imgs)

    def forward(self, batch, batch_path):
        batch = self.transform(batch).to(self.device)

        clip_feats   = self._extract_clip(batch)
        dinov2_feats = self._extract_dinov2(batch)

        s_clip,   map_clip   = self._patchcore(clip_feats,   self.mem_clip,   len(self.feature_list))
        s_dinov2, map_dinov2 = self._patchcore(dinov2_feats, self.mem_dinov2, len(self.feature_list_dinov2))

        anomaly_map = (map_clip + map_dinov2) / 2.0
        # Raw patchcore score (max cosine distance to memory bank).
        # Calibration via sigmoid is omitted: AUROC and F1Max are rank-based
        # so absolute scale does not affect results, and sigmoid saturation
        # causes all scores to collapse to 1.0 when k-shot std is near zero.
        pred_score = max(s_clip, s_dinov2)

        return {
            'pred_score':  torch.tensor(pred_score,  dtype=torch.float32),
            'anomaly_map': torch.tensor(anomaly_map, dtype=torch.float32),
        }

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_clip(self, batch):
        bs = batch.shape[0]
        with torch.no_grad():
            _, patch_tokens, _ = self.model_clip.encode_image(batch, self.feature_list)
            patch_tokens = [p[:, 1:, :] for p in patch_tokens]
            patch_tokens = [p.reshape(bs * p.shape[1], p.shape[2]) for p in patch_tokens]
            feats = torch.cat(patch_tokens, dim=-1)
            feats = feats.view(bs, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            feats = F.interpolate(feats, size=(self.feat_size, self.feat_size),
                                  mode=self.inter_mode, align_corners=self.align_corners)
            feats = feats.permute(0, 2, 3, 1).view(-1, self.vision_width * len(self.feature_list))
            feats = F.normalize(feats, p=2, dim=-1)
        return feats

    def _extract_dinov2(self, batch):
        bs = batch.shape[0]
        with torch.no_grad():
            patch_tokens = self.model_dinov2.forward_features(batch, out_layer_list=self.feature_list_dinov2)
            feats = torch.cat(patch_tokens, dim=-1)
            feats = feats.view(bs, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            feats = F.interpolate(feats, size=(self.feat_size, self.feat_size),
                                  mode=self.inter_mode, align_corners=self.align_corners)
            feats = feats.permute(0, 2, 3, 1).view(-1, self.vision_width_dinov2 * len(self.feature_list_dinov2))
            feats = F.normalize(feats, p=2, dim=-1)
        return feats

    # ------------------------------------------------------------------
    # PatchCore scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _patchcore(query, memory, n_layers):
        feat_len  = query.shape[0]
        feat_size = int(round(feat_len ** 0.5))

        layer_maps = []
        for q_chunk, m_chunk in zip(query.chunk(n_layers, dim=-1), memory.chunk(n_layers, dim=-1)):
            q   = F.normalize(q_chunk, dim=-1)
            m   = F.normalize(m_chunk, dim=-1)
            sim = q @ m.T
            amap = (1 - sim.max(dim=1)[0]).cpu().numpy()
            layer_maps.append(amap)

        avg_map = np.stack(layer_maps).mean(0)
        score   = float(avg_map.max())
        spatial = avg_map.reshape(feat_size, feat_size)
        return score, spatial

