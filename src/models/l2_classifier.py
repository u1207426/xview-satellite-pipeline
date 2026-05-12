# src/models/l2_classifier.py
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from transformers import ViTModel


class L2Classifier(nn.Module):
    """
    Dual-input location classifier.

    Image branch:     ViT backbone (default: facebook/vit-mae-base, swap for SatMAE)
    Structural branch: MLP over 31-dim L1 detection feature vector
    Fusion:           concat → classification head → P(military) logit

    Args:
        backbone_name:    HuggingFace model ID. Replace with SatMAE model ID when
                          weights are available (e.g. 'MathFarm/SatMAE').
        struct_input_dim: Dimension of structural feature vector (31).
        _backbone:        Pass a ViTModel instance directly (used in tests to avoid
                          network download).
    """

    def __init__(
        self,
        backbone_name: str = "facebook/vit-mae-base",
        struct_input_dim: int = 31,
        _backbone: Optional[ViTModel] = None,
    ):
        super().__init__()
        self.backbone: ViTModel = (
            _backbone if _backbone is not None
            else ViTModel.from_pretrained(backbone_name)
        )
        hidden_size: int = self.backbone.config.hidden_size

        self.struct_branch = nn.Sequential(
            nn.Linear(struct_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_size + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """Unfreeze last n transformer encoder blocks + final LayerNorm."""
        for layer in self.backbone.encoder.layer[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.backbone.layernorm.parameters():
            param.requires_grad = True

    def forward(
        self,
        pixel_values: torch.Tensor,     # [B, 3, 224, 224]
        struct_features: torch.Tensor,  # [B, 31]
    ) -> torch.Tensor:                  # [B, 1] raw logits
        img_feat = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        struct_feat = self.struct_branch(struct_features)
        return self.fusion_head(torch.cat([img_feat, struct_feat], dim=1))
