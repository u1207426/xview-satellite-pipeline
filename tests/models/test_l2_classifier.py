# tests/models/test_l2_classifier.py
import torch
import pytest
from transformers import ViTConfig, ViTModel
from src.models.l2_classifier import L2Classifier


def make_tiny_classifier() -> L2Classifier:
    tiny_cfg = ViTConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        image_size=224,
        patch_size=16,
    )
    tiny_vit = ViTModel(tiny_cfg)
    return L2Classifier(_backbone=tiny_vit, struct_input_dim=31)


def test_forward_output_shape():
    model = make_tiny_classifier()
    pixel_values = torch.randn(2, 3, 224, 224)
    struct_features = torch.randn(2, 31)
    out = model(pixel_values, struct_features)
    assert out.shape == (2, 1)


def test_freeze_backbone_stops_gradients():
    model = make_tiny_classifier()
    model.freeze_backbone()
    for param in model.backbone.parameters():
        assert not param.requires_grad


def test_unfreeze_last_n_blocks_enables_gradients():
    model = make_tiny_classifier()
    model.freeze_backbone()
    model.unfreeze_last_n_blocks(1)
    last_block_params = list(model.backbone.encoder.layer[-1].parameters())
    assert any(p.requires_grad for p in last_block_params)


def test_struct_branch_parameters_always_trainable():
    model = make_tiny_classifier()
    model.freeze_backbone()
    for param in model.struct_branch.parameters():
        assert param.requires_grad


def test_fusion_head_parameters_always_trainable():
    model = make_tiny_classifier()
    model.freeze_backbone()
    for param in model.fusion_head.parameters():
        assert param.requires_grad
