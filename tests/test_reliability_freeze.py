from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn

from dino_slam3.training.trainer import _enforce_train_modes


class _TinyReliabilityModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.heat = nn.Conv2d(8, 1, 1)
        self.rel = nn.Conv2d(8, 1, 1)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        return self.heat(feat), self.rel(feat)


class ReliabilityFreezeTests(unittest.TestCase):
    def test_bn_running_stats_frozen_in_rel_only_mode(self) -> None:
        torch.manual_seed(0)
        model = _TinyReliabilityModel()

        # Reliability-only train: freeze backbone + heat, train rel head.
        for n, p in model.named_parameters():
            p.requires_grad_(n.startswith("rel."))

        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        x = torch.randn(4, 3, 24, 24)

        bn = model.backbone[1]
        before_mean = bn.running_mean.detach().clone()
        before_var = bn.running_var.detach().clone()

        model.train()
        _enforce_train_modes(model, freeze_non_trainable_modules=True, freeze_bn_running_stats=True)
        for _ in range(5):
            _enforce_train_modes(model, freeze_non_trainable_modules=True, freeze_bn_running_stats=True)
            _, rel = model(x)
            loss = rel.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        self.assertTrue(torch.allclose(before_mean, bn.running_mean, atol=0, rtol=0))
        self.assertTrue(torch.allclose(before_var, bn.running_var, atol=0, rtol=0))

    def test_frozen_branch_output_stability(self) -> None:
        torch.manual_seed(1)
        model = _TinyReliabilityModel()
        for n, p in model.named_parameters():
            p.requires_grad_(n.startswith("rel."))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-3)

        x = torch.randn(2, 3, 24, 24)
        model.train()
        _enforce_train_modes(model, freeze_non_trainable_modules=True, freeze_bn_running_stats=True)
        heat_before, _ = model(x)
        heat_before = heat_before.detach().clone()

        for _ in range(8):
            _enforce_train_modes(model, freeze_non_trainable_modules=True, freeze_bn_running_stats=True)
            _, rel = model(x)
            loss = (rel ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        heat_after, _ = model(x)
        self.assertTrue(torch.allclose(heat_before, heat_after.detach(), atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
