from __future__ import annotations

import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dino_slam3.models.heads import Heads


class HeadsVariantsTests(unittest.TestCase):
    def test_heads_v1_shape_contract(self) -> None:
        m = Heads(in_ch=64, descriptor_dim=32, variant="v1", use_offset=True, use_reliability=True)
        x = torch.rand(2, 64, 8, 8)
        out = m(x)
        self.assertEqual(tuple(out.heatmap.shape), (2, 1, 8, 8))
        self.assertEqual(tuple(out.desc.shape), (2, 32, 8, 8))
        self.assertEqual(tuple(out.offset.shape), (2, 2, 8, 8))
        self.assertEqual(tuple(out.reliability.shape), (2, 1, 8, 8))

    def test_heads_v2_shape_contract(self) -> None:
        m = Heads(
            in_ch=64,
            descriptor_dim=32,
            variant="v2",
            head_channels=96,
            tower_depth=2,
            norm="group",
            act="silu",
            use_offset=True,
            use_reliability=True,
        )
        x = torch.rand(2, 64, 8, 8)
        out = m(x)
        self.assertEqual(tuple(out.heatmap.shape), (2, 1, 8, 8))
        self.assertEqual(tuple(out.desc.shape), (2, 32, 8, 8))
        self.assertEqual(tuple(out.offset.shape), (2, 2, 8, 8))
        self.assertEqual(tuple(out.reliability.shape), (2, 1, 8, 8))

    def test_heads_v3_shape_contract(self) -> None:
        m = Heads(
            in_ch=64,
            descriptor_dim=32,
            variant="v3",
            head_channels=96,
            tower_depth=3,
            norm="group",
            act="silu",
            use_offset=True,
            use_reliability=True,
        )
        x = torch.rand(2, 64, 8, 8)
        out = m(x)
        self.assertEqual(tuple(out.heatmap.shape), (2, 1, 8, 8))
        self.assertEqual(tuple(out.desc.shape), (2, 32, 8, 8))
        self.assertEqual(tuple(out.offset.shape), (2, 2, 8, 8))
        self.assertEqual(tuple(out.reliability.shape), (2, 1, 8, 8))

    def test_heads_v4_dual_desc_shape_contract(self) -> None:
        m = Heads(
            in_ch=64,
            descriptor_dim=32,
            variant="v4_dual_desc",
            head_channels=96,
            tower_depth=2,
            norm="group",
            act="silu",
            use_offset=True,
            use_reliability=True,
        )
        x = torch.rand(2, 64, 8, 8)
        out = m(x)
        self.assertEqual(tuple(out.heatmap.shape), (2, 1, 8, 8))
        self.assertEqual(tuple(out.desc.shape), (2, 32, 8, 8))
        self.assertEqual(tuple(out.offset.shape), (2, 2, 8, 8))
        self.assertEqual(tuple(out.reliability.shape), (2, 1, 8, 8))
        self.assertTrue(torch.isfinite(out.desc).all().item())

    def test_heads_v5_offset_gated_shape_contract(self) -> None:
        m = Heads(
            in_ch=64,
            descriptor_dim=32,
            variant="v5_offset_gated",
            head_channels=96,
            tower_depth=2,
            norm="group",
            act="silu",
            use_offset=True,
            use_reliability=True,
        )
        x = torch.rand(2, 64, 8, 8)
        out = m(x)
        self.assertEqual(tuple(out.heatmap.shape), (2, 1, 8, 8))
        self.assertEqual(tuple(out.desc.shape), (2, 32, 8, 8))
        self.assertEqual(tuple(out.offset.shape), (2, 2, 8, 8))
        self.assertEqual(tuple(out.reliability.shape), (2, 1, 8, 8))
        self.assertTrue(torch.isfinite(out.offset).all().item())

    def test_heads_v6_dual_desc_relgate_shape_contract(self) -> None:
        m = Heads(
            in_ch=64,
            descriptor_dim=32,
            variant="v6_dual_desc_relgate",
            head_channels=96,
            tower_depth=2,
            norm="group",
            act="silu",
            use_offset=True,
            use_reliability=True,
            desc_relgate_detach=True,
        )
        x = torch.rand(2, 64, 8, 8)
        out = m(x)
        self.assertEqual(tuple(out.heatmap.shape), (2, 1, 8, 8))
        self.assertEqual(tuple(out.desc.shape), (2, 32, 8, 8))
        self.assertEqual(tuple(out.offset.shape), (2, 2, 8, 8))
        self.assertEqual(tuple(out.reliability.shape), (2, 1, 8, 8))
        self.assertTrue(torch.isfinite(out.desc).all().item())

    def test_heads_v7_heatmap_mod_desc_shape_contract(self) -> None:
        m = Heads(
            in_ch=64,
            descriptor_dim=32,
            variant="v7_heatmap_mod_desc",
            head_channels=96,
            tower_depth=2,
            norm="group",
            act="silu",
            use_offset=True,
            use_reliability=True,
        )
        x = torch.rand(2, 64, 8, 8)
        out = m(x)
        self.assertEqual(tuple(out.heatmap.shape), (2, 1, 8, 8))
        self.assertEqual(tuple(out.desc.shape), (2, 32, 8, 8))
        self.assertEqual(tuple(out.offset.shape), (2, 2, 8, 8))
        self.assertEqual(tuple(out.reliability.shape), (2, 1, 8, 8))
        self.assertTrue(torch.isfinite(out.desc).all().item())

    def test_heads_v8_offset_residual_shape_contract(self) -> None:
        m = Heads(
            in_ch=64,
            descriptor_dim=32,
            variant="v8_offset_residual",
            head_channels=96,
            tower_depth=2,
            norm="group",
            act="silu",
            use_offset=True,
            use_reliability=True,
        )
        x = torch.rand(2, 64, 8, 8)
        out = m(x)
        self.assertEqual(tuple(out.heatmap.shape), (2, 1, 8, 8))
        self.assertEqual(tuple(out.desc.shape), (2, 32, 8, 8))
        self.assertEqual(tuple(out.offset.shape), (2, 2, 8, 8))
        self.assertEqual(tuple(out.reliability.shape), (2, 1, 8, 8))
        self.assertTrue(torch.isfinite(out.offset).all().item())

    def test_v8_offset_residual_branch_contributes(self) -> None:
        m = Heads(
            in_ch=32,
            descriptor_dim=16,
            variant="v8_offset_residual",
            head_channels=64,
            tower_depth=1,
            norm="group",
            act="silu",
            use_offset=True,
            use_reliability=True,
        )
        with torch.no_grad():
            for p in m.parameters():
                p.zero_()
            m.off_res.bias.fill_(2.0)
        x = torch.rand(1, 32, 8, 8)
        out = m(x)
        self.assertGreater(float(out.offset.abs().mean().item()), 0.05)

    def test_heads_v9_saliencygate_shape_contract(self) -> None:
        m = Heads(
            in_ch=64,
            descriptor_dim=32,
            variant="v9_dual_desc_saliencygate",
            head_channels=96,
            tower_depth=2,
            norm="group",
            act="silu",
            use_offset=False,
            use_reliability=True,
        )
        x = torch.rand(2, 64, 8, 8)
        out = m(x)
        self.assertEqual(tuple(out.heatmap.shape), (2, 1, 8, 8))
        self.assertEqual(tuple(out.desc.shape), (2, 32, 8, 8))
        self.assertIsNone(out.offset)
        self.assertEqual(tuple(out.reliability.shape), (2, 1, 8, 8))

    def test_heads_v10_layernorm_shape_contract(self) -> None:
        m = Heads(
            in_ch=64,
            descriptor_dim=32,
            variant="v10_dual_desc_layernorm",
            head_channels=96,
            tower_depth=2,
            norm="group",
            act="silu",
            use_offset=True,
            use_reliability=True,
        )
        x = torch.rand(2, 64, 8, 8)
        out = m(x)
        self.assertEqual(tuple(out.heatmap.shape), (2, 1, 8, 8))
        self.assertEqual(tuple(out.desc.shape), (2, 32, 8, 8))
        self.assertEqual(tuple(out.offset.shape), (2, 2, 8, 8))
        self.assertEqual(tuple(out.reliability.shape), (2, 1, 8, 8))

    def test_heads_v11_offset_confidence_suppresses_low_conf(self) -> None:
        m = Heads(
            in_ch=32,
            descriptor_dim=16,
            variant="v11_offset_confidence",
            head_channels=64,
            tower_depth=1,
            norm="group",
            act="silu",
            use_offset=True,
            use_reliability=True,
        )
        with torch.no_grad():
            for p in m.parameters():
                p.zero_()
            m.off.bias.fill_(4.0)
            if m.off_conf is not None:
                m.off_conf.bias.fill_(-30.0)
            m.rel.bias.fill_(-30.0)
        x = torch.rand(1, 32, 8, 8)
        out = m(x)
        self.assertLess(float(out.offset.abs().max().item()), 3e-2)

    def test_heads_v12_moe_desc_shape_contract(self) -> None:
        m = Heads(
            in_ch=64,
            descriptor_dim=32,
            variant="v12_moe_desc",
            head_channels=96,
            tower_depth=2,
            norm="group",
            act="silu",
            use_offset=False,
            use_reliability=True,
        )
        x = torch.rand(2, 64, 8, 8)
        out = m(x)
        self.assertEqual(tuple(out.heatmap.shape), (2, 1, 8, 8))
        self.assertEqual(tuple(out.desc.shape), (2, 32, 8, 8))
        self.assertIsNone(out.offset)
        self.assertEqual(tuple(out.reliability.shape), (2, 1, 8, 8))
        self.assertTrue(torch.isfinite(out.desc).all().item())

    def test_heads_v13_scale_pyramid_desc_shape_contract(self) -> None:
        m = Heads(
            in_ch=64,
            descriptor_dim=32,
            variant="v13_scale_pyramid_desc",
            head_channels=96,
            tower_depth=2,
            norm="group",
            act="silu",
            use_offset=False,
            use_reliability=True,
        )
        x = torch.rand(2, 64, 8, 8)
        out = m(x)
        self.assertEqual(tuple(out.heatmap.shape), (2, 1, 8, 8))
        self.assertEqual(tuple(out.desc.shape), (2, 32, 8, 8))
        self.assertIsNone(out.offset)
        self.assertEqual(tuple(out.reliability.shape), (2, 1, 8, 8))
        self.assertTrue(torch.isfinite(out.desc).all().item())

    def test_heads_v14_det_rel_crossgate_shape_contract(self) -> None:
        m = Heads(
            in_ch=64,
            descriptor_dim=32,
            variant="v14_det_rel_crossgate",
            head_channels=96,
            tower_depth=2,
            norm="group",
            act="silu",
            use_offset=False,
            use_reliability=True,
        )
        x = torch.rand(2, 64, 8, 8)
        out = m(x)
        self.assertEqual(tuple(out.heatmap.shape), (2, 1, 8, 8))
        self.assertEqual(tuple(out.desc.shape), (2, 32, 8, 8))
        self.assertIsNone(out.offset)
        self.assertEqual(tuple(out.reliability.shape), (2, 1, 8, 8))
        self.assertTrue(torch.isfinite(out.desc).all().item())

    def test_v5_offset_gate_suppresses_offset_when_reliability_low(self) -> None:
        m = Heads(
            in_ch=32,
            descriptor_dim=16,
            variant="v5_offset_gated",
            head_channels=64,
            tower_depth=1,
            norm="group",
            act="silu",
            use_offset=True,
            use_reliability=True,
        )
        with torch.no_grad():
            for p in m.parameters():
                p.zero_()
            # Produce strong raw offset before gate.
            m.off.bias.fill_(3.0)
            # Force low reliability -> small gate.
            m.rel.bias.fill_(-20.0)
        x = torch.rand(1, 32, 8, 8)
        out = m(x)
        self.assertLess(float(out.offset.abs().max().item()), 1e-3)


if __name__ == "__main__":
    unittest.main()
