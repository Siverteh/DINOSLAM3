from __future__ import annotations

import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dino_slam3.models.network import (
    _FusionBlockV4FPN,
    _FusionBlockV5FPNAttn,
    _FusionBlockV6FPNXGate,
    _FusionBlockV7FPNASPP,
    _FusionBlockV8BiFPNLite,
    _FusionBlockV9FPNDeformLite,
    _FusionBlockV10TokenCrossAttn,
    _FusionBlockV11BiFPNDepthAux,
)


class FusionVariantTests(unittest.TestCase):
    def _inputs(self):
        fine_s2 = torch.rand(2, 64, 16, 16, requires_grad=True)
        fine_s4 = torch.rand(2, 64, 8, 8, requires_grad=True)
        dino_s4 = torch.rand(2, 96, 8, 8, requires_grad=True)
        return fine_s2, fine_s4, dino_s4

    def test_v4_fpn_shape_and_backward(self) -> None:
        m = _FusionBlockV4FPN(fine_channels=64, dino_channels=96, out_ch=128)
        s2, s4, d = self._inputs()
        y = m(s2, s4, d)
        self.assertEqual(tuple(y.shape), (2, 128, 8, 8))
        loss = y.mean()
        loss.backward()
        self.assertIsNotNone(s2.grad)
        self.assertIsNotNone(s4.grad)
        self.assertIsNotNone(d.grad)

    def test_v5_fpn_attn_shape_and_backward(self) -> None:
        m = _FusionBlockV5FPNAttn(fine_channels=64, dino_channels=96, out_ch=128)
        s2, s4, d = self._inputs()
        y = m(s2, s4, d)
        self.assertEqual(tuple(y.shape), (2, 128, 8, 8))
        self.assertTrue(torch.isfinite(y).all().item())
        y.sum().backward()
        self.assertIsNotNone(s2.grad)
        self.assertIsNotNone(s4.grad)
        self.assertIsNotNone(d.grad)

    def test_v6_fpn_xgate_shape_and_backward(self) -> None:
        m = _FusionBlockV6FPNXGate(fine_channels=64, dino_channels=96, out_ch=128)
        s2, s4, d = self._inputs()
        y = m(s2, s4, d)
        self.assertEqual(tuple(y.shape), (2, 128, 8, 8))
        self.assertTrue(torch.isfinite(y).all().item())
        y.mean().backward()
        self.assertIsNotNone(s2.grad)
        self.assertIsNotNone(s4.grad)
        self.assertIsNotNone(d.grad)

    def test_v7_fpn_aspp_shape_and_backward(self) -> None:
        m = _FusionBlockV7FPNASPP(fine_channels=64, dino_channels=96, out_ch=128)
        s2, s4, d = self._inputs()
        y = m(s2, s4, d)
        self.assertEqual(tuple(y.shape), (2, 128, 8, 8))
        self.assertTrue(torch.isfinite(y).all().item())
        y.mean().backward()
        self.assertIsNotNone(s2.grad)
        self.assertIsNotNone(s4.grad)
        self.assertIsNotNone(d.grad)

    def test_v8_bifpn_lite_shape_and_backward(self) -> None:
        m = _FusionBlockV8BiFPNLite(fine_channels=64, dino_channels=96, out_ch=128)
        s2, s4, d = self._inputs()
        y = m(s2, s4, d)
        self.assertEqual(tuple(y.shape), (2, 128, 8, 8))
        self.assertTrue(torch.isfinite(y).all().item())
        y.mean().backward()
        self.assertIsNotNone(s2.grad)
        self.assertIsNotNone(s4.grad)
        self.assertIsNotNone(d.grad)

    def test_v9_fpn_deformlite_shape_and_backward(self) -> None:
        m = _FusionBlockV9FPNDeformLite(fine_channels=64, dino_channels=96, out_ch=128)
        s2, s4, d = self._inputs()
        y = m(s2, s4, d)
        self.assertEqual(tuple(y.shape), (2, 128, 8, 8))
        self.assertTrue(torch.isfinite(y).all().item())
        y.mean().backward()
        self.assertIsNotNone(s2.grad)
        self.assertIsNotNone(s4.grad)
        self.assertIsNotNone(d.grad)

    def test_v10_token_crossattn_shape_and_backward(self) -> None:
        m = _FusionBlockV10TokenCrossAttn(fine_channels=64, dino_channels=96, out_ch=128)
        s2, s4, d = self._inputs()
        y = m(s2, s4, d)
        self.assertEqual(tuple(y.shape), (2, 128, 8, 8))
        self.assertTrue(torch.isfinite(y).all().item())
        y.mean().backward()
        self.assertIsNotNone(s2.grad)
        self.assertIsNotNone(s4.grad)
        self.assertIsNotNone(d.grad)

    def test_v11_bifpn_depthaux_shape_and_backward(self) -> None:
        m = _FusionBlockV11BiFPNDepthAux(fine_channels=64, dino_channels=96, out_ch=128)
        s2, s4, d = self._inputs()
        depth_aux = torch.rand(2, 64, 8, 8, requires_grad=True)
        y = m(s2, s4, d, depth_aux_s4=depth_aux)
        self.assertEqual(tuple(y.shape), (2, 128, 8, 8))
        self.assertTrue(torch.isfinite(y).all().item())
        y.mean().backward()
        self.assertIsNotNone(s2.grad)
        self.assertIsNotNone(s4.grad)
        self.assertIsNotNone(d.grad)
        self.assertIsNotNone(depth_aux.grad)


if __name__ == "__main__":
    unittest.main()
