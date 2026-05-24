from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "refocus_vo" / "src"))

from refocus_vo.backbones.dinov3 import DinoTokens, DinoV3Backbone  # noqa: E402
from refocus_vo.dino_dpvo.frontend import DinoProposalFrontend  # noqa: E402


class _FakeBackbone:
    def __init__(self, embed_dim: int = 8, *, inference_tokens: bool = False) -> None:
        self.embed_dim = int(embed_dim)
        self.inference_tokens = bool(inference_tokens)

    def has_trainable_params(self) -> bool:
        return True

    def __call__(self, x: torch.Tensor, *, return_hidden_states: bool, hidden_state_indices) -> DinoTokens:
        batch = int(x.shape[0])
        ht = int(x.shape[-2] // 16)
        wt = int(x.shape[-1] // 16)
        ctx = torch.inference_mode if self.inference_tokens else torch.enable_grad
        with ctx():
            hidden_states = {
                int(idx): torch.ones(batch, self.embed_dim, ht, wt, dtype=x.dtype, device=x.device)
                for idx in hidden_state_indices
            }
            pooled_register = torch.full((batch, self.embed_dim), 0.5, dtype=x.dtype, device=x.device)
            return DinoTokens(
                tokens=torch.zeros(batch, self.embed_dim, ht, wt, dtype=x.dtype, device=x.device),
                hidden_states=hidden_states,
                register_tokens=torch.full((batch, 2, self.embed_dim), 0.5, dtype=x.dtype, device=x.device),
                pooled_register_tokens=pooled_register,
            )


class _FakeTeacher:
    def fuse_layers(self, hidden_states, layer_indices):
        return sum(hidden_states[int(idx)] for idx in layer_indices) / float(len(tuple(layer_indices)))


class DinoV3RegisterContextTests(unittest.TestCase):
    def test_backbone_extracts_register_tokens(self) -> None:
        backbone = DinoV3Backbone(name_or_path="dummy")
        backbone.num_register_tokens = 2
        tokens = torch.arange(1 * 7 * 4, dtype=torch.float32).reshape(1, 7, 4)

        register_tokens = backbone._extract_register_tokens(tokens)
        patch_tokens = backbone._extract_patch_tokens(tokens)

        self.assertEqual(tuple(register_tokens.shape), (1, 2, 4))
        self.assertEqual(tuple(patch_tokens.shape), (1, 4, 4))
        self.assertTrue(torch.equal(register_tokens, tokens[:, 1:3, :]))
        self.assertTrue(torch.equal(patch_tokens, tokens[:, 3:, :]))

    def test_frontend_register_context_changes_fused_features_without_changing_shape(self) -> None:
        images = torch.randn(1, 2, 3, 32, 32)

        base = DinoProposalFrontend.__new__(DinoProposalFrontend)
        nn.Module.__init__(base)
        base.backbone = _FakeBackbone(embed_dim=8)
        base.teacher = _FakeTeacher()
        base.layer_indices = (1, 3)
        base.use_register_context = False
        base.register_context_proj = None
        base.register_context_scale = 0.0
        base.register_context_target = "fused"

        with_register = DinoProposalFrontend.__new__(DinoProposalFrontend)
        nn.Module.__init__(with_register)
        with_register.backbone = _FakeBackbone(embed_dim=8)
        with_register.teacher = _FakeTeacher()
        with_register.layer_indices = (1, 3)
        with_register.use_register_context = True
        with_register.register_context_scale = 0.20
        with_register.register_context_target = "fused"
        with_register.register_context_proj = nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            with_register.register_context_proj.weight.copy_(torch.eye(8))

        fused_base, register_base = DinoProposalFrontend._encode_backbone(base, images)
        fused_reg, register_reg = DinoProposalFrontend._encode_backbone(with_register, images)

        self.assertEqual(tuple(fused_base.shape), (1, 2, 8, 2, 2))
        self.assertEqual(tuple(fused_reg.shape), (1, 2, 8, 2, 2))
        self.assertEqual(tuple(register_reg.shape), (1, 2, 8))
        self.assertEqual(tuple(register_base.shape), (1, 2, 8))
        self.assertTrue(torch.isfinite(fused_reg).all().item())
        self.assertFalse(torch.allclose(fused_base, fused_reg))

    def test_anchor_refresh_target_keeps_fused_features_unchanged(self) -> None:
        images = torch.randn(1, 2, 3, 32, 32)

        base = DinoProposalFrontend.__new__(DinoProposalFrontend)
        nn.Module.__init__(base)
        base.backbone = _FakeBackbone(embed_dim=8)
        base.teacher = _FakeTeacher()
        base.layer_indices = (1, 3)
        base.use_register_context = False
        base.register_context_proj = None
        base.register_context_scale = 0.0
        base.register_context_target = "fused"

        anchor_refresh = DinoProposalFrontend.__new__(DinoProposalFrontend)
        nn.Module.__init__(anchor_refresh)
        anchor_refresh.backbone = _FakeBackbone(embed_dim=8)
        anchor_refresh.teacher = _FakeTeacher()
        anchor_refresh.layer_indices = (1, 3)
        anchor_refresh.use_register_context = True
        anchor_refresh.register_context_scale = 0.20
        anchor_refresh.register_context_target = "anchor_refresh"
        anchor_refresh.register_context_proj = nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            anchor_refresh.register_context_proj.weight.copy_(torch.eye(8))

        fused_base, _ = DinoProposalFrontend._encode_backbone(base, images)
        fused_anchor, register_anchor = DinoProposalFrontend._encode_backbone(anchor_refresh, images)

        self.assertTrue(torch.allclose(fused_base, fused_anchor))
        self.assertEqual(tuple(register_anchor.shape), (1, 2, 8))

    def test_register_projection_accepts_inference_backbone_tensors(self) -> None:
        images = torch.randn(1, 2, 3, 32, 32, requires_grad=True)

        with_register = DinoProposalFrontend.__new__(DinoProposalFrontend)
        nn.Module.__init__(with_register)
        with_register.backbone = _FakeBackbone(embed_dim=8, inference_tokens=True)
        with_register.teacher = _FakeTeacher()
        with_register.layer_indices = (1, 3)
        with_register.use_register_context = True
        with_register.register_context_scale = 0.20
        with_register.register_context_target = "fused"
        with_register.register_context_proj = nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            with_register.register_context_proj.weight.copy_(torch.eye(8))

        fused, register_context = DinoProposalFrontend._encode_backbone(with_register, images)
        loss = fused.square().mean() + register_context.square().mean()
        loss.backward()

        self.assertIsNotNone(with_register.register_context_proj.weight.grad)
        self.assertTrue(torch.isfinite(with_register.register_context_proj.weight.grad).all().item())


if __name__ == "__main__":
    unittest.main()
