from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "refocus_vo" / "src"))

from refocus_vo.dino_dpvo.frontend import ConvGRUCell2d, DPTFusionDecoder, DinoProposalFrontend  # noqa: E402


class Pure100FrontendModesTests(unittest.TestCase):
    def test_dpt_decoder_preserves_dense_shape(self) -> None:
        frontend = DinoProposalFrontend.__new__(DinoProposalFrontend)
        nn.Module.__init__(frontend)
        frontend.proposal_decoder_mode = "dpt_fpn"
        frontend.dpt_decoder = DPTFusionDecoder(layer_indices=(1, 3, 6, 11), in_dim=8, out_dim=8)

        fused = torch.randn(2, 3, 8, 10, 12)
        hidden_states = {
            1: torch.randn(2, 3, 8, 10, 12),
            3: torch.randn(2, 3, 8, 10, 12),
            6: torch.randn(2, 3, 8, 10, 12),
            11: torch.randn(2, 3, 8, 10, 12),
        }

        decoded = DinoProposalFrontend._decode_semantic_map(frontend, fused=fused, hidden_states=hidden_states)

        self.assertEqual(tuple(decoded.shape), tuple(fused.shape))
        self.assertTrue(torch.isfinite(decoded).all().item())

    def test_convgru_temporal_memory_preserves_shape(self) -> None:
        frontend = DinoProposalFrontend.__new__(DinoProposalFrontend)
        nn.Module.__init__(frontend)
        frontend.temporal_memory_mode = "convgru"
        frontend.temporal_convgru = ConvGRUCell2d(8, 8)
        frontend.temporal_convgru_out = nn.Conv2d(8, 8, kernel_size=1)

        dense_map = torch.randn(2, 4, 8, 10, 12)
        encoded = DinoProposalFrontend._apply_temporal_memory(frontend, dense_map)

        self.assertEqual(tuple(encoded.shape), tuple(dense_map.shape))
        self.assertTrue(torch.isfinite(encoded).all().item())

    def test_token_gru_temporal_memory_preserves_shape(self) -> None:
        frontend = DinoProposalFrontend.__new__(DinoProposalFrontend)
        nn.Module.__init__(frontend)
        frontend.temporal_memory_mode = "token_gru"
        frontend.semantic_grid_rows = 4
        frontend.semantic_grid_cols = 5
        frontend.temporal_token_gru = nn.GRU(input_size=8, hidden_size=8, batch_first=True)
        frontend.temporal_token_out = nn.Linear(8, 8)

        dense_map = torch.randn(2, 3, 8, 12, 16)
        encoded = DinoProposalFrontend._apply_temporal_memory(frontend, dense_map)

        self.assertEqual(tuple(encoded.shape), tuple(dense_map.shape))
        self.assertTrue(torch.isfinite(encoded).all().item())

    def test_dual_stream_proposal_map_preserves_shape(self) -> None:
        frontend = DinoProposalFrontend.__new__(DinoProposalFrontend)
        nn.Module.__init__(frontend)
        frontend.proposal_decoder_mode = "dual_stream"
        frontend.dpt_decoder = None
        frontend.temporal_memory_mode = "none"
        frontend.local_to_embed = nn.Conv2d(6, 8, kernel_size=1)
        frontend.dual_stream_gate = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=1),
            nn.Sigmoid(),
        )

        fused = torch.randn(2, 3, 8, 12, 16)
        local_map = torch.randn(2, 3, 6, 12, 16)
        proposal_map = DinoProposalFrontend._proposal_feature_map(
            frontend,
            fused=fused,
            hidden_states=None,
            local_map=local_map,
        )

        self.assertEqual(tuple(proposal_map.shape), tuple(fused.shape))
        self.assertTrue(torch.isfinite(proposal_map).all().item())

    def test_gradient_corner_geometry_score_preserves_shape(self) -> None:
        frontend = DinoProposalFrontend.__new__(DinoProposalFrontend)
        nn.Module.__init__(frontend)
        frontend.geometry_score_mode = "gradient_corner"
        frontend.patch_size = 4
        frontend.corner_head = nn.Sequential(
            nn.Conv2d(6, 6, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 1, kernel_size=1),
        )

        images = torch.randn(2, 3, 3, 48, 64)
        local_map = torch.randn(2, 3, 6, 12, 16)
        geometry = DinoProposalFrontend._geometry_score(frontend, images, local_map)

        self.assertEqual(tuple(geometry.shape), (2, 3, 12, 16))
        self.assertTrue(torch.isfinite(geometry).all().item())


if __name__ == "__main__":
    unittest.main()
