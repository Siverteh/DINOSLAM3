from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "refocus_vo" / "src"))

from refocus_vo.dino_dpvo.adapter import build_dpvo_patch_input  # noqa: E402
from refocus_vo.dino_dpvo.frontend import DinoDPVOFrameOutput  # noqa: E402
from refocus_vo.eval.focus071_tumwin_finalists import FinalistCandidate, _expand_mode_candidates  # noqa: E402
from refocus_vo.patchgraph.teacher import PseudoObjectPatchProposal  # noqa: E402
from refocus_vo.sweeps.run_assoc9_sweep import _load_manifest  # noqa: E402
from refocus_vo.train_dino_dpvo_frontend import _winner_mode_from_metrics  # noqa: E402
from refocus_vo.dino_dpvo.config import DinoDPVOConfig  # noqa: E402


class Focus071Arch5x2SweepTests(unittest.TestCase):
    def test_manifest_has_10_runs_and_best_of_mode(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "refocus_vo" / "configs" / "dino_dpvo_focus071_arch5x2_tumwin_sweep_v1.yaml"
        payload, runs = _load_manifest(manifest_path)
        self.assertEqual(payload["name"], "dino_dpvo_focus071_arch5x2_tumwin_sweep_v1")
        self.assertEqual(len(runs), 10)
        eval_cfg = payload["sweep"]["config_overrides"]["eval"]
        self.assertEqual(eval_cfg["selection_mode"], "best_of_pure_hybrid")
        self.assertTrue(bool(eval_cfg["save_best_hybrid"]))
        self.assertTrue(bool(eval_cfg["run_hybrid_dev_eval"]))

    def test_winner_mode_prefers_hybrid_when_proxy_is_better(self) -> None:
        cfg = DinoDPVOConfig(
            method_id="demo",
            feature_type="DEMO",
            raw={
                "eval": {
                    "selection_mode": "best_of_pure_hybrid",
                    "coverage_gate": 0.95,
                    "secondary_eval_sequences": ["freiburg2_coke"],
                    "secondary_coverage_gate": 0.95,
                }
            },
        )
        dev_metrics = {
            "pure100_mean_ate": 0.40,
            "pure100_mean_ate_associated": 0.40,
            "pure100_mean_coverage": 0.99,
            "lowtex_mean_ate": 0.55,
            "lowtex_mean_ate_associated": 0.55,
            "lowtex_mean_coverage": 0.99,
            "hybrid_mean_ate": 0.35,
            "hybrid_mean_ate_associated": 0.34,
            "hybrid_mean_coverage": 0.99,
            "hybrid_lowtex_mean_ate": 0.53,
            "hybrid_lowtex_mean_ate_associated": 0.53,
            "hybrid_lowtex_mean_coverage": 0.99,
        }
        self.assertEqual(_winner_mode_from_metrics(dev_metrics, cfg), "hybrid")

    def test_adapter_emits_micro_patches_and_dual_bias(self) -> None:
        proposal = PseudoObjectPatchProposal(
            patch_indices=torch.tensor([0, 1], dtype=torch.long),
            patch_xy=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
            coarse_pixel_xy=torch.tensor([[80.0, 60.0], [160.0, 120.0]], dtype=torch.float32),
            pixel_xy=torch.tensor([[82.0, 62.0], [158.0, 118.0]], dtype=torch.float32),
            offset_xy=torch.zeros((2, 2), dtype=torch.float32),
            scores=torch.tensor([0.9, 0.8], dtype=torch.float32),
            object_ids=torch.tensor([0, 1], dtype=torch.long),
            descriptors=torch.randn(2, 384),
            local_features=torch.randn(2, 64),
        )
        frame_output = DinoDPVOFrameOutput(
            proposal=proposal,
            selector_logits=torch.zeros((15, 20), dtype=torch.float32),
            staticness_logits=torch.zeros((15, 20), dtype=torch.float32),
            gradient_score=torch.zeros((15, 20), dtype=torch.float32),
            qualities=torch.tensor([0.9, 0.8], dtype=torch.float32),
            descriptor_bias=torch.randn(2, 384),
            gmap_descriptor_bias=torch.randn(2, 128),
        )
        patch_input = build_dpvo_patch_input(
            frame_output,
            patch_budget=6,
            frontend_mode="dino_proposals",
            dpvo_res=4,
            image_height=240,
            image_width=320,
            config={
                "enforce_unique_semantic": True,
                "semantic_grid_rows": 6,
                "semantic_grid_cols": 8,
                "max_semantic_per_cell": 2,
                "semantic_dedupe_schedule_px": [8.0, 4.0],
                "micro_patch_count": 4,
                "micro_patch_pattern": "grid",
                "micro_patch_spread_px": 4.0,
                "micro_patch_center_mode": "refined",
                "quality_mode": "soft",
                "quality_edge_power": 1.2,
            },
        )
        self.assertEqual(tuple(patch_input["external_coords"].shape), (1, 6, 2))
        self.assertEqual(tuple(patch_input["external_quality"].shape), (1, 6, 1))
        self.assertEqual(tuple(patch_input["external_descriptor_bias"].shape), (1, 6, 384))
        self.assertEqual(tuple(patch_input["external_gmap_bias"].shape), (1, 6, 128))
        self.assertEqual(int(patch_input["patch_metadata"]["source_labels"].numel()), 6)

    def test_expand_mode_candidates_returns_run_mode_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "cfg.yaml"
            cfg.write_text("method_id: demo\nfeature_type: DEMO\n", encoding="utf-8")
            pure = root / "best_pure100.pt"
            hybrid = root / "best_hybrid.pt"
            pure.write_text("pure", encoding="utf-8")
            hybrid.write_text("hybrid", encoding="utf-8")
            candidate = FinalistCandidate(
                run_id="demo",
                mode="pure100",
                checkpoint_path=pure,
                config_path=cfg,
                best_assoc=0.1,
                best_secondary_assoc=0.2,
                pure_checkpoint_path=pure,
                hybrid_checkpoint_path=hybrid,
            )
            expanded = _expand_mode_candidates([candidate])
        self.assertEqual([(item.run_id, item.mode) for item in expanded], [("demo", "pure100"), ("demo", "hybrid")])


if __name__ == "__main__":
    unittest.main()
