from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "refocus_vo" / "src"))

from refocus_vo.sweeps.run_assoc9_sweep import _load_manifest, _usable_step_row  # noqa: E402


class Pure100RecoverySweepManifestTests(unittest.TestCase):
    def test_manifest_matches_requested_shape(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "refocus_vo" / "configs" / "dino_dpvo_pure100_tum30_recovery_v1.yaml"
        payload, runs = _load_manifest(manifest_path)

        self.assertEqual(payload["name"], "dino_dpvo_pure100_tum30_recovery_v1")
        self.assertEqual(len(runs), 30)
        self.assertEqual(int(payload["proxy_validation"]["top_k"]), 8)
        self.assertEqual(payload["sweep"]["required_usable_dev_steps"], [500, 1000])
        self.assertEqual(int(payload["sweep"]["min_failure_step"]), 1000)
        self.assertEqual(payload["proxy_validation"]["references"], {"tum": "target_multiscale_32x4_v1_hybrid75_25"})
        self.assertEqual(set(payload["proxy_validation"]["datasets"].keys()), {"tum"})
        self.assertEqual(
            payload["proxy_validation"]["ranking"]["weights"],
            {"primary_ate": 0.55, "pressure_ate": 0.20, "primary_wins": 0.15, "pressure_wins": 0.10},
        )
        self.assertEqual(int(payload["full_benchmark"]["stage1_top_k"]), 8)
        self.assertEqual(int(payload["full_benchmark"]["stage2_top_k"]), 3)

    def test_base_overrides_force_pure_dino_eval(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "refocus_vo" / "configs" / "dino_dpvo_pure100_tum30_recovery_v1.yaml"
        payload, runs = _load_manifest(manifest_path)

        eval_cfg = payload["sweep"]["config_overrides"]["eval"]
        model_cfg = payload["sweep"]["config_overrides"]["model"]

        self.assertEqual(eval_cfg["frontend_mode"], "dino_proposals")
        self.assertEqual(eval_cfg["pure100_frontend_mode"], "dino_proposals")
        self.assertEqual(eval_cfg["selection_metric"], "pure_tum_proxy_score")
        self.assertEqual(eval_cfg["selection_mode"], "pure_only")
        self.assertTrue(eval_cfg["save_best_pure100"])
        self.assertFalse(eval_cfg["save_best_hybrid"])
        self.assertFalse(eval_cfg["run_hybrid_dev_eval"])
        self.assertEqual(float(model_cfg["native_fraction"]), 0.0)
        self.assertEqual(float(model_cfg["dino_fraction"]), 1.0)

        self.assertTrue(all(getattr(run, "expected_eval_mode", "") == "pure100" for run in runs))

    def test_run_order_matches_requested_plan(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "refocus_vo" / "configs" / "dino_dpvo_pure100_tum30_recovery_v1.yaml"
        _, runs = _load_manifest(manifest_path)

        self.assertEqual(
            [run.run_id for run in runs],
            [
                "pure100_focus071_ctrl_v2",
                "pure100_focus071_dpt4_v1",
                "pure100_rescue_ctrl_v2",
                "pure100_rescue_convgru64_v1",
                "multiscale_pure100_transfer_ctrl_v1",
                "multiscale_pure100_dualstream_v1",
                "pure100_rescue_stratified_random_v1",
                "pure100_rescue_token_gru96_v1",
                "pure100_focus071_geomix_v1",
                "pure100_focus071_register_fused_v1",
                "pure100_rescue_widepool_v1",
                "pure100_rescue_dpt_convgru_v1",
                "multiscale_pure100_semgrid88_v1",
                "multiscale_pure100_register_dualstream_v1",
                "pure100_rescue_lowdedupe_v1",
                "pure100_rescue_gram04_v1",
                "pure100_focus071_corner_quality_v1",
                "pure100_focus071_dpt_tokengru_v1",
                "pure100_rescue_keepalive12_v1",
                "pure100_rescue_register_score_v1",
                "multiscale_pure100_grad48_v1",
                "multiscale_pure100_unfreeze1_dpt_v1",
                "pure100_rescue_descimap_v1",
                "pure100_rescue_descimap_register_v1",
                "pure100_focus071_randombackfill35_v1",
                "pure100_focus071_unfreeze1_convgru_v1",
                "pure100_rescue_geomix_keepalive_v1",
                "pure100_rescue_dualstream_register_v1",
                "multiscale_pure100_bestshot_v1",
                "pure100_rescue_bestshot_v1",
            ],
        )


class Pure100RecoverySweepGateTests(unittest.TestCase):
    def test_usable_step_requires_full_dino_patch_fraction(self) -> None:
        rejected = _usable_step_row(
            [
                {
                    "step": "500",
                    "selection_metric": "pure_tum_proxy_score",
                    "selection_passed_gate": "1",
                    "tum_proxy_mean_ate_associated": "0.25",
                    "tum_pressure_mean_ate_associated": "0.30",
                    "live_pure_tum_proxy_score": "0.8",
                    "dino_patch_fraction": "0.90",
                }
            ],
            500,
        )
        accepted = _usable_step_row(
            [
                {
                    "step": "500",
                    "selection_metric": "pure_tum_proxy_score",
                    "selection_passed_gate": "1",
                    "tum_proxy_mean_ate_associated": "0.25",
                    "tum_pressure_mean_ate_associated": "0.30",
                    "live_pure_tum_proxy_score": "0.8",
                    "dino_patch_fraction": "1.0",
                }
            ],
            500,
        )

        self.assertIsNone(rejected)
        self.assertIsNotNone(accepted)


if __name__ == "__main__":
    unittest.main()
