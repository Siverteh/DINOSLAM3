from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "refocus_vo" / "src"))

from refocus_vo.sweeps.run_assoc9_sweep import _load_manifest  # noqa: E402


class TumEurocDualProxy20RunV2ManifestTests(unittest.TestCase):
    def test_manifest_matches_overnight_20run_shape(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "refocus_vo" / "configs" / "dino_dpvo_tum_euroc_live_dualproxy_20run_v2.yaml"
        payload, runs = _load_manifest(manifest_path)

        self.assertEqual(payload["name"], "dino_dpvo_tum_euroc_live_dualproxy_20run_v2")
        self.assertEqual(len(runs), 20)
        self.assertNotIn("kitti", payload["proxy_validation"]["datasets"])
        self.assertEqual(int(payload["proxy_validation"]["top_k"]), 6)
        self.assertEqual(int(payload["sweep"]["min_failure_step"]), 2000)
        self.assertEqual(payload["sweep"]["required_usable_dev_steps"], [1000, 1500, 2000])
        self.assertEqual(payload["proxy_validation"]["ranking"]["weights"], {"tum": 0.55, "euroc": 0.25})
        self.assertEqual(payload["proxy_validation"]["ranking"]["win_weights"], {"tum": 0.15, "euroc": 0.05})
        self.assertEqual(int(payload["full_benchmark"]["stage1_top_k"]), 6)
        self.assertEqual(int(payload["full_benchmark"]["stage1_repeats"]), 1)
        self.assertEqual(int(payload["full_benchmark"]["stage2_top_k"]), 3)
        self.assertEqual(int(payload["full_benchmark"]["repeats"]), 5)

    def test_run_order_matches_alternating_plan(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "refocus_vo" / "configs" / "dino_dpvo_tum_euroc_live_dualproxy_20run_v2.yaml"
        _, runs = _load_manifest(manifest_path)

        self.assertEqual(
            [run.run_id for run in runs],
            [
                "ms32x4_r75_25_ctrl_v2",
                "ms32x4_micro2_register_r80_20_v1",
                "ms32x4_micro2_r80_20_seedmicro_v3",
                "ms24x5_cross3_register_r82_18_v1",
                "ms32x4_micro2_anchor12_r78_22_v2",
                "ms32x4_micro2_gram06_r78_22_v1",
                "ms32x4_micro2_anchor16_r80_20_v2",
                "ms24x5_cross3_gram08_anchor12_v1",
                "ms24x5_cross3_r82_18_v2",
                "ms32x4_micro2_descimapgmap_r78_22_v2",
                "ms32x4_micro3_r78_22_v1",
                "ms24x5_cross3_grad48_unfreeze1_r82_18_v2",
                "ms24x5_cross3_anchor12_r80_20_v1",
                "ms32x4_micro2_register_desc_r78_22_v1",
                "ms32x4_micro2_anchor12_r76_24_longlife_v1",
                "ms24x5_cross3_register_grad_unfreeze1_r82_18_v1",
                "ms32x4_micro2_r78_22_widepool_v1",
                "ms32x4_micro2_4layer_unfreeze1_anchor12_v1",
                "ms32x4_micro2_anchor16_qualitysoft_r78_22_v1",
                "ms24x5_cross3_register_gram_desc_v1",
            ],
        )


if __name__ == "__main__":
    unittest.main()
