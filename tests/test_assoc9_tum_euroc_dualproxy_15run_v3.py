from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "refocus_vo" / "src"))

from refocus_vo.sweeps.run_assoc9_sweep import _load_manifest  # noqa: E402


class TumEurocDualProxy15RunV3ManifestTests(unittest.TestCase):
    def test_manifest_matches_v3_shape(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "refocus_vo" / "configs" / "dino_dpvo_tum_euroc_live_dualproxy_15run_v3.yaml"
        payload, runs = _load_manifest(manifest_path)

        self.assertEqual(payload["name"], "dino_dpvo_tum_euroc_live_dualproxy_15run_v3")
        self.assertEqual(len(runs), 15)
        self.assertNotIn("kitti", payload["proxy_validation"]["datasets"])
        self.assertEqual(int(payload["proxy_validation"]["top_k"]), 5)
        self.assertEqual(int(payload["sweep"]["min_failure_step"]), 2500)
        self.assertEqual(payload["sweep"]["required_usable_dev_steps"], [1500, 2000, 2500])
        self.assertEqual(payload["proxy_validation"]["references"], {"tum": "incumbent_qualitysoft_r78_22", "euroc": "dpvo_native_matched"})
        self.assertEqual(payload["proxy_validation"]["ranking"]["weights"], {"tum": 0.4, "euroc": 0.3})
        self.assertEqual(payload["proxy_validation"]["ranking"]["win_weights"], {"tum": 0.15, "euroc": 0.15})
        self.assertEqual(int(payload["full_benchmark"]["stage1_top_k"]), 5)
        self.assertEqual(int(payload["full_benchmark"]["stage2_top_k"]), 2)
        self.assertEqual(payload["full_benchmark"]["references"], {"tum": "incumbent_qualitysoft_r78_22", "euroc": "dpvo_native_matched"})

    def test_run_order_matches_alternating_plan(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "refocus_vo" / "configs" / "dino_dpvo_tum_euroc_live_dualproxy_15run_v3.yaml"
        _, runs = _load_manifest(manifest_path)

        self.assertEqual(
            [run.run_id for run in runs],
            [
                "ms32x4_micro2_anchor16_qualitysoft_r78_22_v2",
                "ms24x5_cross3_anchor12_register_anchor_r80_20_v1",
                "ms32x4_micro2_anchor12_r76_24_longlife_qualitysoft_v2",
                "ms24x5_cross3_grad48_unfreeze1_gram03_r82_18_v3",
                "ms24x5_cross3_anchor12_r80_20_v2",
                "ms32x4_micro2_anchor16_register_fused_r78_22_v1",
                "ms32x4_micro2_anchor16_r80_20_v3",
                "ms24x5_cross3_anchor12_descimapgmap_r80_20_v2",
                "ms32x4_micro2_anchor16_qualitysoft_r80_20_v1",
                "ms24x5_cross3_register_desc_r80_20_v1",
                "ms32x4_micro2_anchor12_r74_26_longlife_v1",
                "ms32x4_micro2_qualitysoft_gram03_r78_22_v1",
                "ms24x5_cross3_anchor16_qualitysoft_r80_20_v1",
                "ms24x5_cross3_grad48_unfreeze1_register_anchor_r80_20_v1",
                "ms32x4_micro2_anchor16_qualitysoft_desc_r78_22_v1",
            ],
        )


if __name__ == "__main__":
    unittest.main()
