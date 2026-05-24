from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dino_slam3.utils.config import load_config


class SweepArchLoss10bConfigTests(unittest.TestCase):
    def test_candidate_configs_are_deterministic_and_unique(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cfg_dir = root / "configs" / "train" / "sweeps" / "desk_archloss10b"
        expected = [f"c{i:02d}" for i in range(10, 20)]

        seen = set()
        kinds = {"architecture": 0, "training": 0}
        for cid in expected:
            cfg = load_config(str(cfg_dir / f"{cid}.yaml"))
            sweep = cfg.get("sweep", {}) if isinstance(cfg.get("sweep", {}), dict) else {}
            got = str(sweep.get("candidate_id", "")).strip()
            self.assertEqual(got, cid)
            self.assertNotIn(got, seen)
            seen.add(got)

            kind = str(sweep.get("kind", "")).strip()
            self.assertIn(kind, kinds)
            kinds[kind] += 1

        self.assertEqual(kinds["architecture"], 6)
        self.assertEqual(kinds["training"], 4)


if __name__ == "__main__":
    unittest.main()
