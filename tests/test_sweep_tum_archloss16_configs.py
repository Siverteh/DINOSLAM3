from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class SweepTumArchLoss16ConfigTests(unittest.TestCase):
    def test_generator_outputs_4x4_grid(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cfg_dir = root / "configs" / "train" / "sweeps" / "tum_archloss16"
        gen = root / "scripts" / "make_tum_archloss16_configs.py"

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "tum_cfgs"
            cmd = [
                str(root / ".venv" / "bin" / "python"),
                str(gen),
                "--config-dir",
                str(cfg_dir),
                "--out-dir",
                str(out),
            ]
            subprocess.check_call(cmd)

            files = sorted(out.glob("c*.yaml"))
            self.assertEqual(len(files), 16)
            self.assertTrue((out / "c00.yaml").exists())
            self.assertTrue((out / "c15.yaml").exists())

            for aid in range(4):
                for lid in range(4):
                    cid = aid * 4 + lid
                    cfg = yaml.safe_load((out / f"c{cid:02d}.yaml").read_text(encoding="utf-8"))
                    sweep = cfg.get("sweep", {}) if isinstance(cfg.get("sweep", {}), dict) else {}
                    self.assertEqual(str(sweep.get("candidate_id", "")), f"c{cid:02d}")
                    self.assertEqual(str(sweep.get("arch_id", "")), f"A{aid}")
                    self.assertEqual(str(sweep.get("loss_id", "")), f"L{lid}")

    def test_base_config_has_explicit_pair_distances(self) -> None:
        root = Path(__file__).resolve().parents[1]
        base_cfg = root / "configs" / "train" / "sweeps" / "tum_archloss16" / "base.yaml"
        cfg = yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
        pair_sampler = cfg.get("dataset", {}).get("pair_sampler", {})
        self.assertEqual(pair_sampler.get("pair_distances"), [1, 2, 5, 10, 20])


if __name__ == "__main__":
    unittest.main()
