from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class SweepTriArchLoss50ConfigTests(unittest.TestCase):
    def test_generator_outputs_deterministic_grid(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cfg_dir = root / "configs" / "train" / "sweeps" / "tri_archloss50"
        gen = root / "scripts" / "make_tri_archloss50_configs.py"

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "tri_cfgs"
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
            self.assertEqual(len(files), 50)
            self.assertTrue((out / "c100.yaml").exists())
            self.assertTrue((out / "c149.yaml").exists())

            for aid in range(10):
                for lid in range(5):
                    cid = 100 + aid * 5 + lid
                    cfg = yaml.safe_load((out / f"c{cid}.yaml").read_text(encoding="utf-8"))
                    sweep = cfg.get("sweep", {}) if isinstance(cfg.get("sweep", {}), dict) else {}
                    self.assertEqual(str(sweep.get("candidate_id", "")), f"c{cid}")
                    self.assertEqual(str(sweep.get("arch_id", "")), f"A{aid}")
                    self.assertEqual(str(sweep.get("loss_id", "")), f"L{lid}")


if __name__ == "__main__":
    unittest.main()
