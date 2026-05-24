from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dino_slam3.tracking import write_run_manifest, write_semantic_selection_snapshot


class ArtifactWriterTests(unittest.TestCase):
    def test_manifest_immutable_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "run"
            cfg = {
                "run": {
                    "name": "test_run",
                    "tracking": {
                        "artifact_dir": "artifacts",
                        "save_manifest_json": True,
                    },
                }
            }
            p1 = write_run_manifest(
                out_dir=out_dir,
                cfg=cfg,
                experiment_id="exp-123",
                config_hash="hash-1",
                git_commit="abc",
                env={"A": "1"},
            )
            body1 = json.loads(p1.read_text(encoding="utf-8"))
            self.assertEqual(body1["experiment_id"], "exp-123")
            self.assertEqual(body1["config_hash"], "hash-1")

            # second call should not overwrite by default
            p2 = write_run_manifest(
                out_dir=out_dir,
                cfg=cfg,
                experiment_id="exp-123",
                config_hash="hash-2",
                git_commit="def",
                env={"A": "2"},
            )
            self.assertEqual(p1, p2)
            body2 = json.loads(p2.read_text(encoding="utf-8"))
            self.assertEqual(body2["config_hash"], "hash-1")

    def test_selection_snapshot_written(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "run"
            path = write_semantic_selection_snapshot(
                out_dir=out_dir,
                epoch=7,
                payload={"candidate": 1.2, "coverage_gate_ok": True},
            )
            self.assertTrue(path.exists())
            body = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(int(body["epoch"]), 7)
            self.assertAlmostEqual(float(body["candidate"]), 1.2, places=7)


if __name__ == "__main__":
    unittest.main()
