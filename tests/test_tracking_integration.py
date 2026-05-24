from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dino_slam3.tracking import ExperimentStore, write_run_manifest


class TrackingIntegrationTests(unittest.TestCase):
    def test_manifest_and_db_paths_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            db_path = root / "runs" / "experiments.db"
            out_dir = root / "runs" / "stageC_run"
            rel_ckpt = Path("checkpoints") / "geom_best.pt"

            old_cwd = Path.cwd()
            os.chdir(root)
            try:
                cfg = {
                    "run": {
                        "name": "stageC_run",
                        "prompt_tag": "integration",
                        "tracking": {
                            "sqlite_path": str(db_path),
                            "artifact_dir": "artifacts",
                            "save_manifest_json": True,
                        },
                    }
                }
                manifest = write_run_manifest(
                    out_dir=out_dir,
                    cfg=cfg,
                    experiment_id="exp-it-1",
                    config_hash="cfghash",
                    git_commit="abcdef0",
                    env={"PYTHON_BIN": "/tmp/python"},
                )
                self.assertTrue(manifest.exists())
                manifest_body = json.loads(manifest.read_text(encoding="utf-8"))
                self.assertEqual(manifest_body["experiment_id"], "exp-it-1")
                self.assertIn("resolved_paths", manifest_body)

                store = ExperimentStore(db_path)
                store.upsert_experiment(
                    experiment_id="exp-it-1",
                    run_name="stageC_run",
                    prompt_tag="integration",
                    parent_id=None,
                    git_commit="abcdef0",
                    config_hash="cfghash",
                )
                store.log_checkpoint(
                    experiment_id="exp-it-1",
                    stage="stageC_run",
                    epoch=12,
                    path=rel_ckpt,
                    selected_flag=True,
                    selection_score=1.11,
                )
                store.log_artifact(
                    experiment_id="exp-it-1",
                    artifact_type="run_manifest",
                    path=manifest,
                )

                conn = sqlite3.connect(str(db_path))
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM experiments WHERE experiment_id='exp-it-1'")
                self.assertEqual(cur.fetchone()[0], 1)

                cur.execute(
                    "SELECT path, selected_flag FROM checkpoints WHERE experiment_id='exp-it-1' ORDER BY id DESC LIMIT 1"
                )
                cp_path, selected_flag = cur.fetchone()
                self.assertEqual(selected_flag, 1)
                self.assertTrue(Path(cp_path).is_absolute())

                cur.execute(
                    "SELECT path FROM artifacts WHERE experiment_id='exp-it-1' AND type='run_manifest' ORDER BY id DESC LIMIT 1"
                )
                art_path = cur.fetchone()[0]
                self.assertEqual(Path(art_path).resolve(), manifest.resolve())
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
