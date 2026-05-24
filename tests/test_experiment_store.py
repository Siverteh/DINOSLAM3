from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dino_slam3.tracking import ExperimentStore


class ExperimentStoreTests(unittest.TestCase):
    def test_upsert_and_logging(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "experiments.db"
            store = ExperimentStore(db)
            store.upsert_experiment(
                experiment_id="exp-1",
                run_name="stageC",
                prompt_tag="idea1",
                parent_id=None,
                git_commit="abc",
                config_hash="cfg",
            )
            # idempotent upsert
            store.upsert_experiment(
                experiment_id="exp-1",
                run_name="stageC2",
                prompt_tag="idea2",
                parent_id="exp-0",
                git_commit="def",
                config_hash="cfg2",
            )
            # ensure_experiment should not overwrite existing metadata
            store.ensure_experiment(
                experiment_id="exp-1",
                run_name="external_eval",
                prompt_tag="external_eval",
                parent_id=None,
                git_commit="unknown",
                config_hash="unknown",
            )

            store.log_checkpoint(
                experiment_id="exp-1",
                stage="stageC2",
                epoch=3,
                path=Path(td) / "ckpt.pt",
                selected_flag=True,
                selection_score=1.23,
            )
            store.log_sequence_metrics(
                experiment_id="exp-1",
                stage="eval",
                rows=[
                    {
                        "sequence": "freiburg1_desk",
                        "status": "ok",
                        "ate_rmse": 0.9,
                        "ate_rmse_associated": 0.04,
                        "rpe_trans_rmse": 0.1,
                        "rpe_rot_rmse": 1.0,
                        "coverage": 0.98,
                    }
                ],
            )
            store.log_aggregate(
                experiment_id="exp-1",
                metric_name="weighted_penalized_ate",
                metric_value=1.23,
            )
            store.log_artifact(
                experiment_id="exp-1",
                artifact_type="manifest",
                path=Path(td) / "run_manifest.json",
            )

            conn = sqlite3.connect(str(db))
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM experiments WHERE experiment_id='exp-1'")
            self.assertEqual(cur.fetchone()[0], 1)
            cur.execute("SELECT run_name, prompt_tag, parent_id FROM experiments WHERE experiment_id='exp-1'")
            run_name, prompt_tag, parent_id = cur.fetchone()
            self.assertEqual(run_name, "stageC2")
            self.assertEqual(prompt_tag, "idea2")
            self.assertEqual(parent_id, "exp-0")

            cur.execute("SELECT COUNT(*) FROM checkpoints WHERE experiment_id='exp-1'")
            self.assertEqual(cur.fetchone()[0], 1)
            cur.execute("SELECT COUNT(*) FROM sequence_metrics WHERE experiment_id='exp-1'")
            self.assertEqual(cur.fetchone()[0], 1)
            cur.execute("SELECT ate_rmse_associated FROM sequence_metrics WHERE experiment_id='exp-1' LIMIT 1")
            self.assertAlmostEqual(float(cur.fetchone()[0]), 0.04, places=7)
            cur.execute("SELECT COUNT(*) FROM aggregates WHERE experiment_id='exp-1'")
            self.assertEqual(cur.fetchone()[0], 1)
            cur.execute("SELECT COUNT(*) FROM artifacts WHERE experiment_id='exp-1'")
            self.assertEqual(cur.fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
