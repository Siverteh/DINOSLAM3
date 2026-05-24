from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dino_slam3.training.trainer import _scheduled_loss_cfg


class LossScheduleTests(unittest.TestCase):
    def test_schedule_overrides_active_epoch(self) -> None:
        cfg = {
            "loss": {
                "geom": {"pose_weight": 0.1},
                "detector": {"peakiness_weight": 0.2},
            },
            "training": {
                "loss_weight_schedule": [
                    {
                        "epoch_start": 1,
                        "epoch_end": 2,
                        "loss": {"geom": {"pose_weight": 0.3}},
                    },
                    {
                        "epoch_start": 3,
                        "epoch_end": 4,
                        "loss": {"detector": {"peakiness_weight": 0.5}},
                    },
                ]
            },
        }
        c1 = _scheduled_loss_cfg(cfg, epoch=1, total_epochs=4)
        self.assertAlmostEqual(float(c1["geom"]["pose_weight"]), 0.3, places=7)
        self.assertAlmostEqual(float(c1["detector"]["peakiness_weight"]), 0.2, places=7)

        c3 = _scheduled_loss_cfg(cfg, epoch=3, total_epochs=4)
        self.assertAlmostEqual(float(c3["geom"]["pose_weight"]), 0.1, places=7)
        self.assertAlmostEqual(float(c3["detector"]["peakiness_weight"]), 0.5, places=7)

    def test_schedule_does_not_mutate_base(self) -> None:
        cfg = {
            "loss": {"geom": {"pose_weight": 0.1}},
            "training": {
                "loss_weight_schedule": [
                    {
                        "epoch_start": 1,
                        "epoch_end": 10,
                        "loss": {"geom": {"pose_weight": 0.9}},
                    }
                ]
            },
        }
        _ = _scheduled_loss_cfg(cfg, epoch=1, total_epochs=10)
        self.assertAlmostEqual(float(cfg["loss"]["geom"]["pose_weight"]), 0.1, places=7)


if __name__ == "__main__":
    unittest.main()

