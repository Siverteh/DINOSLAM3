"""
Headless-safe evaluation helpers.

Key point:
- DO NOT import evo.tools.plot in headless environments, because it may force a Qt backend (qtagg).
- Computing ATE/RPE does not require plotting.
"""

import os

# Headless-friendly matplotlib backend (must be set before importing matplotlib)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)

from evo.core import sync
from evo.core.metrics import PoseRelation, Unit
from evo.core.metrics import APE, RPE
from evo.tools import file_interface


def eval_ate(gt_file: str, est_file: str):
    """
    Evaluate ATE (Absolute Pose Error) using evo, without plotting.
    Returns evo statistics dict.
    """
    traj_ref = file_interface.read_tum_trajectory_file(gt_file)
    traj_est = file_interface.read_tum_trajectory_file(est_file)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    ape_metric = APE(PoseRelation.translation_part)
    ape_metric.process_data((traj_ref, traj_est))
    return ape_metric.get_all_statistics()


def eval_rpe(gt_file: str, est_file: str, delta: int = 1):
    """
    Evaluate RPE (Relative Pose Error) using evo, without plotting.
    Returns evo statistics dict.
    """
    traj_ref = file_interface.read_tum_trajectory_file(gt_file)
    traj_est = file_interface.read_tum_trajectory_file(est_file)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    rpe_metric = RPE(PoseRelation.translation_part, delta=delta, unit=Unit.frames)
    rpe_metric.process_data((traj_ref, traj_est))
    return rpe_metric.get_all_statistics()
