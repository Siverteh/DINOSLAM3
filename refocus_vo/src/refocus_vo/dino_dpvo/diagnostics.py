from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SOURCE_LABELS = {
    0: "native",
    1: "dino",
}


@dataclass
class GroundTruthFrameContext:
    poses: list[np.ndarray | None]
    depths: list[np.ndarray | None]
    intrinsics: np.ndarray
    image_size: tuple[int, int]
    depth_boundary_thresh: float = 0.15
    unstable_error_px: float = 5.0


def _sample_depth(depth: np.ndarray | None, pixel_xy: np.ndarray) -> float | None:
    if depth is None:
        return None
    h, w = depth.shape[:2]
    x = int(np.clip(round(float(pixel_xy[0])), 0, w - 1))
    y = int(np.clip(round(float(pixel_xy[1])), 0, h - 1))
    value = float(depth[y, x])
    if not math.isfinite(value) or value <= 1e-6:
        return None
    return value


def _depth_boundary_flag(depth: np.ndarray | None, pixel_xy: np.ndarray, thresh: float) -> bool:
    if depth is None:
        return False
    h, w = depth.shape[:2]
    x = int(np.clip(round(float(pixel_xy[0])), 1, max(w - 2, 1)))
    y = int(np.clip(round(float(pixel_xy[1])), 1, max(h - 2, 1)))
    gx = float(depth[y, min(x + 1, w - 1)] - depth[y, max(x - 1, 0)])
    gy = float(depth[min(y + 1, h - 1), x] - depth[max(y - 1, 0), x])
    grad = math.sqrt(gx * gx + gy * gy)
    return math.isfinite(grad) and grad > float(thresh)


def _project_point(
    pose_src: np.ndarray | None,
    pose_tgt: np.ndarray | None,
    depth_src: np.ndarray | None,
    pixel_xy: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray | None:
    if pose_src is None or pose_tgt is None or depth_src is None:
        return None
    z = _sample_depth(depth_src, pixel_xy)
    if z is None:
        return None
    fx, fy, cx, cy = [float(v) for v in np.asarray(intrinsics, dtype=np.float64).reshape(-1)[:4]]
    x = ((float(pixel_xy[0]) - cx) / max(fx, 1e-6)) * z
    y = ((float(pixel_xy[1]) - cy) / max(fy, 1e-6)) * z
    point_cam = np.asarray([x, y, z, 1.0], dtype=np.float64)
    point_world = np.asarray(pose_src, dtype=np.float64) @ point_cam
    point_tgt = np.linalg.inv(np.asarray(pose_tgt, dtype=np.float64)) @ point_world
    if point_tgt[2] <= 1e-6:
        return None
    u = fx * point_tgt[0] / point_tgt[2] + cx
    v = fy * point_tgt[1] / point_tgt[2] + cy
    return np.asarray([u, v], dtype=np.float64)


class PatchDiagnosticsRecorder:
    def __init__(self) -> None:
        self.records: dict[int, dict] = {}

    def observe_step(self, slam) -> None:
        if not getattr(slam, "collect_patch_diagnostics", False):
            return
        state = slam.get_patch_diagnostics_state()
        n = int(state["n"])
        M = int(state["M"])
        res = float(state["RES"])
        patch_uid = np.asarray(state["patch_uid"], dtype=np.int64)
        patch_source = np.asarray(state["patch_source"], dtype=np.int64)
        patch_cell = np.asarray(state["patch_cell"], dtype=np.int64)
        patch_pixel = np.asarray(state["patch_pixel"], dtype=np.float32)
        patch_utility = np.asarray(state["patch_utility"], dtype=np.float32)
        patch_repeated = np.asarray(state.get("patch_repeated", np.zeros_like(patch_utility)), dtype=np.float32)
        patch_unique_semantic_count = np.asarray(
            state.get("patch_unique_semantic_count", np.zeros_like(patch_utility)),
            dtype=np.float32,
        )
        patch_dedupe_radius = np.asarray(
            state.get("patch_dedupe_radius", np.zeros_like(patch_utility)),
            dtype=np.float32,
        )
        patch_insert = np.asarray(state["patch_insert_t"], dtype=np.int64)
        tstamps = np.asarray(state["tstamps"], dtype=np.int64)

        for row in range(n):
            for col in range(M):
                uid = int(patch_uid[row, col])
                if uid < 0:
                    continue
                record = self.records.setdefault(
                    uid,
                    {
                        "patch_uid": uid,
                        "source_label": int(patch_source[row, col]),
                        "initial_cell_xy": patch_cell[row, col].tolist(),
                        "initial_pixel_xy": patch_pixel[row, col].tolist(),
                        "selection_utility": float(patch_utility[row, col]),
                        "repeated_patch_flag": float(patch_repeated[row, col]),
                        "unique_semantic_count_before_repeat": float(patch_unique_semantic_count[row, col]),
                        "dedupe_radius_used": float(patch_dedupe_radius[row, col]),
                        "insert_counter": int(patch_insert[row, col]),
                        "tracked_targets": {},
                        "max_target_counter": int(patch_insert[row, col]),
                    },
                )
                record["source_label"] = int(patch_source[row, col])
                record["initial_cell_xy"] = patch_cell[row, col].tolist()
                record["initial_pixel_xy"] = patch_pixel[row, col].tolist()
                record["selection_utility"] = float(patch_utility[row, col])
                record["repeated_patch_flag"] = float(patch_repeated[row, col])
                record["unique_semantic_count_before_repeat"] = float(patch_unique_semantic_count[row, col])
                record["dedupe_radius_used"] = float(patch_dedupe_radius[row, col])
                record["insert_counter"] = int(patch_insert[row, col])

        self._observe_edges(
            M=M,
            res=res,
            patch_uid=patch_uid,
            tstamps=tstamps,
            kk=np.asarray(state["kk"], dtype=np.int64),
            jj=np.asarray(state["jj"], dtype=np.int64),
            target=np.asarray(state["target"], dtype=np.float32),
        )
        self._observe_edges(
            M=M,
            res=res,
            patch_uid=patch_uid,
            tstamps=tstamps,
            kk=np.asarray(state["kk_inac"], dtype=np.int64),
            jj=np.asarray(state["jj_inac"], dtype=np.int64),
            target=np.asarray(state["target_inac"], dtype=np.float32),
        )

    def _observe_edges(
        self,
        *,
        M: int,
        res: float,
        patch_uid: np.ndarray,
        tstamps: np.ndarray,
        kk: np.ndarray,
        jj: np.ndarray,
        target: np.ndarray,
    ) -> None:
        if target.ndim == 3:
            target = target[0]
        if kk.size == 0 or target.size == 0:
            return
        for edge_idx in range(int(kk.shape[0])):
            patch_slot = int(kk[edge_idx])
            row = patch_slot // int(M)
            col = patch_slot % int(M)
            if row < 0 or row >= patch_uid.shape[0]:
                continue
            uid = int(patch_uid[row, col])
            if uid < 0:
                continue
            target_row = int(jj[edge_idx])
            if target_row < 0 or target_row >= tstamps.shape[0]:
                continue
            target_counter = int(tstamps[target_row])
            tracked_xy = ((np.asarray(target[edge_idx], dtype=np.float64) + 0.5) * float(res)).tolist()
            record = self.records.setdefault(uid, {"tracked_targets": {}, "max_target_counter": target_counter})
            record.setdefault("tracked_targets", {})[str(target_counter)] = tracked_xy
            record["max_target_counter"] = max(int(record.get("max_target_counter", target_counter)), target_counter)

    def summarize(
        self,
        *,
        sequence: str,
        feature_type: str,
        status: str,
        metrics: dict[str, float] | None,
        gt_context: GroundTruthFrameContext,
    ) -> tuple[dict[str, float | str], list[dict[str, float | int | str | bool | None]]]:
        rows: list[dict[str, float | int | str | bool | None]] = []
        max_x = 7
        max_y = 5

        for uid, record in sorted(self.records.items()):
            cell = np.asarray(record.get("initial_cell_xy", [-1, -1]), dtype=np.int64)
            if cell.shape[0] >= 2:
                max_x = max(max_x, int(cell[0]))
                max_y = max(max_y, int(cell[1]))

        cell_counts = np.zeros((max_y + 1, max_x + 1), dtype=np.int64)
        semantic_cell_counts = np.zeros((max_y + 1, max_x + 1), dtype=np.int64)
        native_count = 0
        dino_count = 0
        ages: list[float] = []
        repeated_flags: list[float] = []
        unique_counts: list[float] = []
        dedupe_radii: list[float] = []
        survived_1 = 0
        survived_3 = 0
        survived_5 = 0

        for uid, record in sorted(self.records.items()):
            source_label = int(record.get("source_label", -1))
            source_name = SOURCE_LABELS.get(source_label, "unknown")
            if source_label == 0:
                native_count += 1
            elif source_label == 1:
                dino_count += 1
                repeated_flags.append(float(record.get("repeated_patch_flag", 0.0)))
                unique_count = float(record.get("unique_semantic_count_before_repeat", math.nan))
                dedupe_radius = float(record.get("dedupe_radius_used", math.nan))
                if math.isfinite(unique_count):
                    unique_counts.append(unique_count)
                if math.isfinite(dedupe_radius):
                    dedupe_radii.append(dedupe_radius)
            initial_pixel = np.asarray(record.get("initial_pixel_xy", [math.nan, math.nan]), dtype=np.float64)
            insert_counter = int(record.get("insert_counter", -1))
            max_target_counter = int(record.get("max_target_counter", insert_counter))
            age = max(0, max_target_counter - insert_counter)
            ages.append(float(age))
            survived_1 += int(age >= 1)
            survived_3 += int(age >= 3)
            survived_5 += int(age >= 5)

            cell = np.asarray(record.get("initial_cell_xy", [-1, -1]), dtype=np.int64)
            if cell.shape[0] >= 2 and cell[0] >= 0 and cell[1] >= 0:
                cell_counts[int(cell[1]), int(cell[0])] += 1
                if source_label == 1:
                    semantic_cell_counts[int(cell[1]), int(cell[0])] += 1

            tracked_targets: dict[str, list[float]] = dict(record.get("tracked_targets", {}))
            errors: list[float] = []
            for target_counter_str, tracked_xy in tracked_targets.items():
                target_counter = int(target_counter_str)
                if target_counter < 0 or target_counter >= len(gt_context.poses):
                    continue
                if insert_counter < 0 or insert_counter >= len(gt_context.poses):
                    continue
                projected = _project_point(
                    gt_context.poses[insert_counter],
                    gt_context.poses[target_counter],
                    gt_context.depths[insert_counter],
                    initial_pixel,
                    gt_context.intrinsics,
                )
                if projected is None:
                    continue
                tracked = np.asarray(tracked_xy, dtype=np.float64)
                error = float(np.linalg.norm(projected - tracked))
                if math.isfinite(error):
                    errors.append(error)

            mean_error = float(np.mean(errors)) if errors else math.nan
            boundary = _depth_boundary_flag(
                gt_context.depths[insert_counter] if 0 <= insert_counter < len(gt_context.depths) else None,
                initial_pixel,
                thresh=float(gt_context.depth_boundary_thresh),
            )
            unstable = bool(errors) and mean_error > float(gt_context.unstable_error_px)
            rows.append(
                {
                    "sequence": sequence,
                    "feature_type": feature_type,
                    "status": status,
                    "patch_uid": int(uid),
                    "source": source_name,
                    "insertion_frame": insert_counter,
                    "initial_cell_x": int(cell[0]) if cell.shape[0] >= 2 else -1,
                    "initial_cell_y": int(cell[1]) if cell.shape[0] >= 2 else -1,
                    "initial_x": float(initial_pixel[0]),
                    "initial_y": float(initial_pixel[1]),
                    "selection_utility": float(record.get("selection_utility", math.nan)),
                    "repeated_patch": float(record.get("repeated_patch_flag", math.nan)),
                    "unique_semantic_count_before_repeat": float(record.get("unique_semantic_count_before_repeat", math.nan)),
                    "dedupe_radius_used": float(record.get("dedupe_radius_used", math.nan)),
                    "track_age": int(age),
                    "survived_1": bool(age >= 1),
                    "survived_3": bool(age >= 3),
                    "survived_5": bool(age >= 5),
                    "mean_gt_reprojection_error": mean_error,
                    "valid_gt_targets": int(len(errors)),
                    "near_depth_boundary": bool(boundary),
                    "unstable_motion_proxy": bool(unstable),
                }
            )

        total = max(1, native_count + dino_count)
        occupancy_flat = cell_counts.reshape(-1).astype(np.float64)
        semantic_hist = json.dumps(semantic_cell_counts.tolist())
        summary = {
            "sequence": sequence,
            "feature_type": feature_type,
            "status": status,
            "coverage": float((metrics or {}).get("coverage", math.nan)),
            "ate_rmse": float((metrics or {}).get("ate_rmse", math.nan)),
            "ate_rmse_associated": float((metrics or {}).get("ate_rmse_associated", math.nan)),
            "native_patch_fraction": float(native_count) / float(total),
            "dino_patch_fraction": float(dino_count) / float(total),
            "mean_grid_occupancy": float(np.mean(occupancy_flat)) if occupancy_flat.size else math.nan,
            "grid_occupancy_std": float(np.std(occupancy_flat)) if occupancy_flat.size else math.nan,
            "mean_unique_semantic_count_before_repeat": float(np.mean(unique_counts)) if unique_counts else math.nan,
            "repeated_patch_fraction": float(np.mean(repeated_flags)) if repeated_flags else 0.0,
            "mean_dedupe_radius_used": float(np.mean(dedupe_radii)) if dedupe_radii else math.nan,
            "per_cell_semantic_histogram": semantic_hist,
            "mean_track_age": float(np.mean(ages)) if ages else math.nan,
            "survival_rate_1": float(survived_1) / float(total),
            "survival_rate_3": float(survived_3) / float(total),
            "survival_rate_5": float(survived_5) / float(total),
        }
        return summary, rows


def init_diagnostics_outputs(summary_path: Path, patch_path: Path | None = None) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if patch_path is not None:
        patch_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sequence",
                "feature_type",
                "status",
                "coverage",
                "ate_rmse",
                "ate_rmse_associated",
                "native_patch_fraction",
                "dino_patch_fraction",
                "mean_grid_occupancy",
                "grid_occupancy_std",
                "mean_unique_semantic_count_before_repeat",
                "repeated_patch_fraction",
                "mean_dedupe_radius_used",
                "per_cell_semantic_histogram",
                "mean_track_age",
                "survival_rate_1",
                "survival_rate_3",
                "survival_rate_5",
            ]
        )
    if patch_path is not None:
        patch_path.write_text("", encoding="utf-8")


def append_diagnostics_summary(path: Path, summary: dict[str, float | str]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                summary.get("sequence"),
                summary.get("feature_type"),
                summary.get("status"),
                summary.get("coverage"),
                summary.get("ate_rmse"),
                summary.get("ate_rmse_associated"),
                summary.get("native_patch_fraction"),
                summary.get("dino_patch_fraction"),
                summary.get("mean_grid_occupancy"),
                summary.get("grid_occupancy_std"),
                summary.get("mean_unique_semantic_count_before_repeat"),
                summary.get("repeated_patch_fraction"),
                summary.get("mean_dedupe_radius_used"),
                summary.get("per_cell_semantic_histogram"),
                summary.get("mean_track_age"),
                summary.get("survival_rate_1"),
                summary.get("survival_rate_3"),
                summary.get("survival_rate_5"),
            ]
        )


def append_patch_diagnostics(path: Path | None, rows: list[dict[str, float | int | str | bool | None]]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
