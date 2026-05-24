from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from refocus_vo.eval.aggregate_euroc_three_method_5x import (
    EUROC_SEQUENCE_ORDER,
    aggregate_euroc_three_method_benchmark,
)
from refocus_vo.eval.euroc_three_method_5x import (
    FIXED_BACKEND_THRESH as EUROC_BACKEND_THRESH,
    FIXED_DPVO_OPTS as EUROC_DPVO_OPTS,
    FIXED_FRONTEND_MODE as EUROC_FRONTEND_MODE,
    FIXED_IMAGE_HEIGHT as EUROC_IMAGE_HEIGHT,
    FIXED_IMAGE_WIDTH as EUROC_IMAGE_WIDTH,
    FIXED_STRIDE as EUROC_STRIDE,
    _run_eval as _run_euroc_eval,
    _validate_repeat_dir as _validate_euroc_repeat_dir,
)
from refocus_vo.eval.focus071_tumwin_finalists import (
    ALLOWED_STATUSES as TUM_ALLOWED_STATUSES,
    _read_csv_rows,
    _run_eval as _run_tum_eval,
    _safe_float,
    _write_csv,
    _write_text,
)
from refocus_vo.eval.focus071_vs_dpvo_tum_freiburg123_5x import (
    _enumerate_freiburg_sequences,
    _gpu_heavy_process_lines,
)


REPEATS = 5

TUM_FRONTEND_MODE = "dino_hybrid"
TUM_STRIDE = 4
TUM_BACKEND_THRESH = 32.0
TUM_IMAGE_HEIGHT = 240
TUM_IMAGE_WIDTH = 320
TUM_DPVO_OPTS = (
    "BUFFER_SIZE=512 PATCHES_PER_FRAME=128 REMOVAL_WINDOW=24 "
    "OPTIMIZATION_WINDOW=12 PATCH_LIFETIME=15"
)

TUM_FAMILY_ORDER = ("freiburg1", "freiburg2", "freiburg3")


@dataclass(frozen=True)
class SelectedSweepMethod:
    run_id: str
    method_id: str
    checkpoint_path: Path
    config_path: Path
    selection_score: float


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_paths(repo_root: Path) -> dict[str, Path]:
    subtree_root = repo_root / "refocus_vo"
    return {
        "sweep_root": subtree_root / "runs" / "sweeps" / "dino_dpvo_cross_dataset_tri_proxy_live_v2",
        "leaderboard": subtree_root / "runs" / "sweeps" / "dino_dpvo_cross_dataset_tri_proxy_live_v2" / "leaderboard_dev.csv",
        "tum_dataset_root": repo_root / "src" / "dino_slam3" / "data" / "tum_rgbd",
        "euroc_dataset_root": subtree_root / "data" / "euroc_asl",
        "dpvo_root": subtree_root / "external" / "repos" / "DPVO",
        "dpvo_weights": subtree_root / "external" / "repos" / "DPVO" / "dpvo.pth",
        "dpvo_config": subtree_root / "external" / "repos" / "DPVO" / "config" / "default.yaml",
        "output_root": subtree_root / "runs" / "eval" / "sweepbest_tum_euroc_5x_v1",
    }


def _truthy(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _selection_score_and_gate(row: dict[str, str]) -> tuple[float, bool]:
    score = _safe_float(row.get("best_dual_score"))
    gate = _truthy(row.get("best_dual_gate"))
    if math.isfinite(score):
        return score, gate
    score = _safe_float(row.get("best_dual_proxy_score"))
    gate = _truthy(row.get("best_dual_proxy_gate"))
    if math.isfinite(score):
        return score, gate
    score = _safe_float(row.get("best_tri_score"))
    gate = _truthy(row.get("best_tri_gate"))
    return score, gate


def _select_best_method(leaderboard_path: Path, *, run_id: str | None) -> SelectedSweepMethod:
    rows = _read_csv_rows(leaderboard_path)
    candidates: list[dict[str, str]] = []
    for row in rows:
        if run_id is not None and str(row.get("run_id", "")).strip() != run_id:
            continue
        checkpoint_text = str(row.get("checkpoint_path", "")).strip()
        config_text = str(row.get("config_path", "")).strip()
        selection_score, passed_gate = _selection_score_and_gate(row)
        if not checkpoint_text or not config_text or not math.isfinite(selection_score):
            continue
        if not passed_gate:
            continue
        candidates.append(row)
    if not candidates:
        if run_id is not None:
            raise RuntimeError(f"No gated candidate found for run_id={run_id} in {leaderboard_path}")
        raise RuntimeError(f"No gated candidate found in {leaderboard_path}")
    best_row = min(candidates, key=lambda row: _selection_score_and_gate(row)[0])
    checkpoint_path = Path(str(best_row.get("checkpoint_path", "")).strip()).expanduser().resolve()
    config_path = Path(str(best_row.get("config_path", "")).strip()).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint missing: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Best config missing: {config_path}")
    selected_run_id = str(best_row.get("run_id", "")).strip()
    return SelectedSweepMethod(
        run_id=selected_run_id,
        method_id=selected_run_id,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        selection_score=_selection_score_and_gate(best_row)[0],
    )


def _assert_idle_or_raise() -> None:
    active = _gpu_heavy_process_lines(exclude_pid=None)
    filtered = [line for line in active if "sweepbest_tum_euroc_5x" not in line]
    if filtered:
        raise RuntimeError(
            "Refusing to start because another GPU-heavy job is active:\n"
            + "\n".join(filtered[:10])
        )


def _tum_repeat_dir(*, benchmark_root: Path, method_id: str, repeat_idx: int) -> Path:
    return benchmark_root / method_id / f"repeat_{repeat_idx:02d}"


def _validate_tum_repeat_dir(repeat_dir: Path, *, expected_sequences: list[str]) -> None:
    csv_path = repeat_dir / "dpvo_style_metrics_summary.csv"
    rows = _read_csv_rows(csv_path)
    seen = [str(row.get("sequence", "")).strip() for row in rows]
    if len(rows) != len(expected_sequences):
        raise ValueError(f"{csv_path} has {len(rows)} rows; expected {len(expected_sequences)}")
    if seen != list(expected_sequences):
        raise ValueError(f"{csv_path} sequence order/content mismatch")
    bad_rows = [row for row in rows if str(row.get("status", "")).strip() not in TUM_ALLOWED_STATUSES]
    if bad_rows:
        raise ValueError(
            f"{csv_path} contains unsupported status rows: "
            + ", ".join(f"{row.get('sequence')}:{row.get('status')}" for row in bad_rows[:5])
        )


def _mean(values: list[float]) -> float:
    usable = [float(v) for v in values if math.isfinite(float(v))]
    if not usable:
        return math.nan
    return float(sum(usable) / len(usable))


def _median(values: list[float]) -> float:
    usable = [float(v) for v in values if math.isfinite(float(v))]
    if not usable:
        return math.nan
    return float(statistics.median(usable))


def _tum_family_for_sequence(sequence: str) -> str:
    seq = str(sequence).strip()
    for family in TUM_FAMILY_ORDER:
        if seq.startswith(family + "_"):
            return family
    raise ValueError(f"Unsupported Freiburg sequence: {sequence}")


def _aggregate_tum_single_method(
    *,
    benchmark_root: Path,
    method_id: str,
    expected_sequences: list[str],
    repeats: int,
) -> dict[str, Path]:
    repeat_rows: list[list[dict[str, str]]] = []
    repeat_summary_rows: list[dict[str, object]] = []
    for repeat_idx in range(1, int(repeats) + 1):
        repeat_id = f"repeat_{repeat_idx:02d}"
        csv_path = benchmark_root / method_id / repeat_id / "dpvo_style_metrics_summary.csv"
        rows = _read_csv_rows(csv_path)
        _validate_tum_repeat_dir(csv_path.parent, expected_sequences=expected_sequences)
        repeat_rows.append(rows)
        family_rows = {
            family: [row for row in rows if _tum_family_for_sequence(str(row.get("sequence", ""))) == family]
            for family in TUM_FAMILY_ORDER
        }
        row_out: dict[str, object] = {
            "method_id": method_id,
            "repeat_id": repeat_id,
            "full_mean_ate_rmse_associated": f"{_mean([_safe_float(row.get('ate_rmse_associated')) for row in rows]):.6f}",
            "full_mean_coverage": f"{_mean([_safe_float(row.get('coverage')) for row in rows]):.6f}",
            "ok_count": sum(1 for row in rows if str(row.get("status", "")).strip() == "ok"),
            "non_ok_count": sum(1 for row in rows if str(row.get("status", "")).strip() != "ok"),
            "finite_count": sum(1 for row in rows if math.isfinite(_safe_float(row.get("ate_rmse_associated")))),
        }
        for family in TUM_FAMILY_ORDER:
            row_out[f"{family}_mean_ate_rmse_associated"] = f"{_mean([_safe_float(row.get('ate_rmse_associated')) for row in family_rows[family]]):.6f}"
        repeat_summary_rows.append(row_out)

    per_sequence_rows: list[dict[str, object]] = []
    for sequence in expected_sequences:
        seq_rows = [
            next(row for row in rows if str(row.get("sequence", "")).strip() == sequence)
            for rows in repeat_rows
        ]
        per_sequence_rows.append(
            {
                "method_id": method_id,
                "sequence": sequence,
                "family": _tum_family_for_sequence(sequence),
                "median_ate_rmse_associated": f"{_median([_safe_float(row.get('ate_rmse_associated')) for row in seq_rows]):.6f}",
                "median_coverage": f"{_median([_safe_float(row.get('coverage')) for row in seq_rows]):.6f}",
                "ok_repeat_count": sum(1 for row in seq_rows if str(row.get("status", "")).strip() == "ok"),
                "non_ok_repeat_count": sum(1 for row in seq_rows if str(row.get("status", "")).strip() != "ok"),
            }
        )

    method_row: dict[str, object] = {
        "method_id": method_id,
        "repeat_count": len(repeat_summary_rows),
        "full_mean_of_repeat_means_ate_rmse_associated": f"{_mean([_safe_float(row.get('full_mean_ate_rmse_associated')) for row in repeat_summary_rows]):.6f}",
        "full_median_of_repeat_means_ate_rmse_associated": f"{_median([_safe_float(row.get('full_mean_ate_rmse_associated')) for row in repeat_summary_rows]):.6f}",
        "full_mean_of_sequence_medians_ate_rmse_associated": f"{_mean([_safe_float(row.get('median_ate_rmse_associated')) for row in per_sequence_rows]):.6f}",
        "average_ok_count": f"{_mean([_safe_float(row.get('ok_count')) for row in repeat_summary_rows]):.6f}",
        "average_non_ok_count": f"{_mean([_safe_float(row.get('non_ok_count')) for row in repeat_summary_rows]):.6f}",
    }
    for family in TUM_FAMILY_ORDER:
        family_seq_rows = [row for row in per_sequence_rows if str(row.get("family")) == family]
        method_row[f"{family}_mean_of_repeat_means_ate_rmse_associated"] = f"{_mean([_safe_float(row.get(f'{family}_mean_ate_rmse_associated')) for row in repeat_summary_rows]):.6f}"
        method_row[f"{family}_median_of_repeat_means_ate_rmse_associated"] = f"{_median([_safe_float(row.get(f'{family}_mean_ate_rmse_associated')) for row in repeat_summary_rows]):.6f}"
        method_row[f"{family}_mean_of_sequence_medians_ate_rmse_associated"] = f"{_mean([_safe_float(row.get('median_ate_rmse_associated')) for row in family_seq_rows]):.6f}"

    summary_dir = benchmark_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    repeat_path = summary_dir / "repeat_summary.csv"
    per_sequence_path = summary_dir / "per_sequence_median.csv"
    method_path = summary_dir / "method_comparison.csv"
    md_path = summary_dir / "method_comparison.md"

    _write_csv(
        repeat_path,
        repeat_summary_rows,
        [
            "method_id",
            "repeat_id",
            "full_mean_ate_rmse_associated",
            "full_mean_coverage",
            "ok_count",
            "non_ok_count",
            "finite_count",
            "freiburg1_mean_ate_rmse_associated",
            "freiburg2_mean_ate_rmse_associated",
            "freiburg3_mean_ate_rmse_associated",
        ],
    )
    _write_csv(
        per_sequence_path,
        per_sequence_rows,
        [
            "method_id",
            "sequence",
            "family",
            "median_ate_rmse_associated",
            "median_coverage",
            "ok_repeat_count",
            "non_ok_repeat_count",
        ],
    )
    _write_csv(
        method_path,
        [method_row],
        [
            "method_id",
            "repeat_count",
            "full_mean_of_repeat_means_ate_rmse_associated",
            "full_median_of_repeat_means_ate_rmse_associated",
            "full_mean_of_sequence_medians_ate_rmse_associated",
            "freiburg1_mean_of_repeat_means_ate_rmse_associated",
            "freiburg1_median_of_repeat_means_ate_rmse_associated",
            "freiburg1_mean_of_sequence_medians_ate_rmse_associated",
            "freiburg2_mean_of_repeat_means_ate_rmse_associated",
            "freiburg2_median_of_repeat_means_ate_rmse_associated",
            "freiburg2_mean_of_sequence_medians_ate_rmse_associated",
            "freiburg3_mean_of_repeat_means_ate_rmse_associated",
            "freiburg3_median_of_repeat_means_ate_rmse_associated",
            "freiburg3_mean_of_sequence_medians_ate_rmse_associated",
            "average_ok_count",
            "average_non_ok_count",
        ],
    )
    _write_text(
        md_path,
        "\n".join(
            [
                "# Sweep Best Full TUM 5x",
                "",
                f"Method: `{method_id}`",
                "",
                f"- Mean of repeat means: `{method_row['full_mean_of_repeat_means_ate_rmse_associated']}`",
                f"- Median of repeat means: `{method_row['full_median_of_repeat_means_ate_rmse_associated']}`",
                f"- Mean of sequence medians: `{method_row['full_mean_of_sequence_medians_ate_rmse_associated']}`",
                f"- Freiburg1 mean of sequence medians: `{method_row['freiburg1_mean_of_sequence_medians_ate_rmse_associated']}`",
                f"- Freiburg2 mean of sequence medians: `{method_row['freiburg2_mean_of_sequence_medians_ate_rmse_associated']}`",
                f"- Freiburg3 mean of sequence medians: `{method_row['freiburg3_mean_of_sequence_medians_ate_rmse_associated']}`",
                "",
            ]
        ),
    )
    return {
        "repeat_summary": repeat_path,
        "per_sequence_median": per_sequence_path,
        "method_comparison": method_path,
        "method_comparison_md": md_path,
    }


def _print_dry_run(
    *,
    selected: SelectedSweepMethod,
    tum_root: Path,
    euroc_root: Path,
    tum_sequences: list[str],
    euroc_sequences: list[str],
    repeats: int,
) -> None:
    print(f"selected_run_id: {selected.run_id}")
    print(f"checkpoint: {selected.checkpoint_path}")
    print(f"config: {selected.config_path}")
    print(f"selection_score: {selected.selection_score:.6f}")
    print(f"tum_output_root: {tum_root}")
    print(f"euroc_output_root: {euroc_root}")
    print(f"repeats: {repeats}")
    print(f"tum_sequence_count: {len(tum_sequences)}")
    print(f"euroc_sequence_count: {len(euroc_sequences)}")


def main() -> None:
    repo_root = _repo_root()
    defaults = _default_paths(repo_root)
    ap = argparse.ArgumentParser(
        description="Run full 5x TUM RGB-D and EuRoC benchmarks for the current best live-tri-proxy sweep checkpoint."
    )
    ap.add_argument("--leaderboard", default=str(defaults["leaderboard"]))
    ap.add_argument("--run-id", default="")
    ap.add_argument("--tum-dataset-root", default=str(defaults["tum_dataset_root"]))
    ap.add_argument("--euroc-dataset-root", default=str(defaults["euroc_dataset_root"]))
    ap.add_argument("--dpvo-root", default=str(defaults["dpvo_root"]))
    ap.add_argument("--dpvo-weights", default=str(defaults["dpvo_weights"]))
    ap.add_argument("--dpvo-config", default=str(defaults["dpvo_config"]))
    ap.add_argument("--output-root", default=str(defaults["output_root"]))
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    leaderboard_path = Path(args.leaderboard).expanduser().resolve()
    tum_dataset_root = Path(args.tum_dataset_root).expanduser().resolve()
    euroc_dataset_root = Path(args.euroc_dataset_root).expanduser().resolve()
    dpvo_root = Path(args.dpvo_root).expanduser().resolve()
    dpvo_weights = Path(args.dpvo_weights).expanduser().resolve()
    dpvo_config = Path(args.dpvo_config).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not leaderboard_path.exists():
        raise FileNotFoundError(f"Leaderboard not found: {leaderboard_path}")
    if not tum_dataset_root.exists():
        raise FileNotFoundError(f"TUM dataset root not found: {tum_dataset_root}")
    if not euroc_dataset_root.exists():
        raise FileNotFoundError(f"EuRoC dataset root not found: {euroc_dataset_root}")
    if not dpvo_root.exists():
        raise FileNotFoundError(f"DPVO repo not found: {dpvo_root}")
    if not dpvo_weights.exists():
        raise FileNotFoundError(f"DPVO weights not found: {dpvo_weights}")
    if not dpvo_config.exists():
        raise FileNotFoundError(f"DPVO config not found: {dpvo_config}")

    selected = _select_best_method(
        leaderboard_path,
        run_id=str(args.run_id).strip() or None,
    )
    tum_sequences = _enumerate_freiburg_sequences(tum_dataset_root)
    euroc_sequences = list(EUROC_SEQUENCE_ORDER)
    tum_root = output_root / "tum_rgbd_freiburg123"
    euroc_root = output_root / "euroc"

    if args.dry_run:
        _print_dry_run(
            selected=selected,
            tum_root=tum_root,
            euroc_root=euroc_root,
            tum_sequences=tum_sequences,
            euroc_sequences=euroc_sequences,
            repeats=int(args.repeats),
        )
        return

    if not args.force:
        _assert_idle_or_raise()

    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_root / "selected_candidate.csv",
        [
            {
                "run_id": selected.run_id,
                "method_id": selected.method_id,
                "checkpoint_path": str(selected.checkpoint_path),
                "config_path": str(selected.config_path),
                "selection_score": f"{selected.selection_score:.6f}",
            }
        ],
        ["run_id", "method_id", "checkpoint_path", "config_path", "selection_score"],
    )

    _write_text(output_root / "selected_candidate.md", "\n".join([
        "# Selected Sweep Candidate",
        "",
        f"- run_id: `{selected.run_id}`",
        f"- checkpoint: `{selected.checkpoint_path}`",
        f"- config: `{selected.config_path}`",
        f"- live selection score: `{selected.selection_score:.6f}`",
        "",
    ]))

    _write_text(tum_root / "frozen_sequences.txt", "\n".join(tum_sequences) + "\n")
    for repeat_idx in range(1, int(args.repeats) + 1):
        repeat_dir = _tum_repeat_dir(benchmark_root=tum_root, method_id=selected.method_id, repeat_idx=repeat_idx)
        if repeat_dir.exists():
            try:
                _validate_tum_repeat_dir(repeat_dir, expected_sequences=tum_sequences)
                print(f"[sweepbest_tum_5x] reusing existing {selected.method_id} repeat_{repeat_idx:02d}")
                continue
            except Exception:
                import shutil

                shutil.rmtree(repeat_dir)
        print(f"[sweepbest_tum_5x] running {selected.method_id} repeat_{repeat_idx:02d}")
        _run_tum_eval(
            repo_root=repo_root,
            dataset_root=tum_dataset_root,
            dpvo_root=dpvo_root,
            dpvo_weights=dpvo_weights,
            dpvo_config=dpvo_config,
            sequences=tum_sequences,
            output_dir=repeat_dir,
            frontend_mode=TUM_FRONTEND_MODE,
            frontend_config=selected.config_path,
            checkpoint_path=selected.checkpoint_path,
            stride=TUM_STRIDE,
            backend_thresh=TUM_BACKEND_THRESH,
            image_height=TUM_IMAGE_HEIGHT,
            image_width=TUM_IMAGE_WIDTH,
            dpvo_opts=TUM_DPVO_OPTS,
        )
        _validate_tum_repeat_dir(repeat_dir, expected_sequences=tum_sequences)

    tum_outputs = _aggregate_tum_single_method(
        benchmark_root=tum_root,
        method_id=selected.method_id,
        expected_sequences=tum_sequences,
        repeats=int(args.repeats),
    )
    for key, path in tum_outputs.items():
        print(f"[sweepbest_tum_5x] {key}: {path}")

    _write_text(euroc_root / "frozen_sequences.txt", "\n".join(euroc_sequences) + "\n")
    from refocus_vo.eval.euroc_three_method_5x import LockedMethodSpec

    euroc_method = LockedMethodSpec(
        method_id=selected.method_id,
        frontend_mode=EUROC_FRONTEND_MODE,
        checkpoint_path=selected.checkpoint_path,
        config_path=selected.config_path,
        repeat01_source_dir=None,
    )
    for repeat_idx in range(1, int(args.repeats) + 1):
        repeat_dir = euroc_root / selected.method_id / f"repeat_{repeat_idx:02d}"
        if repeat_dir.exists():
            try:
                _validate_euroc_repeat_dir(repeat_dir, expected_sequences=euroc_sequences)
                print(f"[sweepbest_euroc_5x] reusing existing {selected.method_id} repeat_{repeat_idx:02d}")
                continue
            except Exception:
                import shutil

                shutil.rmtree(repeat_dir)
        print(f"[sweepbest_euroc_5x] running {selected.method_id} repeat_{repeat_idx:02d}")
        _run_euroc_eval(
            repo_root=repo_root,
            dataset_root=euroc_dataset_root,
            dpvo_root=dpvo_root,
            dpvo_weights=dpvo_weights,
            dpvo_config=dpvo_config,
            output_dir=repeat_dir,
            method=euroc_method,
            sequences=euroc_sequences,
        )
        _validate_euroc_repeat_dir(repeat_dir, expected_sequences=euroc_sequences)

    euroc_outputs = aggregate_euroc_three_method_benchmark(
        benchmark_root=euroc_root,
        method_ids=[selected.method_id],
        expected_sequences=euroc_sequences,
        repeats=int(args.repeats),
    )
    for key, path in euroc_outputs.items():
        print(f"[sweepbest_euroc_5x] {key}: {path}")

    print(f"[sweepbest_tum_euroc_5x] benchmark complete: {output_root}")


if __name__ == "__main__":
    main()
