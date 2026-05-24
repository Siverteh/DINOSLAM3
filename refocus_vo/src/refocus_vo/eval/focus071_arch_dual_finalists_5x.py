from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from refocus_vo.eval.aggregate_arch_dual_finalists_5x import (
    ALLOWED_STATUSES as DUAL_FINALIST_ALLOWED_STATUSES,
    aggregate_dual_finalists_benchmark,
)
from refocus_vo.eval.focus071_tumwin_finalists import (
    _assert_idle_or_raise,
    _eval_command,
    _format_command,
    _read_csv_rows,
    _run_eval,
    _write_csv,
    _write_text,
)
from refocus_vo.eval.focus071_vs_dpvo_tum_freiburg123_5x import _enumerate_freiburg_sequences


REPEATS = 5
FIXED_FRONTEND_MODE = "dino_hybrid"
FIXED_STRIDE = 4
FIXED_BACKEND_THRESH = 32.0
FIXED_IMAGE_HEIGHT = 240
FIXED_IMAGE_WIDTH = 320
FIXED_DPVO_OPTS = (
    "BUFFER_SIZE=512 PATCHES_PER_FRAME=128 REMOVAL_WINDOW=24 "
    "OPTIMIZATION_WINDOW=12 PATCH_LIFETIME=15"
)


@dataclass(frozen=True)
class LockedFinalistRequest:
    run_id: str
    ratio_id: str
    finalist_id: str


@dataclass(frozen=True)
class FinalistBenchmarkSpec:
    finalist_id: str
    run_id: str
    ratio_id: str
    frontend_mode: str
    checkpoint_path: Path
    config_path: Path
    repeat01_source_dir: Path


LOCKED_FINALISTS = (
    LockedFinalistRequest(
        run_id="multiscale_32x4_v1",
        ratio_id="hybrid50_50",
        finalist_id="multiscale_32x4_v1_hybrid50_50",
    ),
    LockedFinalistRequest(
        run_id="micro4_grid_v1",
        ratio_id="hybrid90_10",
        finalist_id="micro4_grid_v1_hybrid90_10",
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_paths(repo_root: Path) -> dict[str, Path]:
    subtree_root = repo_root / "refocus_vo"
    ablation_root = subtree_root / "runs" / "eval" / "tum_rgbd_freiburg123_arch_ratio_ablation_v1"
    old_benchmark_root = subtree_root / "runs" / "eval" / "tum_rgbd_freiburg123_dpvo_vs_focus071_v1_rerun6"
    return {
        "dataset_root": repo_root / "src" / "dino_slam3" / "data" / "tum_rgbd",
        "dpvo_root": subtree_root / "external" / "repos" / "DPVO",
        "dpvo_weights": subtree_root / "external" / "repos" / "DPVO" / "dpvo.pth",
        "dpvo_config": subtree_root / "external" / "repos" / "DPVO" / "config" / "default.yaml",
        "ablation_root": ablation_root,
        "screening_summary": ablation_root / "screening_summary.csv",
        "baseline_per_sequence": old_benchmark_root / "summary" / "per_sequence_median.csv",
        "historical_method_comparison": old_benchmark_root / "summary" / "method_comparison.csv",
        "historical_repeat_summary": old_benchmark_root / "summary" / "repeat_summary.csv",
        "output_root": subtree_root / "runs" / "eval" / "tum_rgbd_freiburg123_arch_dual_finalists_5x_v1",
    }


def _load_locked_finalists(
    *,
    screening_summary_path: Path,
    ablation_root: Path,
) -> list[FinalistBenchmarkSpec]:
    rows = {
        (
            str(row.get("run_id", "")).strip(),
            str(row.get("ratio_id", "")).strip(),
        ): row
        for row in _read_csv_rows(screening_summary_path)
    }
    finalists: list[FinalistBenchmarkSpec] = []
    for request in LOCKED_FINALISTS:
        key = (request.run_id, request.ratio_id)
        row = rows.get(key)
        if row is None:
            raise ValueError(f"Locked finalist row missing from {screening_summary_path}: {key}")
        checkpoint_path = Path(str(row.get("checkpoint_path", "")).strip()).expanduser().resolve()
        config_path = Path(str(row.get("config_path", "")).strip()).expanduser().resolve()
        frontend_mode = str(row.get("frontend_mode", "")).strip()
        repeat01_source_dir = (ablation_root / "screening" / request.run_id / request.ratio_id).resolve()
        if frontend_mode != FIXED_FRONTEND_MODE:
            raise ValueError(
                f"Locked finalist {request.run_id}:{request.ratio_id} uses unexpected frontend mode: {frontend_mode}"
            )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Locked finalist checkpoint missing: {checkpoint_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Locked finalist config missing: {config_path}")
        if not repeat01_source_dir.exists():
            raise FileNotFoundError(f"Locked finalist repeat_01 source missing: {repeat01_source_dir}")
        finalists.append(
            FinalistBenchmarkSpec(
                finalist_id=request.finalist_id,
                run_id=request.run_id,
                ratio_id=request.ratio_id,
                frontend_mode=frontend_mode,
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                repeat01_source_dir=repeat01_source_dir,
            )
        )
    return finalists


def _repeat_dir(*, output_root: Path, finalist_id: str, repeat_idx: int) -> Path:
    return output_root / finalist_id / f"repeat_{repeat_idx:02d}"


def _validate_repeat_dir(repeat_dir: Path, *, expected_sequences: list[str]) -> None:
    csv_path = repeat_dir / "dpvo_style_metrics_summary.csv"
    rows = _read_csv_rows(csv_path)
    seen = [str(row.get("sequence", "")).strip() for row in rows]
    if len(rows) != len(expected_sequences):
        raise ValueError(f"{csv_path} has {len(rows)} rows; expected {len(expected_sequences)}")
    if seen != list(expected_sequences):
        raise ValueError(f"{csv_path} sequence order/content mismatch")
    bad_rows = [row for row in rows if str(row.get("status", "")).strip() not in DUAL_FINALIST_ALLOWED_STATUSES]
    if bad_rows:
        raise ValueError(
            f"{csv_path} contains unsupported status rows: "
            + ", ".join(f"{row.get('sequence')}:{row.get('status')}" for row in bad_rows[:5])
        )


def _ensure_reused_repeat01(
    *,
    source_dir: Path,
    dest_dir: Path,
    expected_sequences: list[str],
) -> None:
    if dest_dir.exists():
        try:
            _validate_repeat_dir(dest_dir, expected_sequences=expected_sequences)
            return
        except Exception:
            shutil.rmtree(dest_dir)
    shutil.copytree(source_dir, dest_dir)
    _validate_repeat_dir(dest_dir, expected_sequences=expected_sequences)


def _print_dry_run(
    *,
    repo_root: Path,
    dataset_root: Path,
    dpvo_root: Path,
    dpvo_weights: Path,
    dpvo_config: Path,
    output_root: Path,
    sequences: list[str],
    finalists: list[FinalistBenchmarkSpec],
    repeats: int,
) -> None:
    print(f"output_root: {output_root}")
    print(f"sequence_count: {len(sequences)}")
    print(f"repeats: {repeats}")
    print("locked_finalists:")
    for finalist in finalists:
        print(
            f"  - {finalist.finalist_id}: {finalist.run_id}:{finalist.ratio_id} "
            f"checkpoint={finalist.checkpoint_path}"
        )
        print(f"    repeat_01_source={finalist.repeat01_source_dir}")
    print("repeat_commands:")
    for finalist in finalists:
        for repeat_idx in range(2, int(repeats) + 1):
            repeat_dir = _repeat_dir(
                output_root=output_root,
                finalist_id=finalist.finalist_id,
                repeat_idx=repeat_idx,
            )
            cmd = _eval_command(
                dataset_root=dataset_root,
                dpvo_root=dpvo_root,
                dpvo_weights=dpvo_weights,
                dpvo_config=dpvo_config,
                sequences=sequences,
                output_dir=repeat_dir,
                frontend_mode=finalist.frontend_mode,
                frontend_config=finalist.config_path,
                checkpoint_path=finalist.checkpoint_path,
                stride=FIXED_STRIDE,
                backend_thresh=FIXED_BACKEND_THRESH,
                image_height=FIXED_IMAGE_HEIGHT,
                image_width=FIXED_IMAGE_WIDTH,
                dpvo_opts=FIXED_DPVO_OPTS,
            )
            print(f"  [{finalist.finalist_id} repeat_{repeat_idx:02d}] {_format_command(cmd)}")


def main() -> None:
    repo_root = _repo_root()
    defaults = _default_paths(repo_root)
    ap = argparse.ArgumentParser(
        description="Run a matched 5x Freiburg benchmark for the two best dual-finalist architecture winners."
    )
    ap.add_argument("--dataset-root", default=str(defaults["dataset_root"]))
    ap.add_argument("--dpvo-root", default=str(defaults["dpvo_root"]))
    ap.add_argument("--dpvo-weights", default=str(defaults["dpvo_weights"]))
    ap.add_argument("--dpvo-config", default=str(defaults["dpvo_config"]))
    ap.add_argument("--ablation-root", default=str(defaults["ablation_root"]))
    ap.add_argument("--screening-summary", default=str(defaults["screening_summary"]))
    ap.add_argument("--baseline-per-sequence", default=str(defaults["baseline_per_sequence"]))
    ap.add_argument("--historical-method-comparison", default=str(defaults["historical_method_comparison"]))
    ap.add_argument("--historical-repeat-summary", default=str(defaults["historical_repeat_summary"]))
    ap.add_argument("--output-root", default=str(defaults["output_root"]))
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    dpvo_root = Path(args.dpvo_root).expanduser().resolve()
    dpvo_weights = Path(args.dpvo_weights).expanduser().resolve()
    dpvo_config = Path(args.dpvo_config).expanduser().resolve()
    ablation_root = Path(args.ablation_root).expanduser().resolve()
    screening_summary_path = Path(args.screening_summary).expanduser().resolve()
    baseline_per_sequence_path = Path(args.baseline_per_sequence).expanduser().resolve()
    historical_method_comparison_path = Path(args.historical_method_comparison).expanduser().resolve()
    historical_repeat_summary_path = Path(args.historical_repeat_summary).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"TUM dataset root not found: {dataset_root}")
    if not dpvo_root.exists():
        raise FileNotFoundError(f"DPVO repo not found: {dpvo_root}")
    if not dpvo_weights.exists():
        raise FileNotFoundError(f"DPVO weights not found: {dpvo_weights}")
    if not dpvo_config.exists():
        raise FileNotFoundError(f"DPVO config not found: {dpvo_config}")
    if not screening_summary_path.exists():
        raise FileNotFoundError(f"Screening summary not found: {screening_summary_path}")
    if not baseline_per_sequence_path.exists():
        raise FileNotFoundError(f"Frozen baseline per-sequence summary not found: {baseline_per_sequence_path}")
    if not historical_method_comparison_path.exists():
        raise FileNotFoundError(f"Historical method comparison not found: {historical_method_comparison_path}")
    if not historical_repeat_summary_path.exists():
        raise FileNotFoundError(f"Historical repeat summary not found: {historical_repeat_summary_path}")

    sequences = _enumerate_freiburg_sequences(dataset_root)
    finalists = _load_locked_finalists(
        screening_summary_path=screening_summary_path,
        ablation_root=ablation_root,
    )

    if args.dry_run:
        _print_dry_run(
            repo_root=repo_root,
            dataset_root=dataset_root,
            dpvo_root=dpvo_root,
            dpvo_weights=dpvo_weights,
            dpvo_config=dpvo_config,
            output_root=output_root,
            sequences=sequences,
            finalists=finalists,
            repeats=int(args.repeats),
        )
        return

    _assert_idle_or_raise(force=bool(args.force))

    output_root.mkdir(parents=True, exist_ok=True)
    _write_text(output_root / "frozen_sequences.txt", "\n".join(sequences) + "\n")
    _write_csv(
        output_root / "locked_finalists.csv",
        [
            {
                "finalist_id": finalist.finalist_id,
                "run_id": finalist.run_id,
                "ratio_id": finalist.ratio_id,
                "frontend_mode": finalist.frontend_mode,
                "checkpoint_path": str(finalist.checkpoint_path),
                "config_path": str(finalist.config_path),
                "repeat01_source_dir": str(finalist.repeat01_source_dir),
            }
            for finalist in finalists
        ],
        [
            "finalist_id",
            "run_id",
            "ratio_id",
            "frontend_mode",
            "checkpoint_path",
            "config_path",
            "repeat01_source_dir",
        ],
    )

    for finalist in finalists:
        repeat01_dir = _repeat_dir(
            output_root=output_root,
            finalist_id=finalist.finalist_id,
            repeat_idx=1,
        )
        _ensure_reused_repeat01(
            source_dir=finalist.repeat01_source_dir,
            dest_dir=repeat01_dir,
            expected_sequences=sequences,
        )
        for repeat_idx in range(2, int(args.repeats) + 1):
            repeat_dir = _repeat_dir(
                output_root=output_root,
                finalist_id=finalist.finalist_id,
                repeat_idx=repeat_idx,
            )
            if repeat_dir.exists():
                try:
                    _validate_repeat_dir(repeat_dir, expected_sequences=sequences)
                    print(
                        f"[dual_finalists_5x] reusing existing {finalist.finalist_id} "
                        f"repeat_{repeat_idx:02d}"
                    )
                    continue
                except Exception:
                    shutil.rmtree(repeat_dir)
            print(
                f"[dual_finalists_5x] running {finalist.finalist_id} "
                f"repeat_{repeat_idx:02d} on {len(sequences)} sequences"
            )
            rows = _run_eval(
                repo_root=repo_root,
                dataset_root=dataset_root,
                dpvo_root=dpvo_root,
                dpvo_weights=dpvo_weights,
                dpvo_config=dpvo_config,
                sequences=sequences,
                output_dir=repeat_dir,
                frontend_mode=finalist.frontend_mode,
                frontend_config=finalist.config_path,
                checkpoint_path=finalist.checkpoint_path,
                stride=FIXED_STRIDE,
                backend_thresh=FIXED_BACKEND_THRESH,
                image_height=FIXED_IMAGE_HEIGHT,
                image_width=FIXED_IMAGE_WIDTH,
                dpvo_opts=FIXED_DPVO_OPTS,
            )
            _validate_repeat_dir(repeat_dir, expected_sequences=sequences)

    outputs = aggregate_dual_finalists_benchmark(
        benchmark_root=output_root,
        finalist_ids=[finalist.finalist_id for finalist in finalists],
        expected_sequences=sequences,
        baseline_per_sequence_path=baseline_per_sequence_path,
        historical_method_comparison_path=historical_method_comparison_path,
        historical_repeat_summary_path=historical_repeat_summary_path,
        repeats=int(args.repeats),
    )
    for key, path in outputs.items():
        print(f"[dual_finalists_5x] {key}: {path}")
    print(f"[dual_finalists_5x] benchmark complete: {output_root}")


if __name__ == "__main__":
    main()
