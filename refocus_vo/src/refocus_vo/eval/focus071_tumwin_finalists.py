from __future__ import annotations

import argparse
import csv
import math
import os
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from refocus_vo.eval.focus071_vs_dpvo_tum_freiburg123_5x import (
    TUM_MAX_DT,
    TUM_MISSING_PENALTY_METERS,
    TUM_MIN_COVERAGE_OK,
    _enumerate_freiburg_sequences,
    _gpu_heavy_process_lines,
    _split_dpvo_opts,
)
from refocus_vo.sweeps.run_assoc9_sweep import (
    _load_yaml,
    _materialize_pure_pack_config,
    _top_dev_candidates,
)


ALLOWED_STATUSES = {"ok", "partial_low_coverage"}


@dataclass(frozen=True)
class FinalistCandidate:
    run_id: str
    mode: str
    checkpoint_path: Path
    config_path: Path
    best_assoc: float
    best_secondary_assoc: float
    pure_checkpoint_path: Path | None = None
    hybrid_checkpoint_path: Path | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def _family_for_sequence(sequence: str) -> str:
    seq = str(sequence).strip()
    if seq.startswith("freiburg1_"):
        return "freiburg1"
    if seq.startswith("freiburg2_"):
        return "freiburg2"
    if seq.startswith("freiburg3_"):
        return "freiburg3"
    raise ValueError(f"Unsupported Freiburg sequence: {sequence}")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _default_paths(repo_root: Path) -> dict[str, Path]:
    subtree_root = repo_root / "refocus_vo"
    return {
        "dataset_root": repo_root / "src" / "dino_slam3" / "data" / "tum_rgbd",
        "dpvo_root": subtree_root / "external" / "repos" / "DPVO",
        "dpvo_weights": subtree_root / "external" / "repos" / "DPVO" / "dpvo.pth",
        "dpvo_config": subtree_root / "external" / "repos" / "DPVO" / "config" / "default.yaml",
        "sweep_manifest": subtree_root / "configs" / "dino_dpvo_focus071_arch5x2_tumwin_sweep_v1.yaml",
        "sweep_root": subtree_root / "runs" / "sweeps" / "dino_dpvo_focus071_arch5x2_tumwin_sweep_v1",
        "baseline_per_sequence": subtree_root / "runs" / "eval" / "tum_rgbd_freiburg123_dpvo_vs_focus071_v1_rerun6" / "summary" / "per_sequence_median.csv",
        "screening_output_root": subtree_root / "runs" / "eval" / "tum_rgbd_freiburg123_focus071_arch5x2_finalists_v1",
        "winner_output_root": subtree_root / "runs" / "eval" / "tum_rgbd_freiburg123_focus071_arch5x2_final_winner_5x_v1",
    }


def _read_frozen_dpvo_baseline(path: Path, *, expected_sequences: list[str]) -> dict[str, float]:
    rows = _read_csv_rows(path)
    baseline = {
        str(row.get("sequence", "")).strip(): _safe_float(row.get("median_ate_rmse_associated"))
        for row in rows
        if str(row.get("method", "")).strip() == "dpvo_native"
    }
    missing = [seq for seq in expected_sequences if seq not in baseline or not math.isfinite(baseline[seq])]
    if missing:
        raise ValueError(f"Frozen DPVO baseline missing sequences: {missing}")
    return baseline


def _read_leaderboard_candidates(
    *,
    leaderboard_path: Path,
    top_k: int,
    coverage_gate: float,
    secondary_coverage_gate: float | None,
) -> list[FinalistCandidate]:
    rows = _read_csv_rows(leaderboard_path)
    top_rows = _top_dev_candidates(
        rows,
        coverage_gate=coverage_gate,
        secondary_coverage_gate=secondary_coverage_gate,
        secondary_assoc_guardrail=None,
        limit=top_k,
    )
    candidates: list[FinalistCandidate] = []
    for row in top_rows:
        checkpoint_path = Path(str(row.get("checkpoint_path", "")).strip()).expanduser().resolve()
        config_path = Path(str(row.get("config_path", "")).strip()).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing finalist checkpoint: {checkpoint_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Missing finalist config: {config_path}")
        output_dir = checkpoint_path.parent
        pure_checkpoint_path = output_dir / "best_pure100.pt"
        hybrid_checkpoint_path = output_dir / "best_hybrid.pt"
        candidates.append(
            FinalistCandidate(
                run_id=str(row.get("run_id", "")).strip(),
                mode=str(row.get("best_mode", "pure100")).strip() or "pure100",
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                best_assoc=_safe_float(row.get("best_assoc")),
                best_secondary_assoc=_safe_float(row.get("best_lowtex_assoc")),
                pure_checkpoint_path=pure_checkpoint_path if pure_checkpoint_path.exists() else None,
                hybrid_checkpoint_path=hybrid_checkpoint_path if hybrid_checkpoint_path.exists() else None,
            )
        )
    if len(candidates) < top_k:
        raise RuntimeError(
            f"Expected {top_k} finalist candidates from {leaderboard_path}, found {len(candidates)}"
        )
    return candidates


def _expand_mode_candidates(candidates: list[FinalistCandidate]) -> list[FinalistCandidate]:
    expanded: list[FinalistCandidate] = []
    for candidate in candidates:
        if candidate.pure_checkpoint_path is not None:
            expanded.append(
                FinalistCandidate(
                    run_id=candidate.run_id,
                    mode="pure100",
                    checkpoint_path=candidate.pure_checkpoint_path,
                    config_path=candidate.config_path,
                    best_assoc=candidate.best_assoc,
                    best_secondary_assoc=candidate.best_secondary_assoc,
                    pure_checkpoint_path=candidate.pure_checkpoint_path,
                    hybrid_checkpoint_path=candidate.hybrid_checkpoint_path,
                )
            )
        if candidate.hybrid_checkpoint_path is not None:
            expanded.append(
                FinalistCandidate(
                    run_id=candidate.run_id,
                    mode="hybrid",
                    checkpoint_path=candidate.hybrid_checkpoint_path,
                    config_path=candidate.config_path,
                    best_assoc=candidate.best_assoc,
                    best_secondary_assoc=candidate.best_secondary_assoc,
                    pure_checkpoint_path=candidate.pure_checkpoint_path,
                    hybrid_checkpoint_path=candidate.hybrid_checkpoint_path,
                )
            )
    return expanded


def _format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _eval_command(
    *,
    dataset_root: Path,
    dpvo_root: Path,
    dpvo_weights: Path,
    dpvo_config: Path,
    sequences: list[str],
    output_dir: Path,
    frontend_mode: str,
    frontend_config: Path,
    checkpoint_path: Path,
    stride: int,
    backend_thresh: float,
    image_height: int,
    image_width: int,
    dpvo_opts: str,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "refocus_vo.eval.external_dpvo",
        "--dataset-root",
        str(dataset_root),
        "--dpvo-root",
        str(dpvo_root),
        "--weights",
        str(dpvo_weights),
        "--config",
        str(dpvo_config),
        "--output-dir",
        str(output_dir),
        "--csv-path",
        str(output_dir / "metrics_summary.csv"),
        "--dpvo-style-csv-path",
        str(output_dir / "dpvo_style_metrics_summary.csv"),
        "--sequences",
        ",".join(sequences),
        "--max-dt",
        str(TUM_MAX_DT),
        "--missing-penalty-m",
        str(TUM_MISSING_PENALTY_METERS),
        "--min-coverage-ok",
        str(TUM_MIN_COVERAGE_OK),
        "--stride",
        str(int(stride)),
        "--backend-thresh",
        str(float(backend_thresh)),
        "--image-height",
        str(int(image_height)),
        "--image-width",
        str(int(image_width)),
        "--frontend-mode",
        str(frontend_mode),
        "--frontend-config",
        str(frontend_config),
        "--frontend-checkpoint",
        str(checkpoint_path),
        "--collect-diagnostics",
    ]
    opts = _split_dpvo_opts(dpvo_opts)
    if opts:
        cmd.append("--opts")
        cmd.extend(opts)
    return cmd


def _validate_eval_rows(rows: list[dict[str, str]], *, expected_sequences: list[str], csv_path: Path) -> None:
    seen = [str(row.get("sequence", "")).strip() for row in rows]
    if seen != list(expected_sequences):
        raise ValueError(f"{csv_path} sequence order/content mismatch")
    bad_rows = [row for row in rows if str(row.get("status", "")).strip() not in ALLOWED_STATUSES]
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


def _screening_summary_row(
    *,
    run_id: str,
    mode: str = "pure100",
    rows: list[dict[str, str]],
    baseline_assoc: dict[str, float],
    checkpoint_path: Path,
    config_path: Path,
    best_assoc: float,
    best_secondary_assoc: float,
) -> dict[str, object]:
    wins = 0
    losses = 0
    ties = 0
    family_wins = {"freiburg1": 0, "freiburg2": 0, "freiburg3": 0}
    family_losses = {"freiburg1": 0, "freiburg2": 0, "freiburg3": 0}
    for row in rows:
        sequence = str(row.get("sequence", "")).strip()
        family = _family_for_sequence(sequence)
        assoc = _safe_float(row.get("ate_rmse_associated"))
        baseline = baseline_assoc[sequence]
        if assoc < baseline:
            wins += 1
            family_wins[family] += 1
        elif baseline < assoc:
            losses += 1
            family_losses[family] += 1
        else:
            ties += 1
    return {
        "run_id": run_id,
        "mode": mode,
        "candidate_id": f"{run_id}:{mode}",
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "best_proxy_assoc": f"{best_assoc:.6f}",
        "best_secondary_assoc": f"{best_secondary_assoc:.6f}",
        "wins_vs_dpvo_median": wins,
        "losses_vs_dpvo_median": losses,
        "ties_vs_dpvo_median": ties,
        "full_mean_ate_rmse": f"{_mean([_safe_float(row.get('ate_rmse')) for row in rows]):.6f}",
        "full_mean_ate_rmse_associated": f"{_mean([_safe_float(row.get('ate_rmse_associated')) for row in rows]):.6f}",
        "full_mean_coverage": f"{_mean([_safe_float(row.get('coverage')) for row in rows]):.6f}",
        "freiburg1_wins": family_wins["freiburg1"],
        "freiburg2_wins": family_wins["freiburg2"],
        "freiburg3_wins": family_wins["freiburg3"],
        "freiburg1_losses": family_losses["freiburg1"],
        "freiburg2_losses": family_losses["freiburg2"],
        "freiburg3_losses": family_losses["freiburg3"],
    }


def _winner_key(row: dict[str, object]) -> tuple[float, ...]:
    return (
        -float(row.get("wins_vs_dpvo_median", 0)),
        float(row.get("full_mean_ate_rmse_associated", math.inf)),
        -float(row.get("freiburg2_wins", 0)),
        -float(row.get("freiburg3_wins", 0)),
        0.0 if str(row.get("mode", "pure100")).strip().lower() == "pure100" else 1.0,
    )


def _run_eval(
    *,
    repo_root: Path,
    dataset_root: Path,
    dpvo_root: Path,
    dpvo_weights: Path,
    dpvo_config: Path,
    sequences: list[str],
    output_dir: Path,
    frontend_mode: str,
    frontend_config: Path,
    checkpoint_path: Path,
    stride: int,
    backend_thresh: float,
    image_height: int,
    image_width: int,
    dpvo_opts: str,
) -> list[dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = _eval_command(
        dataset_root=dataset_root,
        dpvo_root=dpvo_root,
        dpvo_weights=dpvo_weights,
        dpvo_config=dpvo_config,
        sequences=sequences,
        output_dir=output_dir,
        frontend_mode=frontend_mode,
        frontend_config=frontend_config,
        checkpoint_path=checkpoint_path,
        stride=stride,
        backend_thresh=backend_thresh,
        image_height=image_height,
        image_width=image_width,
        dpvo_opts=dpvo_opts,
    )
    (output_dir / "command.txt").write_text(_format_command(cmd) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(repo_root / "refocus_vo" / "src") + ":" + str(dpvo_root) + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    with (output_dir / "run.log").open("w", encoding="utf-8") as log_file:
        subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )
    csv_path = output_dir / "dpvo_style_metrics_summary.csv"
    rows = _read_csv_rows(csv_path)
    return rows


def _screening_markdown(rows: list[dict[str, object]], winner_row: dict[str, object]) -> str:
    lines = [
        "# Focus071 TUM Finalists",
        "",
        "| Candidate | Mode | Wins vs DPVO median | Losses | Full mean assoc ATE | Freiburg1 wins | Freiburg2 wins | Freiburg3 wins |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.get('candidate_id', row['run_id'])}` | `{row.get('mode', 'pure100')}` | {row['wins_vs_dpvo_median']} | {row['losses_vs_dpvo_median']} | "
            f"{row['full_mean_ate_rmse_associated']} | {row['freiburg1_wins']} | {row['freiburg2_wins']} | {row['freiburg3_wins']} |"
        )
    lines.extend(
        [
            "",
            f"Winner: `{winner_row.get('candidate_id', winner_row['run_id'])}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _winner_repeat_summary_row(
    *,
    repeat_id: str,
    rows: list[dict[str, str]],
    baseline_assoc: dict[str, float],
) -> dict[str, object]:
    summary = _screening_summary_row(
        run_id=repeat_id,
        rows=rows,
        baseline_assoc=baseline_assoc,
        checkpoint_path=Path(""),
        config_path=Path(""),
        best_assoc=math.nan,
        best_secondary_assoc=math.nan,
    )
    summary.pop("checkpoint_path", None)
    summary.pop("config_path", None)
    summary.pop("best_proxy_assoc", None)
    summary.pop("best_secondary_assoc", None)
    summary["repeat_id"] = summary.pop("run_id")
    return summary


def _winner_per_sequence_medians(
    *,
    repeat_rows: list[list[dict[str, str]]],
    expected_sequences: list[str],
    baseline_assoc: dict[str, float],
) -> list[dict[str, object]]:
    rows_out: list[dict[str, object]] = []
    for sequence in expected_sequences:
        seq_rows = [
            next(row for row in rows if str(row.get("sequence", "")).strip() == sequence)
            for rows in repeat_rows
        ]
        median_assoc = statistics.median([_safe_float(row.get("ate_rmse_associated")) for row in seq_rows])
        baseline = baseline_assoc[sequence]
        winner = "tie"
        if median_assoc < baseline:
            winner = "focus071_best"
        elif baseline < median_assoc:
            winner = "dpvo_native"
        rows_out.append(
            {
                "sequence": sequence,
                "family": _family_for_sequence(sequence),
                "baseline_dpvo_assoc_median": f"{baseline:.6f}",
                "winner_assoc_median": f"{median_assoc:.6f}",
                "winner_ate_median": f"{statistics.median([_safe_float(row.get('ate_rmse')) for row in seq_rows]):.6f}",
                "winner_coverage_median": f"{statistics.median([_safe_float(row.get('coverage')) for row in seq_rows]):.6f}",
                "winner_vs_baseline": winner,
            }
        )
    return rows_out


def _winner_summary_markdown(
    *,
    winner_row: dict[str, object],
    repeat_rows: list[dict[str, object]],
    per_sequence_rows: list[dict[str, object]],
) -> str:
    avg_wins = statistics.mean(float(row["wins_vs_dpvo_median"]) for row in repeat_rows)
    med_wins = statistics.median(float(row["wins_vs_dpvo_median"]) for row in repeat_rows)
    mean_assoc = statistics.mean(float(row["full_mean_ate_rmse_associated"]) for row in repeat_rows)
    median_assoc = statistics.median(float(row["full_mean_ate_rmse_associated"]) for row in repeat_rows)
    seq_wins = sum(1 for row in per_sequence_rows if row["winner_vs_baseline"] == "focus071_best")
    seq_losses = sum(1 for row in per_sequence_rows if row["winner_vs_baseline"] == "dpvo_native")
    lines = [
        "# Focus071 Final Winner 5x",
        "",
        f"Winner candidate: `{winner_row.get('candidate_id', winner_row['run_id'])}`",
        f"Winner mode: `{winner_row.get('mode', 'pure100')}`",
        "",
        f"- Average wins vs frozen DPVO median: `{avg_wins:.2f}` / 38",
        f"- Median wins vs frozen DPVO median: `{med_wins:.2f}` / 38",
        f"- Mean of repeat full assoc ATE: `{mean_assoc:.6f}`",
        f"- Median of repeat full assoc ATE: `{median_assoc:.6f}`",
        f"- Per-sequence median wins: `{seq_wins}` / {len(per_sequence_rows)}",
        f"- Per-sequence median losses: `{seq_losses}` / {len(per_sequence_rows)}",
        "",
    ]
    return "\n".join(lines) + "\n"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _runtime_from_manifest(manifest_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    payload = _load_yaml(manifest_path)
    eval_cfg = payload.get("sweep", {}).get("config_overrides", {}).get("eval", {}) or {}
    model_cfg = payload.get("sweep", {}).get("config_overrides", {}).get("model", {}) or {}
    runtime = {
        "stride": int((eval_cfg.get("pure100_eval_overrides", {}) or {}).get("stride", eval_cfg.get("stride", 4))),
        "backend_thresh": float((eval_cfg.get("pure100_eval_overrides", {}) or {}).get("backend_thresh", eval_cfg.get("backend_thresh", 32.0))),
        "dpvo_opts": str((eval_cfg.get("pure100_eval_overrides", {}) or {}).get("dpvo_opts", eval_cfg.get("dpvo_opts", ""))),
        "image_height": int(model_cfg.get("image_size", [240, 320])[0]),
        "image_width": int(model_cfg.get("image_size", [240, 320])[1]),
        "coverage_gate": float(eval_cfg.get("coverage_gate", 0.95)),
        "secondary_coverage_gate": (
            float(eval_cfg["secondary_coverage_gate"])
            if eval_cfg.get("secondary_coverage_gate") not in (None, "")
            else None
        ),
    }
    return payload, runtime


def _assert_idle_or_raise(*, force: bool) -> None:
    active = _gpu_heavy_process_lines(exclude_pid=os.getpid())
    if active and not force:
        raise RuntimeError(
            "Refusing to start finalist evaluation while other GPU-heavy jobs are active:\n"
            + "\n".join(active[:10])
        )


def _print_dry_run(
    *,
    candidates: list[FinalistCandidate],
    sequences: list[str],
    screening_output_root: Path,
    winner_output_root: Path,
    leaderboard_path: Path,
) -> None:
    print(f"screening_output_root: {screening_output_root}")
    print(f"winner_output_root: {winner_output_root}")
    print(f"leaderboard_path: {leaderboard_path}")
    print(f"candidate_count: {len(candidates)}")
    print("candidates:")
    for candidate in candidates:
        print(
            f"  - {candidate.run_id} (best_mode={candidate.mode}): best_assoc={candidate.best_assoc:.6f} "
            f"best_secondary_assoc={candidate.best_secondary_assoc:.6f} "
            f"checkpoint={candidate.checkpoint_path}"
        )
    print(f"sequence_count: {len(sequences)}")


def main() -> None:
    repo_root = _repo_root()
    defaults = _default_paths(repo_root)
    ap = argparse.ArgumentParser(
        description="Screen top Focus071 sweep finalists on full Freiburg1/2/3 and run a 5x winner benchmark."
    )
    ap.add_argument("--sweep-manifest", default=str(defaults["sweep_manifest"]))
    ap.add_argument("--sweep-root", default=str(defaults["sweep_root"]))
    ap.add_argument("--baseline-per-sequence", default=str(defaults["baseline_per_sequence"]))
    ap.add_argument("--dataset-root", default=str(defaults["dataset_root"]))
    ap.add_argument("--dpvo-root", default=str(defaults["dpvo_root"]))
    ap.add_argument("--dpvo-weights", default=str(defaults["dpvo_weights"]))
    ap.add_argument("--dpvo-config", default=str(defaults["dpvo_config"]))
    ap.add_argument("--screening-output-root", default=str(defaults["screening_output_root"]))
    ap.add_argument("--winner-output-root", default=str(defaults["winner_output_root"]))
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest_path = Path(args.sweep_manifest).expanduser().resolve()
    sweep_root = Path(args.sweep_root).expanduser().resolve()
    leaderboard_path = sweep_root / "leaderboard_dev.csv"
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    dpvo_root = Path(args.dpvo_root).expanduser().resolve()
    dpvo_weights = Path(args.dpvo_weights).expanduser().resolve()
    dpvo_config = Path(args.dpvo_config).expanduser().resolve()
    screening_output_root = Path(args.screening_output_root).expanduser().resolve()
    winner_output_root = Path(args.winner_output_root).expanduser().resolve()

    _assert_idle_or_raise(force=bool(args.force))

    manifest, runtime = _runtime_from_manifest(manifest_path)
    sequences = _enumerate_freiburg_sequences(dataset_root)
    candidates: list[FinalistCandidate] = []
    if leaderboard_path.exists():
        candidates = _read_leaderboard_candidates(
            leaderboard_path=leaderboard_path,
            top_k=int(args.top_k),
            coverage_gate=float(runtime["coverage_gate"]),
            secondary_coverage_gate=runtime["secondary_coverage_gate"],
        )

    if args.dry_run:
        _print_dry_run(
            candidates=candidates,
            sequences=sequences,
            screening_output_root=screening_output_root,
            winner_output_root=winner_output_root,
            leaderboard_path=leaderboard_path,
        )
        return

    baseline_assoc = _read_frozen_dpvo_baseline(
        Path(args.baseline_per_sequence).expanduser().resolve(),
        expected_sequences=sequences,
    )
    if not candidates:
        candidates = _read_leaderboard_candidates(
            leaderboard_path=leaderboard_path,
            top_k=int(args.top_k),
            coverage_gate=float(runtime["coverage_gate"]),
            secondary_coverage_gate=runtime["secondary_coverage_gate"],
        )

    screening_rows: list[dict[str, object]] = []
    pack_config_dir = screening_output_root / "pack_eval_configs"
    mode_candidates = _expand_mode_candidates(candidates)
    for candidate in mode_candidates:
        screening_run_dir = screening_output_root / "screening" / f"{candidate.run_id}__{candidate.mode}"
        if candidate.mode == "pure100":
            frontend_config = _materialize_pure_pack_config(
                candidate.config_path,
                pack_config_dir / f"{candidate.run_id}__pure100.yaml",
                f"{candidate.run_id}_tumwin_screen_pure100",
            )
            frontend_mode = "dino_proposals"
        else:
            frontend_config = candidate.config_path
            frontend_mode = "dino_hybrid"
        rows = _run_eval(
            repo_root=repo_root,
            dataset_root=dataset_root,
            dpvo_root=dpvo_root,
            dpvo_weights=dpvo_weights,
            dpvo_config=dpvo_config,
            sequences=sequences,
            output_dir=screening_run_dir,
            frontend_mode=frontend_mode,
            frontend_config=frontend_config,
            checkpoint_path=candidate.checkpoint_path,
            stride=int(runtime["stride"]),
            backend_thresh=float(runtime["backend_thresh"]),
            image_height=int(runtime["image_height"]),
            image_width=int(runtime["image_width"]),
            dpvo_opts=str(runtime["dpvo_opts"]),
        )
        _validate_eval_rows(
            rows,
            expected_sequences=sequences,
            csv_path=screening_run_dir / "dpvo_style_metrics_summary.csv",
        )
        screening_rows.append(
            _screening_summary_row(
                run_id=candidate.run_id,
                mode=candidate.mode,
                rows=rows,
                baseline_assoc=baseline_assoc,
                checkpoint_path=candidate.checkpoint_path,
                config_path=frontend_config,
                best_assoc=candidate.best_assoc,
                best_secondary_assoc=candidate.best_secondary_assoc,
            )
        )

    screening_rows.sort(key=_winner_key)
    screening_fieldnames = [
        "run_id",
        "mode",
        "candidate_id",
        "checkpoint_path",
        "config_path",
        "best_proxy_assoc",
        "best_secondary_assoc",
        "wins_vs_dpvo_median",
        "losses_vs_dpvo_median",
        "ties_vs_dpvo_median",
        "full_mean_ate_rmse",
        "full_mean_ate_rmse_associated",
        "full_mean_coverage",
        "freiburg1_wins",
        "freiburg2_wins",
        "freiburg3_wins",
        "freiburg1_losses",
        "freiburg2_losses",
        "freiburg3_losses",
    ]
    _write_csv(screening_output_root / "screening_summary.csv", screening_rows, screening_fieldnames)
    winner_row = screening_rows[0]
    _write_text(
        screening_output_root / "screening_summary.md",
        _screening_markdown(screening_rows, winner_row),
    )

    winner_candidate = next(
        candidate
        for candidate in mode_candidates
        if candidate.run_id == winner_row["run_id"] and candidate.mode == winner_row.get("mode", "pure100")
    )
    if winner_candidate.mode == "pure100":
        winner_frontend_mode = "dino_proposals"
        winner_frontend_config = _materialize_pure_pack_config(
            winner_candidate.config_path,
            winner_output_root / "pack_eval_configs" / f"{winner_candidate.run_id}__pure100.yaml",
            f"{winner_candidate.run_id}_tumwin_winner_pure100",
        )
    else:
        winner_frontend_mode = "dino_hybrid"
        winner_frontend_config = winner_candidate.config_path
    repeat_rows: list[dict[str, object]] = []
    repeat_eval_rows: list[list[dict[str, str]]] = []
    for repeat_idx in range(1, int(args.repeats) + 1):
        repeat_id = f"repeat_{repeat_idx:02d}"
        repeat_dir = winner_output_root / repeat_id
        rows = _run_eval(
            repo_root=repo_root,
            dataset_root=dataset_root,
            dpvo_root=dpvo_root,
            dpvo_weights=dpvo_weights,
            dpvo_config=dpvo_config,
            sequences=sequences,
            output_dir=repeat_dir,
            frontend_mode=winner_frontend_mode,
            frontend_config=winner_frontend_config,
            checkpoint_path=winner_candidate.checkpoint_path,
            stride=int(runtime["stride"]),
            backend_thresh=float(runtime["backend_thresh"]),
            image_height=int(runtime["image_height"]),
            image_width=int(runtime["image_width"]),
            dpvo_opts=str(runtime["dpvo_opts"]),
        )
        _validate_eval_rows(
            rows,
            expected_sequences=sequences,
            csv_path=repeat_dir / "dpvo_style_metrics_summary.csv",
        )
        repeat_eval_rows.append(rows)
        repeat_rows.append(
            _winner_repeat_summary_row(
                repeat_id=repeat_id,
                rows=rows,
                baseline_assoc=baseline_assoc,
            )
        )

    winner_repeat_fieldnames = [
        "repeat_id",
        "wins_vs_dpvo_median",
        "losses_vs_dpvo_median",
        "ties_vs_dpvo_median",
        "full_mean_ate_rmse",
        "full_mean_ate_rmse_associated",
        "full_mean_coverage",
        "freiburg1_wins",
        "freiburg2_wins",
        "freiburg3_wins",
        "freiburg1_losses",
        "freiburg2_losses",
        "freiburg3_losses",
    ]
    _write_csv(winner_output_root / "winner_repeat_summary.csv", repeat_rows, winner_repeat_fieldnames)

    per_sequence_rows = _winner_per_sequence_medians(
        repeat_rows=repeat_eval_rows,
        expected_sequences=sequences,
        baseline_assoc=baseline_assoc,
    )
    _write_csv(
        winner_output_root / "winner_per_sequence_median.csv",
        per_sequence_rows,
        [
            "sequence",
            "family",
            "baseline_dpvo_assoc_median",
            "winner_assoc_median",
            "winner_ate_median",
            "winner_coverage_median",
            "winner_vs_baseline",
        ],
    )
    _write_text(
        winner_output_root / "winner_summary.md",
        _winner_summary_markdown(
            winner_row=winner_row,
            repeat_rows=repeat_rows,
            per_sequence_rows=per_sequence_rows,
        ),
    )


if __name__ == "__main__":
    main()
