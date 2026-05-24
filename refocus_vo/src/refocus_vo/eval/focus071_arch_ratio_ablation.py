from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from refocus_vo.eval.focus071_tumwin_finalists import (
    FinalistCandidate,
    _assert_idle_or_raise,
    _read_csv_rows,
    _read_frozen_dpvo_baseline,
    _run_eval,
    _runtime_from_manifest,
    _screening_summary_row,
    _validate_eval_rows,
    _write_csv,
    _write_text,
)
from refocus_vo.eval.focus071_vs_dpvo_tum_freiburg123_5x import _enumerate_freiburg_sequences
from refocus_vo.sweeps.run_assoc9_sweep import _load_yaml, _materialize_pure_pack_config


@dataclass(frozen=True)
class RatioSpec:
    ratio_id: str
    frontend_mode: str
    native_fraction: float
    dino_fraction: float


@dataclass(frozen=True)
class RatioCandidate:
    run_id: str
    source_mode: str
    checkpoint_path: Path
    config_path: Path
    best_assoc: float
    best_secondary_assoc: float


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_paths(repo_root: Path) -> dict[str, Path]:
    subtree_root = repo_root / "refocus_vo"
    return {
        "dataset_root": repo_root / "src" / "dino_slam3" / "data" / "tum_rgbd",
        "dpvo_root": subtree_root / "external" / "repos" / "DPVO",
        "dpvo_weights": subtree_root / "external" / "repos" / "DPVO" / "dpvo.pth",
        "dpvo_config": subtree_root / "external" / "repos" / "DPVO" / "config" / "default.yaml",
        "sweep_manifest": subtree_root / "configs" / "dino_dpvo_focus071_arch5x2_tumwin_sweep_v1.yaml",
        "sweep_root": subtree_root / "runs" / "sweeps" / "dino_dpvo_focus071_arch5x2_tumwin_sweep_v1_rerun1",
        "baseline_per_sequence": subtree_root / "runs" / "eval" / "tum_rgbd_freiburg123_dpvo_vs_focus071_v1_rerun6" / "summary" / "per_sequence_median.csv",
        "output_root": subtree_root / "runs" / "eval" / "tum_rgbd_freiburg123_arch_ratio_ablation_v1",
    }


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def _coverage_ok(row: dict[str, str], *, coverage_gate: float, secondary_coverage_gate: float | None) -> bool:
    best_coverage = _safe_float(row.get("best_coverage"))
    best_lowtex_coverage = _safe_float(row.get("best_lowtex_coverage"))
    if not (math.isfinite(best_coverage) and best_coverage >= float(coverage_gate)):
        return False
    if secondary_coverage_gate is None:
        return True
    return math.isfinite(best_lowtex_coverage) and best_lowtex_coverage >= float(secondary_coverage_gate)


def _load_ranked_candidates(
    *,
    leaderboard_path: Path,
    coverage_gate: float,
    secondary_coverage_gate: float | None,
    overall_top_k: int,
    include_lowtex_specialist: bool,
) -> list[RatioCandidate]:
    rows = _read_csv_rows(leaderboard_path)
    eligible = []
    for row in rows:
        status = str(row.get("status", "")).strip().lower()
        if status not in {"completed", "early_stopped"}:
            continue
        if not _coverage_ok(
            row,
            coverage_gate=coverage_gate,
            secondary_coverage_gate=secondary_coverage_gate,
        ):
            continue
        checkpoint_path = Path(str(row.get("checkpoint_path", "")).strip()).expanduser().resolve()
        config_path = Path(str(row.get("config_path", "")).strip()).expanduser().resolve()
        if not checkpoint_path.exists() or not config_path.exists():
            continue
        candidate = RatioCandidate(
            run_id=str(row.get("run_id", "")).strip(),
            source_mode=str(row.get("best_mode", "hybrid")).strip() or "hybrid",
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            best_assoc=_safe_float(row.get("best_assoc")),
            best_secondary_assoc=_safe_float(row.get("best_lowtex_assoc")),
        )
        eligible.append(candidate)
    if not eligible:
        raise RuntimeError(f"No eligible candidates found in {leaderboard_path}")

    overall_sorted = sorted(
        eligible,
        key=lambda item: (
            item.best_assoc if math.isfinite(item.best_assoc) else math.inf,
            item.best_secondary_assoc if math.isfinite(item.best_secondary_assoc) else math.inf,
            item.run_id,
        ),
    )
    selected: list[RatioCandidate] = []
    seen: set[str] = set()
    for candidate in overall_sorted[: int(overall_top_k)]:
        selected.append(candidate)
        seen.add(candidate.run_id)

    if include_lowtex_specialist:
        lowtex_sorted = sorted(
            eligible,
            key=lambda item: (
                item.best_secondary_assoc if math.isfinite(item.best_secondary_assoc) else math.inf,
                item.best_assoc if math.isfinite(item.best_assoc) else math.inf,
                item.run_id,
            ),
        )
        for candidate in lowtex_sorted:
            if candidate.run_id in seen:
                continue
            selected.append(candidate)
            seen.add(candidate.run_id)
            break

    return selected


def _parse_ratio_specs(text: str) -> list[RatioSpec]:
    specs: list[RatioSpec] = []
    for raw_item in [part.strip() for part in str(text).split(",") if part.strip()]:
        item = raw_item.lower()
        if item in {"pure", "pure100", "0/100", "0:100"}:
            specs.append(
                RatioSpec(
                    ratio_id="pure100",
                    frontend_mode="dino_proposals",
                    native_fraction=0.0,
                    dino_fraction=1.0,
                )
            )
            continue
        normalized = item.replace(":", "/")
        left, right = normalized.split("/", 1)
        native_fraction = float(left) / 100.0
        dino_fraction = float(right) / 100.0
        specs.append(
            RatioSpec(
                ratio_id=f"hybrid{int(round(native_fraction * 100)):02d}_{int(round(dino_fraction * 100)):02d}",
                frontend_mode="dino_hybrid",
                native_fraction=native_fraction,
                dino_fraction=dino_fraction,
            )
        )
    if not specs:
        raise ValueError("No ratio specs were parsed")
    deduped: list[RatioSpec] = []
    seen: set[str] = set()
    for spec in specs:
        if spec.ratio_id in seen:
            continue
        deduped.append(spec)
        seen.add(spec.ratio_id)
    return deduped


def _materialize_ratio_config(
    *,
    source_config: Path,
    output_config: Path,
    run_label: str,
    native_fraction: float,
    dino_fraction: float,
) -> Path:
    payload = _load_yaml(source_config)
    payload.setdefault("model", {})
    payload.setdefault("eval", {})
    payload["model"]["native_fraction"] = float(native_fraction)
    payload["model"]["dino_fraction"] = float(dino_fraction)
    payload["method_id"] = f"{payload.get('method_id', 'dino_dpvo')}_{run_label}"
    payload["feature_type"] = f"{payload.get('feature_type', 'DINO_DPVO')}_{run_label.upper()}"
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_config


def _ratio_result_key(row: dict[str, object]) -> tuple[float, ...]:
    return (
        -float(row.get("wins_vs_dpvo_median", 0)),
        float(row.get("full_mean_ate_rmse_associated", math.inf)),
        -float(row.get("freiburg2_wins", 0)),
        -float(row.get("freiburg3_wins", 0)),
        -float(row.get("dino_fraction", 0.0)),
    )


def _add_ratio_deltas(rows: list[dict[str, object]]) -> None:
    by_run: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        by_run.setdefault(str(row["run_id"]), {})[str(row["ratio_id"])] = row
    for run_rows in by_run.values():
        baseline = run_rows.get("hybrid90_10")
        baseline_assoc = (
            float(baseline["full_mean_ate_rmse_associated"])
            if baseline is not None and math.isfinite(float(baseline["full_mean_ate_rmse_associated"]))
            else math.nan
        )
        baseline_wins = float(baseline["wins_vs_dpvo_median"]) if baseline is not None else math.nan
        for row in run_rows.values():
            assoc = float(row["full_mean_ate_rmse_associated"])
            wins = float(row["wins_vs_dpvo_median"])
            row["delta_assoc_vs_90_10"] = (
                f"{assoc - baseline_assoc:+.6f}" if math.isfinite(assoc) and math.isfinite(baseline_assoc) else ""
            )
            row["delta_wins_vs_90_10"] = (
                f"{wins - baseline_wins:+.0f}" if math.isfinite(wins) and math.isfinite(baseline_wins) else ""
            )


def _best_ratio_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_run: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_run.setdefault(str(row["run_id"]), []).append(row)
    winners: list[dict[str, object]] = []
    for run_id, run_rows in by_run.items():
        ranked = sorted(run_rows, key=_ratio_result_key)
        winner = dict(ranked[0])
        winner["run_id"] = run_id
        winners.append(winner)
    winners.sort(key=_ratio_result_key)
    return winners


def _summary_markdown(
    *,
    candidates: list[RatioCandidate],
    ratios: list[RatioSpec],
    rows: list[dict[str, object]],
    best_rows: list[dict[str, object]],
) -> str:
    lines = [
        "# Focus071 Architecture Ratio Ablation",
        "",
        "Candidates screened:",
    ]
    for candidate in candidates:
        lines.append(
            f"- `{candidate.run_id}` from `{candidate.source_mode}` checkpoint "
            f"(proxy `{candidate.best_assoc:.6f}`, Freiburg2-pressure `{candidate.best_secondary_assoc:.6f}`)"
        )
    lines.extend(
        [
            "",
            "Ratios:",
            "",
        ]
    )
    for ratio in ratios:
        lines.append(
            f"- `{ratio.ratio_id}`: mode `{ratio.frontend_mode}`, native `{ratio.native_fraction:.2f}`, dino `{ratio.dino_fraction:.2f}`"
        )
    lines.extend(
        [
            "",
            "| Run | Best ratio | Wins vs DPVO median | Mean assoc ATE | Freiburg2 wins | Freiburg3 wins | Delta assoc vs 90/10 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in best_rows:
        lines.append(
            f"| `{row['run_id']}` | `{row['ratio_id']}` | {row['wins_vs_dpvo_median']} | "
            f"{row['full_mean_ate_rmse_associated']} | {row['freiburg2_wins']} | {row['freiburg3_wins']} | {row.get('delta_assoc_vs_90_10', '')} |"
        )
    best_overall = sorted(rows, key=_ratio_result_key)[0]
    lines.extend(
        [
            "",
            f"Best overall screen: `{best_overall['run_id']}:{best_overall['ratio_id']}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _print_dry_run(*, candidates: list[RatioCandidate], ratios: list[RatioSpec], output_root: Path) -> None:
    print(f"output_root: {output_root}")
    print("candidates:")
    for candidate in candidates:
        print(
            f"  - {candidate.run_id}: source_mode={candidate.source_mode} "
            f"proxy={candidate.best_assoc:.6f} lowtex={candidate.best_secondary_assoc:.6f} "
            f"checkpoint={candidate.checkpoint_path}"
        )
    print("ratios:")
    for ratio in ratios:
        print(
            f"  - {ratio.ratio_id}: frontend_mode={ratio.frontend_mode} "
            f"native={ratio.native_fraction:.2f} dino={ratio.dino_fraction:.2f}"
        )


def main() -> None:
    repo_root = _repo_root()
    defaults = _default_paths(repo_root)
    ap = argparse.ArgumentParser(
        description="Run a full Freiburg1/2/3 ratio ablation on the best Focus071 architecture sweep checkpoints."
    )
    ap.add_argument("--sweep-manifest", default=str(defaults["sweep_manifest"]))
    ap.add_argument("--sweep-root", default=str(defaults["sweep_root"]))
    ap.add_argument("--baseline-per-sequence", default=str(defaults["baseline_per_sequence"]))
    ap.add_argument("--dataset-root", default=str(defaults["dataset_root"]))
    ap.add_argument("--dpvo-root", default=str(defaults["dpvo_root"]))
    ap.add_argument("--dpvo-weights", default=str(defaults["dpvo_weights"]))
    ap.add_argument("--dpvo-config", default=str(defaults["dpvo_config"]))
    ap.add_argument("--output-root", default=str(defaults["output_root"]))
    ap.add_argument("--ratios", default="90/10,75/25,50/50,25/75,pure100")
    ap.add_argument("--overall-top-k", type=int, default=2)
    ap.add_argument("--include-lowtex-specialist", action="store_true", default=True)
    ap.add_argument("--no-lowtex-specialist", dest="include_lowtex_specialist", action="store_false")
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
    output_root = Path(args.output_root).expanduser().resolve()

    _assert_idle_or_raise(force=bool(args.force))

    _manifest, runtime = _runtime_from_manifest(manifest_path)
    sequences = _enumerate_freiburg_sequences(dataset_root)
    baseline_assoc = _read_frozen_dpvo_baseline(
        Path(args.baseline_per_sequence).expanduser().resolve(),
        expected_sequences=sequences,
    )
    candidates = _load_ranked_candidates(
        leaderboard_path=leaderboard_path,
        coverage_gate=float(runtime["coverage_gate"]),
        secondary_coverage_gate=runtime["secondary_coverage_gate"],
        overall_top_k=int(args.overall_top_k),
        include_lowtex_specialist=bool(args.include_lowtex_specialist),
    )
    ratios = _parse_ratio_specs(str(args.ratios))

    if args.dry_run:
        _print_dry_run(candidates=candidates, ratios=ratios, output_root=output_root)
        return

    config_root = output_root / "ratio_eval_configs"
    screening_rows: list[dict[str, object]] = []

    for candidate in candidates:
        for ratio in ratios:
            eval_dir = output_root / "screening" / candidate.run_id / ratio.ratio_id
            if ratio.frontend_mode == "dino_proposals":
                frontend_config = _materialize_pure_pack_config(
                    candidate.config_path,
                    config_root / f"{candidate.run_id}__{ratio.ratio_id}.yaml",
                    f"{candidate.run_id}_{ratio.ratio_id}",
                )
            else:
                frontend_config = _materialize_ratio_config(
                    source_config=candidate.config_path,
                    output_config=config_root / f"{candidate.run_id}__{ratio.ratio_id}.yaml",
                    run_label=f"{candidate.run_id}_{ratio.ratio_id}",
                    native_fraction=ratio.native_fraction,
                    dino_fraction=ratio.dino_fraction,
                )
            rows = _run_eval(
                repo_root=repo_root,
                dataset_root=dataset_root,
                dpvo_root=dpvo_root,
                dpvo_weights=dpvo_weights,
                dpvo_config=dpvo_config,
                sequences=sequences,
                output_dir=eval_dir,
                frontend_mode=ratio.frontend_mode,
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
                csv_path=eval_dir / "dpvo_style_metrics_summary.csv",
            )
            summary = _screening_summary_row(
                run_id=candidate.run_id,
                mode=ratio.ratio_id,
                rows=rows,
                baseline_assoc=baseline_assoc,
                checkpoint_path=candidate.checkpoint_path,
                config_path=frontend_config,
                best_assoc=candidate.best_assoc,
                best_secondary_assoc=candidate.best_secondary_assoc,
            )
            summary["ratio_id"] = ratio.ratio_id
            summary["frontend_mode"] = ratio.frontend_mode
            summary["native_fraction"] = f"{ratio.native_fraction:.2f}"
            summary["dino_fraction"] = f"{ratio.dino_fraction:.2f}"
            summary["checkpoint_source_mode"] = candidate.source_mode
            screening_rows.append(summary)

    _add_ratio_deltas(screening_rows)
    screening_rows.sort(key=_ratio_result_key)
    fieldnames = [
        "run_id",
        "mode",
        "ratio_id",
        "frontend_mode",
        "native_fraction",
        "dino_fraction",
        "checkpoint_source_mode",
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
        "delta_assoc_vs_90_10",
        "delta_wins_vs_90_10",
    ]
    _write_csv(output_root / "screening_summary.csv", screening_rows, fieldnames)

    best_rows = _best_ratio_rows(screening_rows)
    _write_csv(output_root / "best_ratio_per_run.csv", best_rows, fieldnames)
    _write_text(
        output_root / "screening_summary.md",
        _summary_markdown(
            candidates=candidates,
            ratios=ratios,
            rows=screening_rows,
            best_rows=best_rows,
        ),
    )


if __name__ == "__main__":
    main()
