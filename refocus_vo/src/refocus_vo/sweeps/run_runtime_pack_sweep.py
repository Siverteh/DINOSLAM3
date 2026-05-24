from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SUBTREE_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = SUBTREE_ROOT.parent


@dataclass
class CheckpointSpec:
    checkpoint_family: str
    checkpoint: Path
    frontend_config: Path
    imported_baseline_dpvo_style_csv: Path


@dataclass
class RuntimeProfile:
    runtime_profile: str
    imported: bool
    dpvo_stride: int
    dpvo_backend_thresh: float
    dpvo_opts: str


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object at {path}")
    return payload


def _load_manifest(path: Path) -> tuple[dict[str, Any], list[CheckpointSpec], list[RuntimeProfile]]:
    payload = _load_yaml(path)
    checkpoints = [
        CheckpointSpec(
            checkpoint_family=str(item["checkpoint_family"]),
            checkpoint=_resolve_path(item["checkpoint"], REPO_ROOT),
            frontend_config=_resolve_path(item["frontend_config"], REPO_ROOT),
            imported_baseline_dpvo_style_csv=_resolve_path(item["imported_baseline_dpvo_style_csv"], REPO_ROOT),
        )
        for item in payload.get("checkpoints", [])
    ]
    profiles = [
        RuntimeProfile(
            runtime_profile=str(item["runtime_profile"]),
            imported=bool(item.get("imported", False)),
            dpvo_stride=int(item["dpvo_stride"]),
            dpvo_backend_thresh=float(item["dpvo_backend_thresh"]),
            dpvo_opts=str(item["dpvo_opts"]),
        )
        for item in payload.get("profiles", [])
    ]
    return payload, checkpoints, profiles


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = []
    for row in rows:
        try:
            values.append(float(row[key]))
        except Exception:
            continue
    return sum(values) / len(values) if values else math.nan


def _candidate_key(row: dict[str, Any], coverage_gate: float) -> tuple[float, float, float]:
    assoc = float(row["pack_assoc"])
    ate = float(row["pack_ate"])
    coverage = float(row["pack_coverage"])
    if coverage < float(coverage_gate):
        return (math.inf, math.inf, -coverage)
    return (assoc, ate, -coverage)


def _pid_is_live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _wait_for_previous_sweep(wait_cfg: dict[str, Any]) -> None:
    runner_pid_file = wait_cfg.get("runner_pid_file")
    patterns = [str(item) for item in wait_cfg.get("process_patterns", []) if str(item).strip()]
    runner_pid_path = _resolve_path(runner_pid_file, REPO_ROOT) if runner_pid_file else None

    while True:
        live = False
        if runner_pid_path and runner_pid_path.exists():
            try:
                runner_pid = int(runner_pid_path.read_text(encoding="utf-8").strip())
            except Exception:
                runner_pid = 0
            if runner_pid and _pid_is_live(runner_pid):
                live = True
        for pattern in patterns:
            proc = subprocess.run(
                ["pgrep", "-af", pattern],
                check=False,
                capture_output=True,
                text=True,
            )
            lines = [line for line in proc.stdout.splitlines() if line.strip()]
            if lines:
                live = True
                break
        if not live:
            return
        time.sleep(10)


def _run_pack_eval(
    *,
    pack_script: Path,
    base_output_dir: Path,
    sequences: list[str],
    checkpoint: Path,
    frontend_config: Path,
    candidate_id: str,
    profile: RuntimeProfile,
) -> tuple[Path, Path]:
    run_id = f"{candidate_id}_eval"
    pack_id = f"{candidate_id}_pack"
    env = os.environ.copy()
    env.update(
        {
            "RUNS_ROOT": str(base_output_dir),
            "SEQUENCES": ",".join(sequences),
            "CHECKPOINT": str(checkpoint),
            "FRONTEND_CONFIG": str(frontend_config),
            "FRONTEND_MODE": "dino_proposals",
            "PACK_ID": pack_id,
            "DINO_DPVO_RUN_ID": run_id,
            "RUN_ORB_BASELINE": "0",
            "RUN_DPVO_BASELINE": "0",
            "DPVO_STRIDE": str(profile.dpvo_stride),
            "DPVO_BACKEND_THRESH": f"{profile.dpvo_backend_thresh}",
            "DPVO_OPTS": profile.dpvo_opts,
        }
    )
    subprocess.run(["bash", str(pack_script)], check=True, cwd=str(REPO_ROOT), env=env)
    eval_csv = base_output_dir / "eval" / run_id / "dpvo_style_metrics_summary.csv"
    metrics_csv = base_output_dir / "eval" / run_id / "metrics_summary.csv"
    if not eval_csv.exists():
        raise FileNotFoundError(f"Missing evaluation CSV: {eval_csv}")
    return eval_csv, metrics_csv


def _import_or_run_candidates(
    *,
    manifest: dict[str, Any],
    checkpoints: list[CheckpointSpec],
    profiles: list[RuntimeProfile],
    base_output_dir: Path,
    dry_run: bool,
) -> list[dict[str, Any]]:
    pack_cfg = manifest["pack"]
    pack_script = _resolve_path(pack_cfg["script"], REPO_ROOT)
    sequences = [str(seq) for seq in pack_cfg["sequences"]]
    entries: list[dict[str, Any]] = []
    for ckpt in checkpoints:
        family_slug = ckpt.checkpoint_family
        for profile in profiles:
            candidate_id = f"{family_slug}__{profile.runtime_profile}"
            run_eval_csv: Path | None = None
            metrics_csv: Path | None = None
            source = "imported_baseline" if profile.imported else "fresh_eval"
            if dry_run:
                entries.append(
                    {
                        "candidate_id": candidate_id,
                        "checkpoint_family": family_slug,
                        "runtime_profile": profile.runtime_profile,
                        "dpvo_stride": str(profile.dpvo_stride),
                        "dpvo_backend_thresh": f"{profile.dpvo_backend_thresh}",
                        "dpvo_opts": profile.dpvo_opts,
                        "source": source,
                        "checkpoint": str(ckpt.checkpoint),
                        "config_path": str(ckpt.frontend_config),
                        "csv_path": str(ckpt.imported_baseline_dpvo_style_csv if profile.imported else (base_output_dir / "eval" / f"{candidate_id}_eval" / "dpvo_style_metrics_summary.csv")),
                        "metrics_csv_path": str(ckpt.imported_baseline_dpvo_style_csv if profile.imported else (base_output_dir / "eval" / f"{candidate_id}_eval" / "metrics_summary.csv")),
                    }
                )
                continue
            if profile.imported:
                run_eval_csv = ckpt.imported_baseline_dpvo_style_csv
                metrics_csv = ckpt.imported_baseline_dpvo_style_csv
            else:
                run_eval_csv, metrics_csv = _run_pack_eval(
                    pack_script=pack_script,
                    base_output_dir=base_output_dir,
                    sequences=sequences,
                    checkpoint=ckpt.checkpoint,
                    frontend_config=ckpt.frontend_config,
                    candidate_id=candidate_id,
                    profile=profile,
                )
            rows = _read_csv_rows(run_eval_csv)
            entries.append(
                {
                    "candidate_id": candidate_id,
                    "checkpoint_family": family_slug,
                    "runtime_profile": profile.runtime_profile,
                    "dpvo_stride": str(profile.dpvo_stride),
                    "dpvo_backend_thresh": f"{profile.dpvo_backend_thresh}",
                    "dpvo_opts": profile.dpvo_opts,
                    "source": source,
                    "checkpoint": str(ckpt.checkpoint),
                    "config_path": str(ckpt.frontend_config),
                    "csv_path": str(run_eval_csv),
                    "metrics_csv_path": str(metrics_csv) if metrics_csv else "",
                    "pack_assoc": f"{_mean_metric(rows, 'ate_rmse_associated'):.6f}",
                    "pack_ate": f"{_mean_metric(rows, 'ate_rmse'):.6f}",
                    "pack_coverage": f"{_mean_metric(rows, 'coverage'):.6f}",
                }
            )
    return entries


def _write_leaderboard(path: Path, rows: list[dict[str, Any]], coverage_gate: float) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda row: _candidate_key(row, coverage_gate))
    fieldnames = [
        "candidate_id",
        "checkpoint_family",
        "runtime_profile",
        "dpvo_stride",
        "dpvo_backend_thresh",
        "dpvo_opts",
        "source",
        "checkpoint",
        "config_path",
        "csv_path",
        "pack_assoc",
        "pack_ate",
        "pack_coverage",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return sorted_rows


def _write_per_sequence_assoc(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "candidate_id",
        "checkpoint_family",
        "runtime_profile",
        "sequence",
        "assoc",
        "ate",
        "coverage",
        "dpvo_stride",
        "dpvo_backend_thresh",
        "dpvo_opts",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_rows = _read_csv_rows(Path(row["csv_path"]))
            for seq_row in csv_rows:
                writer.writerow(
                    {
                        "candidate_id": row["candidate_id"],
                        "checkpoint_family": row["checkpoint_family"],
                        "runtime_profile": row["runtime_profile"],
                        "sequence": seq_row["sequence"],
                        "assoc": seq_row["ate_rmse_associated"],
                        "ate": seq_row["ate_rmse"],
                        "coverage": seq_row["coverage"],
                        "dpvo_stride": row["dpvo_stride"],
                        "dpvo_backend_thresh": row["dpvo_backend_thresh"],
                        "dpvo_opts": row["dpvo_opts"],
                    }
                )


def _winner_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    winner = rows[0]
    broad = next(row for row in rows if row["checkpoint_family"] == "broad_canonical_winner" and row["runtime_profile"] == "fixed96_baseline")
    trained = next(row for row in rows if row["checkpoint_family"] == "best_new_trained_run" and row["runtime_profile"] == "fixed96_baseline")
    winner_assoc = float(winner["pack_assoc"])
    broad_assoc = float(broad["pack_assoc"])
    trained_assoc = float(trained["pack_assoc"])
    note = (
        "Winner uses aggressive runtime stride != 4, so this is an ATE-optimized runtime result rather than a paper-protocol-matched result."
        if int(winner["dpvo_stride"]) != 4
        else "Winner keeps stride=4, so it remains closer to the paper-style runtime protocol."
    )
    path.write_text(
        "\n".join(
            [
                "# Freiburg1 Runtime Sweep Winner",
                "",
                f"- Winner: `{winner['candidate_id']}`",
                f"- Family: `{winner['checkpoint_family']}`",
                f"- Runtime profile: `{winner['runtime_profile']}`",
                f"- Pack assoc: `{winner_assoc:.6f}`",
                f"- Pack ATE: `{float(winner['pack_ate']):.6f}`",
                f"- Pack coverage: `{float(winner['pack_coverage']):.6f}`",
                f"- Delta vs broad canonical baseline `0.087841`: `{winner_assoc - broad_assoc:+.6f}`",
                f"- Delta vs best new-trained baseline `0.089047`: `{winner_assoc - trained_assoc:+.6f}`",
                f"- Runtime stride: `{winner['dpvo_stride']}`",
                f"- Backend thresh: `{winner['dpvo_backend_thresh']}`",
                f"- DPVO opts: `{winner['dpvo_opts']}`",
                "",
                note,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _paper_vs_runtime(path: Path, winner: dict[str, Any], paper_metrics: dict[str, Any]) -> None:
    rows = _read_csv_rows(Path(winner["csv_path"]))
    seq_to_assoc = {row["sequence"]: float(row["ate_rmse_associated"]) for row in rows}
    header = "| Sequence | Winner | DPVO Default | DPVO Fast | DROID-VO |"
    divider = "|---|---:|---:|---:|---:|"
    lines = [
        "# Winner vs Paper Baselines",
        "",
        f"- Winner: `{winner['candidate_id']}`",
        f"- Runtime profile: `{winner['runtime_profile']}`",
        "",
        header,
        divider,
    ]
    for seq in sorted(seq_to_assoc):
        lines.append(
            "| {seq} | {winner_val:.6f} | {default_val:.3f} | {fast_val:.3f} | {droid_val:.3f} |".format(
                seq=seq,
                winner_val=seq_to_assoc[seq],
                default_val=float(paper_metrics["dpvo_default"][seq]),
                fast_val=float(paper_metrics["dpvo_fast"][seq]),
                droid_val=float(paper_metrics["droid_vo"][seq]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_lowtex_followup(
    *,
    manifest: dict[str, Any],
    base_output_dir: Path,
    winner: dict[str, Any],
) -> None:
    pack_cfg = manifest["pack"]
    lowtex_script = _resolve_path(pack_cfg["lowtex_script"], REPO_ROOT)
    lowtex_baseline_csv = _resolve_path(pack_cfg["lowtex_baseline_csv"], REPO_ROOT)
    winner_id = winner["candidate_id"]
    env = os.environ.copy()
    env.update(
        {
            "RUNS_ROOT": str(base_output_dir),
            "CHECKPOINT": winner["checkpoint"],
            "FRONTEND_CONFIG": winner["config_path"],
            "FRONTEND_MODE": "dino_proposals",
            "PACK_ID": f"{winner_id}_lowtex_pack",
            "DINO_DPVO_RUN_ID": f"{winner_id}_lowtex_eval",
            "RUN_ORB_BASELINE": "0",
            "RUN_DPVO_BASELINE": "0",
            "DPVO_STRIDE": winner["dpvo_stride"],
            "DPVO_BACKEND_THRESH": winner["dpvo_backend_thresh"],
            "DPVO_OPTS": winner["dpvo_opts"],
        }
    )
    subprocess.run(["bash", str(lowtex_script)], check=True, cwd=str(REPO_ROOT), env=env)
    new_csv = base_output_dir / "eval" / f"{winner_id}_lowtex_eval" / "metrics_summary.csv"
    if not (lowtex_baseline_csv.exists() and new_csv.exists()):
        return
    baseline_rows = {row["sequence"]: row for row in _read_csv_rows(lowtex_baseline_csv)}
    new_rows = {row["sequence"]: row for row in _read_csv_rows(new_csv)}
    compare_csv = base_output_dir / "lowtex_winner_comparison.csv"
    compare_md = base_output_dir / "lowtex_winner_comparison.md"
    fieldnames = [
        "sequence",
        "baseline_assoc",
        "winner_assoc",
        "delta_assoc",
        "baseline_ate",
        "winner_ate",
        "delta_ate",
    ]
    with compare_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seq in sorted(set(baseline_rows) & set(new_rows)):
            base_assoc = float(baseline_rows[seq]["ate_rmse_associated"])
            new_assoc = float(new_rows[seq]["ate_rmse_associated"])
            base_ate = float(baseline_rows[seq]["ate_rmse"])
            new_ate = float(new_rows[seq]["ate_rmse"])
            writer.writerow(
                {
                    "sequence": seq,
                    "baseline_assoc": f"{base_assoc:.6f}",
                    "winner_assoc": f"{new_assoc:.6f}",
                    "delta_assoc": f"{new_assoc - base_assoc:+.6f}",
                    "baseline_ate": f"{base_ate:.6f}",
                    "winner_ate": f"{new_ate:.6f}",
                    "delta_ate": f"{new_ate - base_ate:+.6f}",
                }
            )
    seqs = sorted(set(baseline_rows) & set(new_rows))
    base_assoc_mean = sum(float(baseline_rows[seq]["ate_rmse_associated"]) for seq in seqs) / len(seqs)
    new_assoc_mean = sum(float(new_rows[seq]["ate_rmse_associated"]) for seq in seqs) / len(seqs)
    compare_md.write_text(
        "\n".join(
            [
                "# Low-Texture Winner Comparison",
                "",
                f"- Winner: `{winner_id}`",
                f"- Baseline assoc mean: `{base_assoc_mean:.6f}`",
                f"- Winner assoc mean: `{new_assoc_mean:.6f}`",
                f"- Delta assoc mean: `{new_assoc_mean - base_assoc_mean:+.6f}`",
                "",
                f"Per-sequence CSV: `{compare_csv}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base-output-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    manifest_path = _resolve_path(args.manifest, REPO_ROOT)
    base_output_dir = _resolve_path(args.base_output_dir, REPO_ROOT)
    manifest, checkpoints, profiles = _load_manifest(manifest_path)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    (base_output_dir / "manifest.yaml").write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")

    if args.dry_run:
        rows = _import_or_run_candidates(
            manifest=manifest,
            checkpoints=checkpoints,
            profiles=profiles,
            base_output_dir=base_output_dir,
            dry_run=True,
        )
        for row in rows:
            print(
                "\t".join(
                    [
                        row["candidate_id"],
                        row["checkpoint_family"],
                        row["runtime_profile"],
                        row["source"],
                        row["dpvo_stride"],
                        row["dpvo_backend_thresh"],
                        row["dpvo_opts"],
                    ]
                )
            )
        return 0

    _wait_for_previous_sweep(manifest.get("wait", {}))
    rows = _import_or_run_candidates(
        manifest=manifest,
        checkpoints=checkpoints,
        profiles=profiles,
        base_output_dir=base_output_dir,
        dry_run=False,
    )
    coverage_gate = float(manifest["pack"].get("coverage_gate", 0.95))
    leaderboard = _write_leaderboard(base_output_dir / "leaderboard_pack.csv", rows, coverage_gate)
    _write_per_sequence_assoc(base_output_dir / "per_sequence_assoc.csv", leaderboard)
    _winner_summary(base_output_dir / "winner_summary.md", leaderboard)
    _paper_vs_runtime(base_output_dir / "paper_vs_runtime_comparison.md", leaderboard[0], manifest["paper_metrics"])
    _run_lowtex_followup(manifest=manifest, base_output_dir=base_output_dir, winner=leaderboard[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
