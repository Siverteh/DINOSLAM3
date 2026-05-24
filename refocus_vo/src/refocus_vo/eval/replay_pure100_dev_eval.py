from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch

from refocus_vo.dino_dpvo import load_dino_dpvo_config
from refocus_vo.dino_dpvo.frontend import build_dino_dpvo_frontend
from refocus_vo.train_dino_dpvo_frontend import _evaluate_external_tum_ate


def _write_summary_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "repeat",
                "step",
                "external_mean_ate",
                "external_mean_ate_associated",
                "external_mean_coverage",
                "diagnostics_summary_path",
                "eval_dir",
            ]
        )


def _append_summary_row(path: Path, repeat_idx: int, step: int, metrics: dict[str, float], eval_dir: Path) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                repeat_idx,
                step,
                f"{float(metrics.get('external_mean_ate', math.nan)):.12f}",
                f"{float(metrics.get('external_mean_ate_associated', math.nan)):.12f}",
                f"{float(metrics.get('external_mean_coverage', math.nan)):.12f}",
                str(eval_dir / "diagnostics_summary.csv"),
                str(eval_dir),
            ]
        )


def _write_markdown_summary(path: Path, checkpoint: Path, config: Path, repeats: list[dict[str, float]], step: int) -> None:
    assoc_values = [float(item["external_mean_ate_associated"]) for item in repeats]
    ate_values = [float(item["external_mean_ate"]) for item in repeats]
    cov_values = [float(item["external_mean_coverage"]) for item in repeats]

    def _fmt_mean(values: list[float]) -> str:
        return f"{sum(values) / len(values):.6f}" if values else "NaN"

    def _fmt_min(values: list[float]) -> str:
        return f"{min(values):.6f}" if values else "NaN"

    def _fmt_max(values: list[float]) -> str:
        return f"{max(values):.6f}" if values else "NaN"

    lines = [
        "# Pure100 Dev Replay x5",
        "",
        f"- checkpoint: `{checkpoint}`",
        f"- config: `{config}`",
        f"- checkpoint step: `{step}`",
        "",
        "| Metric | Mean | Min | Max |",
        "|---|---:|---:|---:|",
        f"| `ate_rmse_associated` | {_fmt_mean(assoc_values)} | {_fmt_min(assoc_values)} | {_fmt_max(assoc_values)} |",
        f"| `ate_rmse` | {_fmt_mean(ate_values)} | {_fmt_min(ate_values)} | {_fmt_max(ate_values)} |",
        f"| `coverage` | {_fmt_mean(cov_values)} | {_fmt_min(cov_values)} | {_fmt_max(cov_values)} |",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay the trainer's pure100 dev eval multiple times for a saved checkpoint.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tum-dataset-root", required=True)
    ap.add_argument("--dpvo-root", required=True)
    ap.add_argument("--dpvo-weights", required=True)
    ap.add_argument("--dpvo-config", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    tum_dataset_root = Path(args.tum_dataset_root).expanduser().resolve()
    dpvo_root = Path(args.dpvo_root).expanduser().resolve()
    dpvo_weights = Path(args.dpvo_weights).expanduser().resolve()
    dpvo_config = Path(args.dpvo_config).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "repeat_summary.csv"
    summary_md = output_dir / "repeat_summary.md"

    cfg = load_dino_dpvo_config(config_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    step = int(args.step if args.step is not None else payload.get("step", 0))

    pure100_sequences = [str(s) for s in cfg.eval.get("pure100_eval_sequences", cfg.eval.get("primary_eval_sequences", []))]
    pure100_frontend_mode = str(cfg.eval.get("pure100_frontend_mode", "dino_proposals"))
    pure100_model_overrides = dict(cfg.eval.get("pure100_model_overrides", {}))
    pure100_eval_overrides = dict(cfg.eval.get("pure100_eval_overrides", {}))

    _write_summary_header(summary_csv)
    repeat_metrics: list[dict[str, float]] = []

    for repeat_idx in range(1, int(args.repeats) + 1):
        repeat_dir = output_dir / f"repeat_{repeat_idx:02d}"
        repeat_dir.mkdir(parents=True, exist_ok=True)

        # Rebuild and reload the frontend for each repeat so every trial starts
        # from the exact same checkpoint state before the stochastic DPVO eval.
        cfg.raw.setdefault("training", {})["device"] = str(args.device)
        model = build_dino_dpvo_frontend(cfg)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()

        metrics = _evaluate_external_tum_ate(
            model,
            cfg,
            dataset_root=tum_dataset_root,
            dpvo_root=dpvo_root,
            dpvo_weights=dpvo_weights,
            dpvo_config=dpvo_config,
            output_dir=repeat_dir,
            step=step,
            sequences=pure100_sequences,
            frontend_mode=pure100_frontend_mode,
            run_tag="pure100",
            model_overrides=pure100_model_overrides,
            eval_overrides=pure100_eval_overrides,
        )
        repeat_metrics.append(metrics)
        _append_summary_row(
            summary_csv,
            repeat_idx=repeat_idx,
            step=step,
            metrics=metrics,
            eval_dir=repeat_dir / "dev_eval" / f"step_{step:06d}" / "pure100",
        )
        print(
            f"[repeat {repeat_idx}/{args.repeats}] "
            f"assoc={float(metrics.get('external_mean_ate_associated', math.nan)):.6f} "
            f"ate={float(metrics.get('external_mean_ate', math.nan)):.6f} "
            f"cov={float(metrics.get('external_mean_coverage', math.nan)):.6f}"
        )

    _write_markdown_summary(summary_md, checkpoint=checkpoint_path, config=config_path, repeats=repeat_metrics, step=step)


if __name__ == "__main__":
    main()
