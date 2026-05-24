from __future__ import annotations

import argparse
import csv
import json
import platform
import shlex
import statistics
import time
from pathlib import Path

import torch
import yaml

from refocus_vo.eval.external_dpvo import _load_frontend, _sequence_image_files, evaluate_sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQUENCE = "freiburg1_desk"
DATASET_ROOT = ROOT / "src/dino_slam3/data/tum_rgbd"
DPVO_ROOT = ROOT / "refocus_vo/external/repos/DPVO"
DPVO_WEIGHTS = DPVO_ROOT / "dpvo.pth"
DPVO_CONFIG = DPVO_ROOT / "config/default.yaml"
FRONTEND_CONFIG = (
    ROOT
    / "refocus_vo/runs/eval/tum_rgbd_freiburg123_arch_ratio_ablation_v1/ratio_eval_configs/multiscale_32x4_v1__hybrid50_50.yaml"
)
FRONTEND_CHECKPOINT = (
    ROOT
    / "refocus_vo/runs/sweeps/dino_dpvo_focus071_arch5x2_tumwin_sweep_v1_rerun1/train/multiscale_32x4_v1/best_hybrid.pt"
)
TARGET_HEIGHT = 240
TARGET_WIDTH = 320
STRIDE = 4
BACKEND_THRESH = 32.0
MAX_DT = 0.02
MISSING_PENALTY_M = 3.0
MIN_COVERAGE_OK = 0.95


def _sequence_dir(sequence: str) -> Path:
    full = sequence if sequence.startswith("rgbd_dataset_") else f"rgbd_dataset_{sequence}"
    return DATASET_ROOT / full


def _load_dpvo_opts() -> list[str]:
    payload = yaml.safe_load(FRONTEND_CONFIG.read_text(encoding="utf-8"))
    raw_opts = str(payload.get("eval", {}).get("dpvo_opts", "")).strip()
    opts: list[str] = []
    for item in shlex.split(raw_opts):
        if "=" in item:
            key, value = item.split("=", 1)
            opts.extend([key, value])
        else:
            opts.append(item)
    return opts


def _metric(metrics: dict | None, *keys: str) -> float:
    if not metrics:
        return float("nan")
    for key in keys:
        if key in metrics:
            return float(metrics[key])
    return float("nan")


def run_once(
    method: str,
    sequence: str,
    output_dir: Path,
    dpvo_opts: list[str],
    frontend_cfg=None,
    frontend=None,
) -> tuple[str, float, float, float, float, float, float, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_dir = _sequence_dir(sequence)
    frames = len(_sequence_image_files(seq_dir, stride=STRIDE))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        alloc_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
    else:
        alloc_before = 0
        reserved_before = 0
    start = time.perf_counter()
    status, metrics = evaluate_sequence(
        dataset_root=DATASET_ROOT,
        sequence=sequence,
        dpvo_root=DPVO_ROOT,
        weights=DPVO_WEIGHTS,
        config_path=DPVO_CONFIG,
        output_dir=output_dir,
        max_dt=MAX_DT,
        missing_penalty_m=MISSING_PENALTY_M,
        min_coverage_ok=MIN_COVERAGE_OK,
        stride=STRIDE,
        backend_thresh=BACKEND_THRESH,
        viz=False,
        opts=dpvo_opts,
        target_height=TARGET_HEIGHT,
        target_width=TARGET_WIDTH,
        frontend_mode="dino_hybrid" if method == "DINO-DPVO 50/50" else "dpvo_native",
        frontend_cfg=frontend_cfg,
        frontend=frontend,
        collect_diagnostics=False,
        feature_type=(frontend_cfg.feature_type if frontend_cfg is not None else "DPVO"),
        write_plots=False,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
    else:
        peak_allocated = 0
        peak_reserved = 0
    wall_s = time.perf_counter() - start
    fps = frames / wall_s
    coverage = _metric(metrics, "coverage")
    associated_ate = _metric(metrics, "ate_rmse", "associated_ate", "ate")
    dpvo_style_ate = _metric(metrics, "dpvo_style_ate_rmse")
    peak_allocated_mib = max(peak_allocated, alloc_before) / (1024**2)
    peak_reserved_mib = max(peak_reserved, reserved_before) / (1024**2)
    return status, wall_s, fps, coverage, associated_ate, dpvo_style_ate, peak_allocated_mib, peak_reserved_mib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["native", "dino"], required=True)
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output-root", default=str(ROOT / "runtime_sanity_check/accurate"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the runtime sanity check.")

    method_name = "Native DPVO" if args.method == "native" else "DINO-DPVO 50/50"
    output_root = Path(args.output_root).expanduser().resolve() / args.method
    output_root.mkdir(parents=True, exist_ok=True)
    dpvo_opts = _load_dpvo_opts()
    processed_frames = len(_sequence_image_files(_sequence_dir(args.sequence), stride=STRIDE))

    frontend_cfg = None
    frontend = None
    if args.method == "dino":
        frontend_cfg, frontend = _load_frontend(
            frontend_mode="dino_hybrid",
            frontend_config=FRONTEND_CONFIG,
            frontend_checkpoint=FRONTEND_CHECKPOINT,
            device="cuda",
        )

    manifest = {
        "method": method_name,
        "sequence": args.sequence,
        "processed_frames": processed_frames,
        "warmups": int(args.warmups),
        "repeats": int(args.repeats),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python": platform.python_version(),
        "resolution": f"{TARGET_HEIGHT}x{TARGET_WIDTH}",
        "stride": STRIDE,
        "backend_thresh": BACKEND_THRESH,
        "max_dt": MAX_DT,
        "missing_penalty_m": MISSING_PENALTY_M,
        "min_coverage_ok": MIN_COVERAGE_OK,
        "dpvo_config": str(DPVO_CONFIG),
        "dpvo_weights": str(DPVO_WEIGHTS),
        "dpvo_opts": " ".join(
            f"{dpvo_opts[i]}={dpvo_opts[i + 1]}" if i + 1 < len(dpvo_opts) else dpvo_opts[i]
            for i in range(0, len(dpvo_opts), 2)
        ),
        "frontend_config": str(FRONTEND_CONFIG) if args.method == "dino" else "",
        "frontend_checkpoint": str(FRONTEND_CHECKPOINT) if args.method == "dino" else "",
        "plots": False,
        "diagnostics": False,
    }
    (output_root / "runtime_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for warmup_idx in range(1, int(args.warmups) + 1):
        status, wall_s, fps, *_ = run_once(
            method_name,
            args.sequence,
            output_root / f"warmup_{warmup_idx:02d}",
            dpvo_opts,
            frontend_cfg=frontend_cfg,
            frontend=frontend,
        )
        print(f"{method_name} warmup {warmup_idx}: status={status} fps={fps:.3f} wall_s={wall_s:.3f}")

    rows = []
    for idx in range(1, int(args.repeats) + 1):
        (
            status,
            wall_s,
            fps,
            coverage,
            associated_ate,
            dpvo_style_ate,
            peak_allocated_mib,
            peak_reserved_mib,
        ) = run_once(
            method_name,
            args.sequence,
            output_root / f"repeat_{idx:02d}",
            dpvo_opts,
            frontend_cfg=frontend_cfg,
            frontend=frontend,
        )
        rows.append(
            {
                "method": method_name,
                "sequence": args.sequence,
                "gpu": torch.cuda.get_device_name(0),
                "resolution": f"{TARGET_HEIGHT}x{TARGET_WIDTH}",
                "stride": STRIDE,
                "repeat": idx,
                "processed_frames": processed_frames,
                "wall_s": wall_s,
                "fps": fps,
                "coverage": coverage,
                "associated_ate_rmse": associated_ate,
                "dpvo_style_ate_rmse": dpvo_style_ate,
                "torch_peak_allocated_mib": peak_allocated_mib,
                "torch_peak_reserved_mib": peak_reserved_mib,
                "status": status,
            }
        )
        print(f"{method_name} repeat {idx}: status={status} fps={fps:.3f} wall_s={wall_s:.3f}")

    fps_values = [float(r["fps"]) for r in rows]
    wall_values = [float(r["wall_s"]) for r in rows]
    ate_values = [float(r["associated_ate_rmse"]) for r in rows]
    allocated_values = [float(r["torch_peak_allocated_mib"]) for r in rows]
    reserved_values = [float(r["torch_peak_reserved_mib"]) for r in rows]
    summary = {
        "method": method_name,
        "sequence": args.sequence,
        "gpu": torch.cuda.get_device_name(0),
        "resolution": f"{TARGET_HEIGHT}x{TARGET_WIDTH}",
        "stride": STRIDE,
        "warmups": int(args.warmups),
        "repeats": len(rows),
        "processed_frames": processed_frames,
        "fps_mean": statistics.mean(fps_values),
        "fps_std": statistics.pstdev(fps_values) if len(fps_values) > 1 else 0.0,
        "fps_median": statistics.median(fps_values),
        "fps_min": min(fps_values),
        "fps_max": max(fps_values),
        "wall_s_mean": statistics.mean(wall_values),
        "wall_s_std": statistics.pstdev(wall_values) if len(wall_values) > 1 else 0.0,
        "associated_ate_rmse_mean": statistics.mean(ate_values),
        "associated_ate_rmse_std": statistics.pstdev(ate_values) if len(ate_values) > 1 else 0.0,
        "torch_peak_allocated_mib_mean": statistics.mean(allocated_values),
        "torch_peak_allocated_mib_std": statistics.pstdev(allocated_values) if len(allocated_values) > 1 else 0.0,
        "torch_peak_allocated_mib_median": statistics.median(allocated_values),
        "torch_peak_reserved_mib_mean": statistics.mean(reserved_values),
        "torch_peak_reserved_mib_std": statistics.pstdev(reserved_values) if len(reserved_values) > 1 else 0.0,
        "torch_peak_reserved_mib_median": statistics.median(reserved_values),
        "all_status_ok": all(r["status"] == "ok" for r in rows),
    }

    with (output_root / "per_repeat_runtime.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (output_root / "runtime_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(summary)


if __name__ == "__main__":
    main()
