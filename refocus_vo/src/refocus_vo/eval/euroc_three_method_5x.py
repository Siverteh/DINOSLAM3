from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from refocus_vo.eval.aggregate_euroc_three_method_5x import (
    ALLOWED_STATUSES,
    EUROC_SEQUENCE_ORDER,
    aggregate_euroc_three_method_benchmark,
)
from refocus_vo.eval.focus071_tumwin_finalists import _write_csv, _write_text
from refocus_vo.eval.focus071_vs_dpvo_tum_freiburg123_5x import _split_dpvo_opts


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
class LockedMethodSpec:
    method_id: str
    frontend_mode: str
    checkpoint_path: Path | None
    config_path: Path | None
    repeat01_source_dir: Path | None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_paths(repo_root: Path) -> dict[str, Path]:
    subtree_root = repo_root / "refocus_vo"
    multiscale_root = subtree_root / "runs" / "eval" / "euroc_multiscale32x4_hybrid75_25_vs_dpvo_v1"
    micro_root = subtree_root / "runs" / "eval" / "euroc_micro4_grid_hybrid90_10_vs_dpvo_v1"
    sweep_root = subtree_root / "runs" / "sweeps" / "dino_dpvo_focus071_arch5x2_tumwin_sweep_v1_rerun1" / "train"
    ratio_cfg_root = subtree_root / "runs" / "eval" / "tum_rgbd_freiburg123_arch_ratio_ablation_v1" / "ratio_eval_configs"
    return {
        "dataset_root": subtree_root / "data" / "euroc_asl",
        "dpvo_root": subtree_root / "external" / "repos" / "DPVO",
        "dpvo_weights": subtree_root / "external" / "repos" / "DPVO" / "dpvo.pth",
        "dpvo_config": subtree_root / "external" / "repos" / "DPVO" / "config" / "default.yaml",
        "dpvo_repeat01": multiscale_root / "dpvo_native_matched",
        "multiscale_repeat01": multiscale_root / "winner_multiscale_32x4_v1_hybrid75_25",
        "multiscale_checkpoint": sweep_root / "multiscale_32x4_v1" / "best_hybrid.pt",
        "multiscale_config": ratio_cfg_root / "multiscale_32x4_v1__hybrid75_25.yaml",
        "micro_repeat01": micro_root / "winner_micro4_grid_v1_hybrid90_10",
        "micro_checkpoint": sweep_root / "micro4_grid_v1" / "best_hybrid.pt",
        "micro_config": ratio_cfg_root / "micro4_grid_v1__hybrid90_10.yaml",
        "output_root": subtree_root / "runs" / "eval" / "euroc_dpvo_multiscale_micro_5x_v1",
    }


def _gpu_heavy_process_lines(exclude_pids: set[int] | None = None) -> list[str]:
    patterns = (
        "refocus_vo.train_dino_dpvo_frontend",
        "refocus_vo.sweeps.run_assoc9_sweep",
        "refocus_vo.eval.external_dpvo",
        "refocus_vo.eval.external_dpvo_euroc",
        "refocus_vo.eval.focus071_arch_dual_finalists_5x",
        "refocus_vo.eval.euroc_three_method_5x",
        "run_euroc_dpvo_multiscale_micro_5x_v1.sh",
        "run_multiscale_vs_dpvo_euroc_v1.sh",
    )
    result = subprocess.run(
        ["ps", "-eo", "pid,ppid,cmd"],
        text=True,
        capture_output=True,
        check=True,
    )
    output: list[str] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid_text, _ppid_text, cmd = parts
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if exclude_pids and pid in exclude_pids:
            continue
        if any(pattern in cmd for pattern in patterns):
            output.append(line)
    return output


def _assert_idle_or_raise(*, force: bool) -> None:
    active = _gpu_heavy_process_lines(exclude_pids={os.getpid(), os.getppid()})
    if active and not force:
        raise RuntimeError(
            "Refusing to start EuRoC 5x while other GPU-heavy jobs are active:\n"
            + "\n".join(active[:10])
        )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        import csv

        return list(csv.DictReader(f))


def _repeat_dir(*, output_root: Path, method_id: str, repeat_idx: int) -> Path:
    return output_root / method_id / f"repeat_{repeat_idx:02d}"


def _validate_repeat_dir(repeat_dir: Path, *, expected_sequences: list[str]) -> None:
    csv_path = repeat_dir / "dpvo_style_metrics_summary.csv"
    rows = _read_csv_rows(csv_path)
    seen = [str(row.get("sequence", "")).strip() for row in rows]
    if len(rows) != len(expected_sequences):
        raise ValueError(f"{csv_path} has {len(rows)} rows; expected {len(expected_sequences)}")
    if seen != list(expected_sequences):
        raise ValueError(f"{csv_path} sequence order/content mismatch")
    bad_rows = [row for row in rows if str(row.get("status", "")).strip() not in ALLOWED_STATUSES]
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


def _locked_methods(
    *,
    defaults: dict[str, Path],
) -> list[LockedMethodSpec]:
    return _selected_methods(
        defaults=defaults,
        method_ids=[
            "dpvo_native_matched",
            "multiscale_32x4_v1_hybrid75_25",
            "micro4_grid_v1_hybrid90_10",
        ],
    )


def _selected_methods(
    *,
    defaults: dict[str, Path],
    method_ids: list[str],
) -> list[LockedMethodSpec]:
    all_methods = {
        "dpvo_native_matched": LockedMethodSpec(
            method_id="dpvo_native_matched",
            frontend_mode="dpvo_native",
            checkpoint_path=None,
            config_path=None,
            repeat01_source_dir=defaults["dpvo_repeat01"],
        ),
        "multiscale_32x4_v1_hybrid75_25": LockedMethodSpec(
            method_id="multiscale_32x4_v1_hybrid75_25",
            frontend_mode=FIXED_FRONTEND_MODE,
            checkpoint_path=defaults["multiscale_checkpoint"],
            config_path=defaults["multiscale_config"],
            repeat01_source_dir=defaults["multiscale_repeat01"],
        ),
        "micro4_grid_v1_hybrid90_10": LockedMethodSpec(
            method_id="micro4_grid_v1_hybrid90_10",
            frontend_mode=FIXED_FRONTEND_MODE,
            checkpoint_path=defaults["micro_checkpoint"],
            config_path=defaults["micro_config"],
            repeat01_source_dir=defaults["micro_repeat01"],
        ),
    }
    methods: list[LockedMethodSpec] = []
    for method_id in method_ids:
        method = all_methods.get(method_id)
        if method is None:
            raise ValueError(f"Unsupported method_id: {method_id}")
        methods.append(method)
    for method in methods:
        if method.repeat01_source_dir is not None and not method.repeat01_source_dir.exists():
            raise FileNotFoundError(f"repeat_01 source missing: {method.repeat01_source_dir}")
        if method.checkpoint_path is not None and not method.checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint missing: {method.checkpoint_path}")
        if method.config_path is not None and not method.config_path.exists():
            raise FileNotFoundError(f"config missing: {method.config_path}")
    return methods


def _parse_method_ids(raw: str) -> list[str]:
    items = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not items:
        raise ValueError("method_ids must not be empty")
    return items


def _method_csv_rows(methods: list[LockedMethodSpec]) -> list[dict[str, object]]:
    return [
        {
            "method_id": method.method_id,
            "frontend_mode": method.frontend_mode,
            "checkpoint_path": str(method.checkpoint_path) if method.checkpoint_path is not None else "",
            "config_path": str(method.config_path) if method.config_path is not None else "",
            "repeat01_source_dir": str(method.repeat01_source_dir) if method.repeat01_source_dir is not None else "",
        }
        for method in methods
    ]


def _format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _eval_command(
    *,
    dataset_root: Path,
    dpvo_root: Path,
    dpvo_weights: Path,
    dpvo_config: Path,
    output_dir: Path,
    method: LockedMethodSpec,
    sequences: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "refocus_vo.eval.external_dpvo_euroc",
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
        "--frontend-mode",
        str(method.frontend_mode),
        "--image-height",
        str(FIXED_IMAGE_HEIGHT),
        "--image-width",
        str(FIXED_IMAGE_WIDTH),
        "--stride",
        str(FIXED_STRIDE),
        "--backend-thresh",
        str(FIXED_BACKEND_THRESH),
    ]
    opts = _split_dpvo_opts(FIXED_DPVO_OPTS)
    if opts:
        cmd.append("--opts")
        cmd.extend(opts)
    if method.config_path is not None:
        cmd.extend(["--frontend-config", str(method.config_path)])
    if method.checkpoint_path is not None:
        cmd.extend(["--frontend-checkpoint", str(method.checkpoint_path)])
    return cmd


def _run_eval(
    *,
    repo_root: Path,
    dataset_root: Path,
    dpvo_root: Path,
    dpvo_weights: Path,
    dpvo_config: Path,
    output_dir: Path,
    method: LockedMethodSpec,
    sequences: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = _eval_command(
        dataset_root=dataset_root,
        dpvo_root=dpvo_root,
        dpvo_weights=dpvo_weights,
        dpvo_config=dpvo_config,
        output_dir=output_dir,
        method=method,
        sequences=sequences,
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


def _print_dry_run(
    *,
    output_root: Path,
    methods: list[LockedMethodSpec],
    sequences: list[str],
    dataset_root: Path,
    dpvo_root: Path,
    dpvo_weights: Path,
    dpvo_config: Path,
    repeats: int,
    fresh: bool,
) -> None:
    print(f"output_root: {output_root}")
    print(f"dataset_root: {dataset_root}")
    print(f"sequence_count: {len(sequences)}")
    print(f"repeats: {repeats}")
    print(f"fresh: {fresh}")
    print("locked_methods:")
    for method in methods:
        print(f"  - {method.method_id}: frontend_mode={method.frontend_mode}")
        if method.repeat01_source_dir is not None:
            print(f"    repeat_01_source={method.repeat01_source_dir}")
        if method.checkpoint_path is not None:
            print(f"    checkpoint={method.checkpoint_path}")
        if method.config_path is not None:
            print(f"    config={method.config_path}")
    print("repeat_commands:")
    for method in methods:
        start_repeat = 1 if fresh else 2
        for repeat_idx in range(start_repeat, int(repeats) + 1):
            repeat_dir = _repeat_dir(output_root=output_root, method_id=method.method_id, repeat_idx=repeat_idx)
            cmd = _eval_command(
                dataset_root=dataset_root,
                dpvo_root=dpvo_root,
                dpvo_weights=dpvo_weights,
                dpvo_config=dpvo_config,
                output_dir=repeat_dir,
                method=method,
                sequences=sequences,
            )
            print(f"  [{method.method_id} repeat_{repeat_idx:02d}] {_format_command(cmd)}")


def main() -> None:
    repo_root = _repo_root()
    defaults = _default_paths(repo_root)
    ap = argparse.ArgumentParser(description="Run a EuRoC repeat benchmark for selected methods.")
    ap.add_argument("--dataset-root", default=str(defaults["dataset_root"]))
    ap.add_argument("--dpvo-root", default=str(defaults["dpvo_root"]))
    ap.add_argument("--dpvo-weights", default=str(defaults["dpvo_weights"]))
    ap.add_argument("--dpvo-config", default=str(defaults["dpvo_config"]))
    ap.add_argument("--output-root", default=str(defaults["output_root"]))
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument(
        "--method-ids",
        default="dpvo_native_matched,multiscale_32x4_v1_hybrid75_25,micro4_grid_v1_hybrid90_10",
    )
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    dpvo_root = Path(args.dpvo_root).expanduser().resolve()
    dpvo_weights = Path(args.dpvo_weights).expanduser().resolve()
    dpvo_config = Path(args.dpvo_config).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    sequences = list(EUROC_SEQUENCE_ORDER)

    if not dataset_root.exists():
        raise FileNotFoundError(f"EuRoC dataset root not found: {dataset_root}")
    if not dpvo_root.exists():
        raise FileNotFoundError(f"DPVO repo not found: {dpvo_root}")
    if not dpvo_weights.exists():
        raise FileNotFoundError(f"DPVO weights not found: {dpvo_weights}")
    if not dpvo_config.exists():
        raise FileNotFoundError(f"DPVO config not found: {dpvo_config}")

    methods = _selected_methods(
        defaults=defaults,
        method_ids=_parse_method_ids(args.method_ids),
    )

    if args.dry_run:
        _print_dry_run(
            output_root=output_root,
            methods=methods,
            sequences=sequences,
            dataset_root=dataset_root,
            dpvo_root=dpvo_root,
            dpvo_weights=dpvo_weights,
            dpvo_config=dpvo_config,
            repeats=int(args.repeats),
            fresh=bool(args.fresh),
        )
        return

    _assert_idle_or_raise(force=bool(args.force))

    output_root.mkdir(parents=True, exist_ok=True)
    _write_text(output_root / "frozen_sequences.txt", "\n".join(sequences) + "\n")
    _write_csv(
        output_root / "locked_methods.csv",
        _method_csv_rows(methods),
        ["method_id", "frontend_mode", "checkpoint_path", "config_path", "repeat01_source_dir"],
    )

    for method in methods:
        start_repeat = 1
        if not args.fresh:
            if method.repeat01_source_dir is None:
                raise ValueError(f"repeat_01 reuse requested, but no source is defined for {method.method_id}")
            repeat01_dir = _repeat_dir(output_root=output_root, method_id=method.method_id, repeat_idx=1)
            _ensure_reused_repeat01(
                source_dir=method.repeat01_source_dir,
                dest_dir=repeat01_dir,
                expected_sequences=sequences,
            )
            start_repeat = 2
        for repeat_idx in range(start_repeat, int(args.repeats) + 1):
            repeat_dir = _repeat_dir(output_root=output_root, method_id=method.method_id, repeat_idx=repeat_idx)
            if repeat_dir.exists():
                try:
                    _validate_repeat_dir(repeat_dir, expected_sequences=sequences)
                    print(f"[euroc_3method_5x] reusing existing {method.method_id} repeat_{repeat_idx:02d}")
                    continue
                except Exception:
                    shutil.rmtree(repeat_dir)
            print(f"[euroc_3method_5x] running {method.method_id} repeat_{repeat_idx:02d}")
            _run_eval(
                repo_root=repo_root,
                dataset_root=dataset_root,
                dpvo_root=dpvo_root,
                dpvo_weights=dpvo_weights,
                dpvo_config=dpvo_config,
                output_dir=repeat_dir,
                method=method,
                sequences=sequences,
            )
            _validate_repeat_dir(repeat_dir, expected_sequences=sequences)

    outputs = aggregate_euroc_three_method_benchmark(
        benchmark_root=output_root,
        method_ids=[method.method_id for method in methods],
        expected_sequences=sequences,
        repeats=int(args.repeats),
    )
    for key, path in outputs.items():
        print(f"[euroc_3method_5x] {key}: {path}")
    print(f"[euroc_3method_5x] benchmark complete: {output_root}")


if __name__ == "__main__":
    main()
