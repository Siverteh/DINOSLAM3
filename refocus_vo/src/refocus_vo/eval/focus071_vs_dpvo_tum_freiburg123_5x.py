from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from refocus_vo.eval.aggregate_tum_freiburg123_repeats import aggregate_benchmark
from refocus_vo.eval.sparse_vo import _normalize_sequence_name
from refocus_vo.sweeps.run_assoc9_sweep import _materialize_pure_pack_config


REPEATS = 5
TUM_MAX_DT = 0.02
TUM_MISSING_PENALTY_METERS = 3.0
TUM_MIN_COVERAGE_OK = 0.95

DPVO_DEFAULT_OPTS = "BUFFER_SIZE=384 PATCHES_PER_FRAME=24 REMOVAL_WINDOW=12 OPTIMIZATION_WINDOW=6 PATCH_LIFETIME=9"
FOCUS071_PURE100_OPTS = "BUFFER_SIZE=512 PATCHES_PER_FRAME=128 REMOVAL_WINDOW=24 OPTIMIZATION_WINDOW=12 PATCH_LIFETIME=15"


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    frontend_mode: str
    stride: int
    backend_thresh: float
    image_height: int
    image_width: int
    dpvo_opts: str
    frontend_config: Path | None = None
    checkpoint: Path | None = None
    collect_diagnostics: bool = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_paths(repo_root: Path) -> dict[str, Path]:
    subtree_root = repo_root / "refocus_vo"
    return {
        "dataset_root": repo_root / "src" / "dino_slam3" / "data" / "tum_rgbd",
        "dpvo_root": subtree_root / "external" / "repos" / "DPVO",
        "dpvo_weights": subtree_root / "external" / "repos" / "DPVO" / "dpvo.pth",
        "dpvo_config": subtree_root / "external" / "repos" / "DPVO" / "config" / "default.yaml",
        "focus071_checkpoint": subtree_root / "runs" / "sweeps" / "dino_dpvo_focus071_lr_only_sweep_v1" / "train" / "focus071lr_orig_const3e6_v1" / "best_pure100.pt",
        "focus071_source_config": subtree_root / "runs" / "sweeps" / "dino_dpvo_focus071_lr_only_sweep_v1" / "generated_train_configs" / "focus071lr_orig_const3e6_v1.yaml",
        "output_root": subtree_root / "runs" / "eval" / "tum_rgbd_freiburg123_dpvo_vs_focus071_v1",
    }


def _split_dpvo_opts(opts_text: str) -> list[str]:
    raw = [item for item in shlex.split(str(opts_text)) if item.strip()]
    output: list[str] = []
    for item in raw:
        if "=" in item:
            key, value = item.split("=", 1)
            output.extend([key, value])
        else:
            output.append(item)
    return output


def _enumerate_freiburg_sequences(dataset_root: Path) -> list[str]:
    sequences = []
    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir():
            continue
        name = str(child.name)
        if not (
            name.startswith("rgbd_dataset_freiburg1_")
            or name.startswith("rgbd_dataset_freiburg2_")
            or name.startswith("rgbd_dataset_freiburg3_")
        ):
            continue
        sequences.append(_normalize_sequence_name(name))
    return sorted(sequences)


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _gpu_heavy_process_lines(exclude_pid: int | None = None) -> list[str]:
    patterns = (
        "refocus_vo.train_dino_dpvo_frontend",
        "refocus_vo.sweeps.run_assoc9_sweep",
        "refocus_vo.eval.external_dpvo",
        "refocus_vo.eval.external_dpvo_tartanair",
        "refocus_vo.eval.external_dpvo_tartanair_mono",
        "refocus_vo.eval.focus071_tartanair_mono_paper_eval",
        "refocus_vo.eval.focus071_vs_dpvo_tum_freiburg123_5x",
        "scripts/train_dino_dpvo_frontend.sh",
        "run_dpvo_tum.sh",
        "eval_dino_dpvo_tum.sh",
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
        if exclude_pid is not None and pid == exclude_pid:
            continue
        if any(pattern in cmd for pattern in patterns):
            output.append(line)
    return output


def _materialize_focus071_pure100_config(
    *,
    source_config: Path,
    output_root: Path,
) -> Path:
    output_path = output_root / "focus071_best_pure100_eval.yaml"
    return _materialize_pure_pack_config(
        source_config,
        output_path,
        "tum_freiburg123_focus071_pure100",
    )


def _method_specs(
    *,
    pure100_config: Path,
    focus071_checkpoint: Path,
) -> list[MethodSpec]:
    return [
        MethodSpec(
            method_id="dpvo_native",
            frontend_mode="dpvo_native",
            stride=4,
            backend_thresh=32.0,
            image_height=240,
            image_width=320,
            dpvo_opts=DPVO_DEFAULT_OPTS,
            collect_diagnostics=False,
        ),
        MethodSpec(
            method_id="focus071_best",
            frontend_mode="dino_proposals",
            stride=4,
            backend_thresh=32.0,
            image_height=240,
            image_width=320,
            dpvo_opts=FOCUS071_PURE100_OPTS,
            frontend_config=pure100_config,
            checkpoint=focus071_checkpoint,
            collect_diagnostics=True,
        ),
    ]


def _format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _eval_command(
    *,
    repo_root: Path,
    dataset_root: Path,
    dpvo_root: Path,
    dpvo_weights: Path,
    dpvo_config: Path,
    method: MethodSpec,
    sequences: list[str],
    output_dir: Path,
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
        str(method.stride),
        "--backend-thresh",
        str(method.backend_thresh),
        "--image-height",
        str(method.image_height),
        "--image-width",
        str(method.image_width),
        "--frontend-mode",
        str(method.frontend_mode),
    ]
    if method.frontend_config is not None:
        cmd.extend(["--frontend-config", str(method.frontend_config)])
    if method.checkpoint is not None:
        cmd.extend(["--frontend-checkpoint", str(method.checkpoint)])
    if method.collect_diagnostics:
        cmd.append("--collect-diagnostics")
    opts = _split_dpvo_opts(method.dpvo_opts)
    if opts:
        cmd.append("--opts")
        cmd.extend(opts)
    return cmd


def _run_repeat(
    *,
    repo_root: Path,
    dataset_root: Path,
    dpvo_root: Path,
    dpvo_weights: Path,
    dpvo_config: Path,
    method: MethodSpec,
    sequences: list[str],
    repeat_idx: int,
    output_root: Path,
) -> None:
    repeat_dir = output_root / method.method_id / f"repeat_{repeat_idx:02d}"
    repeat_dir.mkdir(parents=True, exist_ok=True)
    cmd = _eval_command(
        repo_root=repo_root,
        dataset_root=dataset_root,
        dpvo_root=dpvo_root,
        dpvo_weights=dpvo_weights,
        dpvo_config=dpvo_config,
        method=method,
        sequences=sequences,
        output_dir=repeat_dir,
    )
    (repeat_dir / "command.txt").write_text(_format_command(cmd) + "\n", encoding="utf-8")
    print(
        f"[tum_freiburg123_5x] running {method.method_id} repeat_{repeat_idx:02d} "
        f"on {len(sequences)} sequences"
    )
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(repo_root / "refocus_vo" / "src") + ":" + str(dpvo_root) + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    log_path = repeat_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log_file:
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
    sequences: list[str],
    methods: list[MethodSpec],
    repo_root: Path,
    dataset_root: Path,
    dpvo_root: Path,
    dpvo_weights: Path,
    dpvo_config: Path,
) -> None:
    print(f"output_root: {output_root}")
    print(f"sequence_count: {len(sequences)}")
    print("sequences:")
    for seq in sequences:
        print(f"  - {seq}")
    print("commands:")
    for method in methods:
        cmd = _eval_command(
            repo_root=repo_root,
            dataset_root=dataset_root,
            dpvo_root=dpvo_root,
            dpvo_weights=dpvo_weights,
            dpvo_config=dpvo_config,
            method=method,
            sequences=sequences,
            output_dir=output_root / method.method_id / "repeat_01",
        )
        print(f"  [{method.method_id}] {_format_command(cmd)}")


def main() -> None:
    repo_root = _repo_root()
    defaults = _default_paths(repo_root)
    ap = argparse.ArgumentParser(
        description="Run a full 5x TUM RGB-D Freiburg1/2/3 benchmark for DPVO and Focus071."
    )
    ap.add_argument("--dataset-root", default=str(defaults["dataset_root"]))
    ap.add_argument("--dpvo-root", default=str(defaults["dpvo_root"]))
    ap.add_argument("--dpvo-weights", default=str(defaults["dpvo_weights"]))
    ap.add_argument("--dpvo-config", default=str(defaults["dpvo_config"]))
    ap.add_argument("--focus071-checkpoint", default=str(defaults["focus071_checkpoint"]))
    ap.add_argument("--focus071-source-config", default=str(defaults["focus071_source_config"]))
    ap.add_argument("--output-root", default=str(defaults["output_root"]))
    ap.add_argument("--sequences", default="")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    dpvo_root = Path(args.dpvo_root).expanduser().resolve()
    dpvo_weights = Path(args.dpvo_weights).expanduser().resolve()
    dpvo_config = Path(args.dpvo_config).expanduser().resolve()
    focus071_checkpoint = Path(args.focus071_checkpoint).expanduser().resolve()
    focus071_source_config = Path(args.focus071_source_config).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"TUM dataset root not found: {dataset_root}")
    if not dpvo_root.exists():
        raise FileNotFoundError(f"DPVO repo not found: {dpvo_root}")
    if not dpvo_weights.exists():
        raise FileNotFoundError(f"DPVO weights not found: {dpvo_weights}")
    if not dpvo_config.exists():
        raise FileNotFoundError(f"DPVO config not found: {dpvo_config}")
    if not focus071_checkpoint.exists():
        raise FileNotFoundError(f"Focus071 checkpoint not found: {focus071_checkpoint}")
    if not focus071_source_config.exists():
        raise FileNotFoundError(f"Focus071 source config not found: {focus071_source_config}")

    sequences = (
        [item.strip() for item in str(args.sequences).split(",") if item.strip()]
        if str(args.sequences).strip()
        else _enumerate_freiburg_sequences(dataset_root)
    )
    if not sequences:
        raise ValueError("No Freiburg sequences were selected")

    output_root.mkdir(parents=True, exist_ok=True)
    _write_lines(output_root / "frozen_sequences.txt", sequences)
    pure100_config = _materialize_focus071_pure100_config(
        source_config=focus071_source_config,
        output_root=output_root,
    )
    methods = _method_specs(
        pure100_config=pure100_config,
        focus071_checkpoint=focus071_checkpoint,
    )

    if args.dry_run:
        _print_dry_run(
            output_root=output_root,
            sequences=sequences,
            methods=methods,
            repo_root=repo_root,
            dataset_root=dataset_root,
            dpvo_root=dpvo_root,
            dpvo_weights=dpvo_weights,
            dpvo_config=dpvo_config,
        )
        return

    active = _gpu_heavy_process_lines(exclude_pid=os.getpid())
    if active:
        raise RuntimeError(
            "Refusing to start because another GPU-heavy eval/training job is active:\n"
            + "\n".join(active[:10])
        )

    for method in methods:
        for repeat_idx in range(1, int(args.repeats) + 1):
            _run_repeat(
                repo_root=repo_root,
                dataset_root=dataset_root,
                dpvo_root=dpvo_root,
                dpvo_weights=dpvo_weights,
                dpvo_config=dpvo_config,
                method=method,
                sequences=sequences,
                repeat_idx=repeat_idx,
                output_root=output_root,
            )

    aggregate_benchmark(
        benchmark_root=output_root,
        expected_sequences=sequences,
        repeats=int(args.repeats),
    )
    print(f"[tum_freiburg123_5x] benchmark complete: {output_root}")


if __name__ == "__main__":
    main()
