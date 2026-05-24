from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests


TEST_SPLIT = [f"ME{i:03d}" for i in range(8)] + [f"MH{i:03d}" for i in range(8)]

PAPER_OURS_DEFAULT = {
    "ME000": 0.16,
    "ME001": 0.11,
    "ME002": 0.11,
    "ME003": 0.66,
    "ME004": 0.31,
    "ME005": 0.14,
    "ME006": 0.30,
    "ME007": 0.13,
    "MH000": 0.21,
    "MH001": 0.04,
    "MH002": 0.04,
    "MH003": 0.08,
    "MH004": 0.58,
    "MH005": 0.17,
    "MH006": 0.11,
    "MH007": 0.15,
}

PAPER_OURS_FAST = {
    "ME000": 0.35,
    "ME001": 0.13,
    "ME002": 0.27,
    "ME003": 0.71,
    "ME004": 0.47,
    "ME005": 0.16,
    "ME006": 0.30,
    "ME007": 0.13,
    "MH000": 0.34,
    "MH001": 0.05,
    "MH002": 0.06,
    "MH003": 0.07,
    "MH004": 0.81,
    "MH005": 0.41,
    "MH006": 0.09,
    "MH007": 0.14,
}

PAPER_DROID_VO = {
    "ME000": 0.22,
    "ME001": 0.15,
    "ME002": 0.24,
    "ME003": 1.27,
    "ME004": 1.04,
    "ME005": 0.14,
    "ME006": 1.32,
    "ME007": 0.77,
    "MH000": 0.32,
    "MH001": 0.13,
    "MH002": 0.08,
    "MH003": 0.09,
    "MH004": 1.52,
    "MH005": 0.69,
    "MH006": 0.39,
    "MH007": 0.97,
}

GOOGLE_DRIVE_IMAGE_FILE_ID = "1N8qoU-oEjRKdaKSrHPWA-xsnRtofR_jJ"
BOX_GROUNDTRUTH_URL = "https://cmu.box.com/shared/static/3p1sf0eljfwrz4qgbpc6g95xtn2alyfk.zip"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    checkpoint: Path
    frontend_config: Path


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_candidates(repo_root: Path) -> list[Candidate]:
    return [
        Candidate(
            candidate_id="new_best",
            checkpoint=repo_root / "refocus_vo" / "runs" / "sweeps" / "dino_dpvo_focus071_lr_only_sweep_v1" / "train" / "focus071lr_orig_const3e6_v1" / "best_pure100.pt",
            frontend_config=repo_root / "refocus_vo" / "runs" / "sweeps" / "dino_dpvo_focus071_lr_only_sweep_v1" / "generated_train_configs" / "focus071lr_orig_const3e6_v1.yaml",
        ),
        Candidate(
            candidate_id="old_best",
            checkpoint=repo_root / "refocus_vo" / "runs" / "sweeps" / "dino_dpvo_paper_room_noroom_family_sweep_v1_rerun1" / "train" / "focus071_exact_lowlr_v1" / "best_pure100.pt",
            frontend_config=repo_root / "refocus_vo" / "runs" / "sweeps" / "dino_dpvo_paper_room_noroom_family_sweep_v1_rerun1" / "generated_train_configs" / "focus071_exact_lowlr_v1.yaml",
        ),
    ]


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_dpvo_style_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _mean_ate_from_rows(rows: list[dict[str, str]]) -> float:
    values = [_safe_float(row.get("ate_rmse")) for row in rows if math.isfinite(_safe_float(row.get("ate_rmse")))]
    if not values:
        return math.nan
    return float(sum(values) / len(values))


def _mean_coverage_from_rows(rows: list[dict[str, str]]) -> float:
    values = [_safe_float(row.get("coverage")) for row in rows if math.isfinite(_safe_float(row.get("coverage")))]
    if not values:
        return math.nan
    return float(sum(values) / len(values))


def _median_per_sequence(rows_by_repeat: list[list[dict[str, str]]]) -> list[dict[str, object]]:
    per_sequence: dict[str, list[float]] = {}
    for rows in rows_by_repeat:
        for row in rows:
            sequence = str(row.get("sequence", "")).strip()
            value = _safe_float(row.get("ate_rmse"))
            if not sequence or not math.isfinite(value):
                continue
            per_sequence.setdefault(sequence, []).append(value)
    output = []
    for sequence in TEST_SPLIT:
        values = per_sequence.get(sequence, [])
        median = statistics.median(values) if values else math.nan
        output.append({"sequence": sequence, "median_ate_rmse": median})
    return output


def _training_process_lines_from_ps_output(ps_output: str, *, exclude_pid: int | None = None) -> list[str]:
    lines: list[str] = []
    for raw_line in ps_output.splitlines():
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
        if (
            "refocus_vo.sweeps.run_assoc9_sweep" in cmd
            or "refocus_vo.train_dino_dpvo_frontend" in cmd
            or "scripts/train_dino_dpvo_frontend.sh" in cmd
        ):
            lines.append(line)
    return lines


def _detect_active_training() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "pid,ppid,cmd"],
        text=True,
        capture_output=True,
        check=True,
    )
    return _training_process_lines_from_ps_output(result.stdout, exclude_pid=os.getpid())


def _wait_for_training_idle(poll_seconds: int) -> None:
    while True:
        lines = _detect_active_training()
        if not lines:
            return
        print("[focus071_tartanair_eval] waiting for current training sweep to finish...")
        for line in lines[:4]:
            print(f"  {line}")
        time.sleep(max(5, int(poll_seconds)))


def _google_drive_confirm_token(text: str, cookies: requests.cookies.RequestsCookieJar) -> str | None:
    for key, value in cookies.items():
        if str(key).startswith("download_warning"):
            return str(value)
    patterns = [
        r"confirm=([0-9A-Za-z_]+)",
        r"confirm=([0-9A-Za-z_-]+)&",
        r'"downloadUrl":"[^"]*confirm=([0-9A-Za-z_-]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return str(match.group(1))
    return None


def _looks_like_html_bytes(data: bytes) -> bool:
    prefix = data.lstrip().lower()
    return (
        prefix.startswith(b"<!doctype html")
        or prefix.startswith(b"<html")
        or b"<title>google drive" in prefix[:256]
    )


def _archive_is_valid(path: Path, archive_type: str) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    with path.open("rb") as f:
        prefix = f.read(512)
    if _looks_like_html_bytes(prefix):
        return False
    if archive_type == "tar.gz":
        return prefix.startswith(b"\x1f\x8b")
    if archive_type == "zip":
        return prefix.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"))
    raise ValueError(f"Unsupported archive type: {archive_type}")


def _google_drive_form(text: str) -> tuple[str, dict[str, str]] | None:
    action_match = re.search(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"', text)
    if not action_match:
        return None
    params = {
        name: value
        for name, value in re.findall(
            r'<input[^>]+type="hidden"[^>]+name="([^"]+)"[^>]+value="([^"]*)"', text
        )
    }
    if not params:
        return None
    return str(action_match.group(1)), params


def _download_google_drive(file_id: str, destination: Path) -> None:
    url = "https://drive.google.com/uc?export=download"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        response = session.get(url, params={"id": file_id}, stream=True, timeout=60)
        response.raise_for_status()
        content_type = str(response.headers.get("content-type", "")).lower()
        if "text/html" in content_type:
            form = _google_drive_form(response.text[:200000])
            token = _google_drive_confirm_token(response.text[:200000], response.cookies)
            response.close()
            if form is not None:
                form_action, form_params = form
                response = session.get(form_action, params=form_params, stream=True, timeout=60)
            elif token is not None:
                response = session.get(url, params={"id": file_id, "confirm": token}, stream=True, timeout=60)
            else:
                raise RuntimeError("Google Drive download page did not contain usable confirmation parameters")
            response.raise_for_status()
            second_content_type = str(response.headers.get("content-type", "")).lower()
            if "text/html" in second_content_type:
                preview = response.text[:500].replace("\n", " ")
                raise RuntimeError(f"Google Drive returned HTML instead of archive content: {preview}")
        with destination.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _download_url(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _locate_sequence_root(root: Path, *, want_files: bool) -> Path | None:
    for candidate in [root] + [path for path in root.rglob("*") if path.is_dir()]:
        if want_files:
            names = {path.name for path in candidate.glob("*.txt")}
            required = {f"{sequence}.txt" for sequence in TEST_SPLIT}
        else:
            names = {path.name for path in candidate.iterdir() if path.is_dir()} if candidate.exists() else set()
            required = set(TEST_SPLIT)
        if required.issubset(names):
            return candidate
    return None


def _dataset_ready(data_root: Path) -> bool:
    mono_root = data_root / "mono"
    gt_root = data_root / "mono_gt"
    if not mono_root.exists() or not gt_root.exists():
        return False
    for sequence in TEST_SPLIT:
        if not (mono_root / sequence).exists():
            return False
        if not (gt_root / f"{sequence}.txt").exists():
            return False
    return True


def _normalize_sequence_dir(source_root: Path, target_root: Path) -> None:
    if target_root.exists():
        return
    if source_root == target_root:
        return
    if source_root == target_root.parent:
        target_root.mkdir(parents=True, exist_ok=True)
        for sequence in TEST_SPLIT:
            src = source_root / sequence
            if src.exists():
                shutil.move(str(src), str(target_root / sequence))
        return
    target_root.symlink_to(source_root, target_is_directory=True)


def _normalize_gt_dir(source_root: Path, target_root: Path) -> None:
    if target_root.exists():
        return
    if source_root == target_root:
        return
    if source_root == target_root.parent:
        target_root.mkdir(parents=True, exist_ok=True)
        for sequence in TEST_SPLIT:
            src = source_root / f"{sequence}.txt"
            if src.exists():
                shutil.move(str(src), str(target_root / f"{sequence}.txt"))
        return
    target_root.symlink_to(source_root, target_is_directory=True)


def _ensure_dataset(data_root: Path) -> tuple[Path, Path]:
    mono_root = data_root / "mono"
    gt_root = data_root / "mono_gt"
    if _dataset_ready(data_root):
        return mono_root, gt_root

    data_root.mkdir(parents=True, exist_ok=True)
    images_archive = data_root / "images.tar.gz"
    gt_archive = data_root / "groundtruth.zip"

    if images_archive.exists() and not _archive_is_valid(images_archive, "tar.gz"):
        print("[focus071_tartanair_eval] removing invalid mono image archive before retry...")
        images_archive.unlink()
    if gt_archive.exists() and not _archive_is_valid(gt_archive, "zip"):
        print("[focus071_tartanair_eval] removing invalid mono ground truth archive before retry...")
        gt_archive.unlink()

    if not images_archive.exists():
        print("[focus071_tartanair_eval] downloading TartanAir mono test images...")
        _download_google_drive(GOOGLE_DRIVE_IMAGE_FILE_ID, images_archive)
    if not _archive_is_valid(images_archive, "tar.gz"):
        raise RuntimeError(f"Downloaded image archive is invalid: {images_archive}")
    if not gt_archive.exists():
        print("[focus071_tartanair_eval] downloading TartanAir mono ground truth...")
        _download_url(BOX_GROUNDTRUTH_URL, gt_archive)
    if not _archive_is_valid(gt_archive, "zip"):
        raise RuntimeError(f"Downloaded ground-truth archive is invalid: {gt_archive}")

    print("[focus071_tartanair_eval] extracting mono test archives...")
    with tarfile.open(images_archive, "r:gz") as tf:
        tf.extractall(data_root)
    with zipfile.ZipFile(gt_archive, "r") as zf:
        zf.extractall(data_root)

    found_mono = _locate_sequence_root(data_root, want_files=False)
    found_gt = _locate_sequence_root(data_root, want_files=True)
    if found_mono is None or found_gt is None:
        raise FileNotFoundError(f"Could not locate mono/mono_gt structure under {data_root}")

    _normalize_sequence_dir(found_mono, mono_root)
    _normalize_gt_dir(found_gt, gt_root)

    if not _dataset_ready(data_root):
        raise FileNotFoundError(f"Mono test split still incomplete under {data_root}")
    return mono_root, gt_root


def _python_bin(repo_root: Path) -> Path:
    return repo_root / "refocus_vo" / ".micromamba" / "envs" / "dpvo" / "bin" / "python"


def _run_eval(
    *,
    repo_root: Path,
    candidate: Candidate,
    dataset_root: Path,
    groundtruth_root: Path,
    output_dir: Path,
    sequences: list[str],
) -> Path:
    dpvo_root = repo_root / "refocus_vo" / "external" / "repos" / "DPVO"
    dpvo_weights = dpvo_root / "dpvo.pth"
    dpvo_config = dpvo_root / "config" / "default.yaml"
    python_bin = _python_bin(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root / 'refocus_vo' / 'src'}:{repo_root / 'src'}" + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    cmd = [
        str(python_bin),
        "-m",
        "refocus_vo.eval.external_dpvo_tartanair_mono",
        "--dataset-root",
        str(dataset_root),
        "--groundtruth-root",
        str(groundtruth_root),
        "--dpvo-root",
        str(dpvo_root),
        "--weights",
        str(dpvo_weights),
        "--config",
        str(dpvo_config),
        "--output-dir",
        str(output_dir),
        "--sequences",
        ",".join(sequences),
        "--frontend-mode",
        "dino_proposals",
        "--frontend-config",
        str(candidate.frontend_config),
        "--frontend-checkpoint",
        str(candidate.checkpoint),
        "--stride",
        "1",
        "--backend-thresh",
        "18.0",
        "--image-height",
        "240",
        "--image-width",
        "320",
    ]
    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)
    return output_dir / "dpvo_style_metrics_summary.csv"


def _build_screening_summary(
    *,
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    _write_csv(
        output_path,
        rows,
        fieldnames=["candidate_id", "checkpoint_path", "frontend_config_path", "mean_ate_rmse", "mean_coverage", "output_dir"],
    )


def _build_winner_repeat_summary(
    *,
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    _write_csv(
        output_path,
        rows,
        fieldnames=["repeat_id", "candidate_id", "mean_ate_rmse", "mean_coverage", "output_dir"],
    )


def _paper_comparison_rows(
    *,
    old_rows: list[dict[str, str]],
    new_rows: list[dict[str, str]],
    median_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    old_map = {row["sequence"]: _safe_float(row["ate_rmse"]) for row in old_rows}
    new_map = {row["sequence"]: _safe_float(row["ate_rmse"]) for row in new_rows}
    median_map = {str(row["sequence"]): _safe_float(row["median_ate_rmse"]) for row in median_rows}
    rows: list[dict[str, object]] = []
    for sequence in TEST_SPLIT:
        rows.append(
            {
                "sequence": sequence,
                "old_best_ate_rmse": old_map.get(sequence, math.nan),
                "new_best_ate_rmse": new_map.get(sequence, math.nan),
                "winner_median_ate_rmse": median_map.get(sequence, math.nan),
                "paper_ours_default": PAPER_OURS_DEFAULT[sequence],
                "paper_ours_fast": PAPER_OURS_FAST[sequence],
                "paper_droid_vo": PAPER_DROID_VO[sequence],
            }
        )
    return rows


def _mean_from_dict_rows(rows: list[dict[str, object]], key: str) -> float:
    values = [_safe_float(row.get(key)) for row in rows if math.isfinite(_safe_float(row.get(key)))]
    if not values:
        return math.nan
    return float(sum(values) / len(values))


def _write_paper_comparison_md(
    *,
    output_path: Path,
    comparison_rows: list[dict[str, object]],
    winner_candidate_id: str,
) -> None:
    old_mean = _mean_from_dict_rows(comparison_rows, "old_best_ate_rmse")
    new_mean = _mean_from_dict_rows(comparison_rows, "new_best_ate_rmse")
    winner_mean = _mean_from_dict_rows(comparison_rows, "winner_median_ate_rmse")
    paper_default_mean = _mean_from_dict_rows(comparison_rows, "paper_ours_default")
    paper_fast_mean = _mean_from_dict_rows(comparison_rows, "paper_ours_fast")
    paper_droid_mean = _mean_from_dict_rows(comparison_rows, "paper_droid_vo")

    lines = [
        "# Focus071 TartanAir Mono Paper Comparison",
        "",
        f"Winner after screening: `{winner_candidate_id}`",
        "",
        "## Mean ATE (scale aligned)",
        "",
        f"- old best single run: `{old_mean:.6f}`",
        f"- new best single run: `{new_mean:.6f}`",
        f"- winner 5-run median: `{winner_mean:.6f}`",
        f"- paper Ours (Default): `{paper_default_mean:.6f}`",
        f"- paper Ours (Fast): `{paper_fast_mean:.6f}`",
        f"- paper DROID-VO: `{paper_droid_mean:.6f}`",
        "",
        "## Per-sequence table",
        "",
        "| Sequence | Old Best | New Best | Winner Median | Paper Default | Paper Fast | DROID-VO |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| `{row['sequence']}` | "
            f"{_safe_float(row['old_best_ate_rmse']):.2f} | "
            f"{_safe_float(row['new_best_ate_rmse']):.2f} | "
            f"{_safe_float(row['winner_median_ate_rmse']):.2f} | "
            f"{_safe_float(row['paper_ours_default']):.2f} | "
            f"{_safe_float(row['paper_ours_fast']):.2f} | "
            f"{_safe_float(row['paper_droid_vo']):.2f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_dry_run(repo_root: Path, data_root: Path, output_root: Path, candidates: list[Candidate]) -> None:
    print(f"repo_root: {repo_root}")
    print(f"data_root: {data_root}")
    print(f"output_root: {output_root}")
    print("candidates:")
    for candidate in candidates:
        print(f"  - {candidate.candidate_id}: checkpoint={candidate.checkpoint} config={candidate.frontend_config}")
    print(f"test_split: {','.join(TEST_SPLIT)}")


def main() -> None:
    repo_root = _repo_root()
    ap = argparse.ArgumentParser(description="Run the Focus071 TartanAir mono paper evaluation.")
    ap.add_argument("--data-root", default=str(repo_root / "refocus_vo" / "data" / "tartanair_test"))
    ap.add_argument("--output-root", default=str(repo_root / "refocus_vo" / "runs" / "eval" / "tartanair_mono_paper_focus071_v1"))
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    candidates = _default_candidates(repo_root)
    for candidate in candidates:
        if not candidate.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {candidate.checkpoint}")
        if not candidate.frontend_config.exists():
            raise FileNotFoundError(f"Frontend config not found: {candidate.frontend_config}")

    if args.dry_run:
        _print_dry_run(repo_root, data_root, output_root, candidates)
        return

    mono_root, gt_root = _ensure_dataset(data_root)
    _wait_for_training_idle(int(args.poll_seconds))

    smoke_dir = output_root / "smoke_me000"
    _run_eval(
        repo_root=repo_root,
        candidate=candidates[0],
        dataset_root=mono_root,
        groundtruth_root=gt_root,
        output_dir=smoke_dir,
        sequences=["ME000"],
    )

    screening_dir = output_root / "screening"
    screening_rows: list[dict[str, object]] = []
    screening_results: dict[str, list[dict[str, str]]] = {}
    for candidate in candidates:
        candidate_dir = screening_dir / candidate.candidate_id
        csv_path = _run_eval(
            repo_root=repo_root,
            candidate=candidate,
            dataset_root=mono_root,
            groundtruth_root=gt_root,
            output_dir=candidate_dir,
            sequences=TEST_SPLIT,
        )
        rows = _read_dpvo_style_csv(csv_path)
        screening_results[candidate.candidate_id] = rows
        screening_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "checkpoint_path": str(candidate.checkpoint),
                "frontend_config_path": str(candidate.frontend_config),
                "mean_ate_rmse": _mean_ate_from_rows(rows),
                "mean_coverage": _mean_coverage_from_rows(rows),
                "output_dir": str(candidate_dir),
            }
        )
    _build_screening_summary(rows=screening_rows, output_path=output_root / "screening_summary.csv")

    screening_rows_sorted = sorted(screening_rows, key=lambda row: (_safe_float(row["mean_ate_rmse"]), str(row["candidate_id"])))
    winner_candidate_id = str(screening_rows_sorted[0]["candidate_id"])
    winner = next(candidate for candidate in candidates if candidate.candidate_id == winner_candidate_id)

    repeat_rows: list[dict[str, object]] = []
    repeat_metrics: list[list[dict[str, str]]] = []
    for repeat_idx in range(1, 6):
        repeat_id = f"winner_repeat_{repeat_idx:02d}"
        repeat_dir = output_root / repeat_id
        csv_path = _run_eval(
            repo_root=repo_root,
            candidate=winner,
            dataset_root=mono_root,
            groundtruth_root=gt_root,
            output_dir=repeat_dir,
            sequences=TEST_SPLIT,
        )
        rows = _read_dpvo_style_csv(csv_path)
        repeat_metrics.append(rows)
        repeat_rows.append(
            {
                "repeat_id": repeat_id,
                "candidate_id": winner_candidate_id,
                "mean_ate_rmse": _mean_ate_from_rows(rows),
                "mean_coverage": _mean_coverage_from_rows(rows),
                "output_dir": str(repeat_dir),
            }
        )
    _build_winner_repeat_summary(rows=repeat_rows, output_path=output_root / "winner_repeat_summary.csv")

    median_rows = _median_per_sequence(repeat_metrics)
    _write_csv(
        output_root / "winner_median_per_sequence.csv",
        median_rows,
        fieldnames=["sequence", "median_ate_rmse"],
    )

    comparison_rows = _paper_comparison_rows(
        old_rows=screening_results["old_best"],
        new_rows=screening_results["new_best"],
        median_rows=median_rows,
    )
    _write_csv(
        output_root / "paper_comparison.csv",
        comparison_rows,
        fieldnames=[
            "sequence",
            "old_best_ate_rmse",
            "new_best_ate_rmse",
            "winner_median_ate_rmse",
            "paper_ours_default",
            "paper_ours_fast",
            "paper_droid_vo",
        ],
    )
    _write_paper_comparison_md(
        output_path=output_root / "paper_comparison.md",
        comparison_rows=comparison_rows,
        winner_candidate_id=winner_candidate_id,
    )


if __name__ == "__main__":
    main()
