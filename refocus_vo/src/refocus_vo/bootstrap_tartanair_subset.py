from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import yaml

CANONICAL_TO_V2_ENV = {
    "abandonedfactory": "AbandonedFactory",
    "abandonedfactory_night": "AbandonedFactory2",
    "amusement": "AmusementPark",
    "carwelding": "CarWelding",
    "endofworld": "EndofTheWorld",
    "gascola": "Gascola",
    "hospital": "Hospital",
    "japanesealley": "JapaneseAlley",
    "neighborhood": "ModularNeighborhood",
    "ocean": "Ocean",
    "office": "Office",
    "office2": "RetroOffice",
    "oldtown": "OldTownNight",
    "seasidetown": "SeasideTown",
    "seasonsforest": "SeasonalForestSpring",
    "seasonsforest_winter": "SeasonalForestWinter",
    "soulcity": "SoulCity",
    "westerndesert": "WesternDesertTown",
}

V2_TO_CANONICAL_ENV = {v: k for k, v in CANONICAL_TO_V2_ENV.items()}

CANONICAL_TO_HF_ENV = {
    "abandonedfactory": "abandonedfactory",
    "abandonedfactory_night": "abandonedfactory_night",
    "amusement": "amusement",
    "carwelding": "carwelding",
    "endofworld": "endofworld",
    "gascola": "gascola",
    "hospital": "hospital",
    "japanesealley": "japanesealley",
    "neighborhood": "neighborhood",
    "ocean": "ocean",
    "office": "office",
    "office2": "office2",
    "oldtown": "oldtown",
    "seasidetown": "seasidetown",
    "seasonsforest": "seasonsforest",
    "seasonsforest_winter": "seasonsforest_winter",
    "soulcity": "soulcity",
    "westerndesert": "westerndesert",
}

HF_BASE_URL = "https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main"


def _try_tartanair_v2_download(
    raw_root: Path,
    subset_cfg: dict,
) -> None:
    try:
        import tartanair as ta  # type: ignore
    except Exception as exc:
        raise RuntimeError("tartanair package is not installed") from exc

    envs = [
        CANONICAL_TO_V2_ENV.get(str(env), str(env))
        for env in subset_cfg.get("subset", {}).get("environments", [])
    ]
    diffs = [str(v).lower() for v in subset_cfg.get("subset", {}).get("difficulties", ["Easy", "Hard"])]
    requested_modalities = [str(mod) for mod in subset_cfg.get("modalities", ["image", "depth"])]
    camera_name = [str(subset_cfg.get("camera_name", "lcam_left"))]

    ta.init(str(raw_root))
    available = ta.get_all_data()
    valid_envs = set(available.get("env", []))
    valid_modalities = set(available.get("modality", []))
    valid_difficulties = {str(v).lower() for v in available.get("difficulty", [])}
    envs = [env for env in envs if env in valid_envs]
    diffs = [diff for diff in diffs if diff in valid_difficulties]
    modalities = [mod for mod in requested_modalities if mod in valid_modalities]
    ignored_modalities = [mod for mod in requested_modalities if mod not in valid_modalities]
    if ignored_modalities:
        print(
            "[bootstrap_tartanair_subset] WARNING: ignoring unsupported V2 modalities: "
            + ", ".join(ignored_modalities)
        )
    if not envs:
        raise RuntimeError("no valid TartanAir V2 environments remain after mapping/filtering")
    if not modalities:
        raise RuntimeError("no valid TartanAir V2 modalities remain after filtering")
    if not diffs:
        raise RuntimeError("no valid TartanAir V2 difficulties remain after filtering")
    try:
        ta.download(
            env=envs,
            difficulty=diffs,
            modality=modalities,
            camera_name=camera_name,
            unzip=True,
        )
    except TypeError:
        ta.download(
            env=envs,
            difficulty=diffs,
            modality=modalities,
            camera_name=camera_name,
        )


def _hf_env_name(name: str) -> str:
    return CANONICAL_TO_HF_ENV.get(str(name), str(name).strip().lower())


def _download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    with urllib.request.urlopen(url) as response, dst.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _extract_zip(src_zip: Path, dst_dir: Path) -> None:
    marker = src_zip.with_suffix(src_zip.suffix + ".extracted")
    if marker.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(src_zip) as archive:
        archive.extractall(dst_dir)
    marker.write_text("ok\n", encoding="utf-8")


def _try_hf_download(raw_root: Path, subset_cfg: dict) -> None:
    subset = subset_cfg.get("subset", {})
    envs = [_hf_env_name(env) for env in subset.get("environments", [])]
    difficulties = [str(diff).strip().capitalize() for diff in subset.get("difficulties", ["Easy", "Hard"])]
    requested_modalities = [str(mod) for mod in subset_cfg.get("modalities", ["image", "depth"])]
    modalities = [mod for mod in requested_modalities if mod in {"image", "depth"}]
    ignored_modalities = [mod for mod in requested_modalities if mod not in {"image", "depth"}]
    if ignored_modalities:
        print(
            "[bootstrap_tartanair_subset] WARNING: HF fallback only supports image/depth; "
            "ignoring: " + ", ".join(ignored_modalities)
        )
    if not envs:
        raise RuntimeError("no environments configured for HF fallback")
    if not modalities:
        raise RuntimeError("no downloadable modalities configured for HF fallback")

    download_root = raw_root / "_hf_downloads"
    for env in envs:
        for difficulty in difficulties:
            for modality in modalities:
                archive_name = f"{modality}_left.zip"
                url = f"{HF_BASE_URL}/{env}/{difficulty}/{archive_name}?download=true"
                local_zip = download_root / env / difficulty / archive_name
                print(f"[bootstrap_tartanair_subset] HF download: {env}/{difficulty}/{archive_name}")
                _download_file(url, local_zip)
                _extract_zip(local_zip, raw_root)


def _symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst)
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _canonical_env_name(name: str) -> str:
    if name in V2_TO_CANONICAL_ENV:
        return V2_TO_CANONICAL_ENV[name]
    lowered = str(name).strip().lower().replace("-", "").replace(" ", "")
    return lowered


def _canonical_difficulty(name: str) -> str:
    raw = str(name).strip()
    lowered = raw.lower()
    if lowered in {"data_easy", "easy"}:
        return "Easy"
    if lowered in {"data_hard", "hard"}:
        return "Hard"
    return raw


def _resolve_v2_paths(sequence_dir: Path) -> tuple[Path | None, Path | None, Path | None]:
    image_candidates = [
        sequence_dir / "image_left",
        sequence_dir / "image_lcam_left",
        sequence_dir / "image" / "lcam_left",
        sequence_dir / "image" / "left",
    ]
    depth_candidates = [
        sequence_dir / "depth_left",
        sequence_dir / "depth_lcam_left",
        sequence_dir / "depth" / "lcam_left",
        sequence_dir / "depth" / "left",
    ]
    pose_candidates = [
        sequence_dir / "pose_left.txt",
        sequence_dir / "pose_lcam_left.txt",
        sequence_dir / "meta" / "pose_lcam_left.txt",
        sequence_dir / "meta" / "pose_left.txt",
        sequence_dir / "pose.txt",
    ]
    image_dir = next((p for p in image_candidates if p.exists()), None)
    depth_dir = next((p for p in depth_candidates if p.exists()), None)
    pose_file = next((p for p in pose_candidates if p.exists()), None)
    return image_dir, depth_dir, pose_file


def _select_sequence_relpaths(all_relpaths: list[tuple[str, ...]], subset_cfg: dict) -> set[tuple[str, ...]]:
    subset = subset_cfg.get("subset", {})
    max_per_group = int(subset.get("max_trajectories_per_env_difficulty", 0) or 0)
    total_target = int(subset.get("total_target_trajectories", 0) or 0)
    selected: set[tuple[str, ...]] = set()
    grouped: dict[tuple[str, str], list[tuple[str, ...]]] = defaultdict(list)
    for rel_parts in sorted(all_relpaths):
        if len(rel_parts) < 4:
            continue
        env_name, _, difficulty_name, trajectory = rel_parts[:4]
        grouped[(env_name, difficulty_name)].append((env_name, env_name, difficulty_name, trajectory))

    for key in sorted(grouped):
        entries = grouped[key]
        if max_per_group > 0:
            entries = entries[:max_per_group]
        for rel_parts in entries:
            if total_target > 0 and len(selected) >= total_target:
                return selected
            selected.add(rel_parts)
    return selected if selected else set(all_relpaths)


def convert_existing_tree(raw_root: Path, converted_root: Path, subset_cfg: dict) -> list[str]:
    converted_root.mkdir(parents=True, exist_ok=True)
    discovered_relpaths: list[tuple[str, ...]] = []
    for pose_file in sorted(raw_root.glob("*/*/*/*/pose_left.txt")):
        seq_dir = pose_file.parent
        image_dir = seq_dir / "image_left"
        depth_dir = seq_dir / "depth_left"
        if not image_dir.exists() or not depth_dir.exists():
            continue
        rel = seq_dir.relative_to(raw_root)
        discovered_relpaths.append(tuple(rel.parts))

    for candidate in sorted(raw_root.glob("*/*/*")):
        if not candidate.is_dir():
            continue
        parts = candidate.relative_to(raw_root).parts
        if len(parts) < 3:
            continue
        env_raw, difficulty_raw, trajectory = parts[:3]
        canonical_env = _canonical_env_name(env_raw)
        canonical_difficulty = _canonical_difficulty(difficulty_raw)
        rel_parts = (canonical_env, canonical_env, canonical_difficulty, trajectory)
        image_dir, depth_dir, pose_file = _resolve_v2_paths(candidate)
        if image_dir is None or depth_dir is None or pose_file is None:
            continue
        discovered_relpaths.append(rel_parts)

    selected_relpaths = _select_sequence_relpaths(sorted(set(discovered_relpaths)), subset_cfg)
    converted = []
    for rel_parts in sorted(selected_relpaths):
        rel_path = Path(*rel_parts)
        out_dir = converted_root / rel_path
        if (out_dir / "pose_left.txt").exists():
            converted.append("/".join(rel_parts))
            continue
        candidate = raw_root / rel_path
        image_dir = candidate / "image_left"
        depth_dir = candidate / "depth_left"
        pose_file = candidate / "pose_left.txt"
        if not image_dir.exists() or not depth_dir.exists() or not pose_file.exists():
            v1_candidate = raw_root / rel_parts[0] / rel_parts[2] / rel_parts[3]
            image_dir = v1_candidate / "image_left"
            depth_dir = v1_candidate / "depth_left"
            pose_file = v1_candidate / "pose_left.txt"
        if not image_dir.exists() or not depth_dir.exists() or not pose_file.exists():
            v2_candidate = raw_root / rel_parts[0] / f"Data_{rel_parts[2].lower()}" / rel_parts[3]
            image_dir, depth_dir, pose_file = _resolve_v2_paths(v2_candidate)
        if image_dir is None or depth_dir is None or pose_file is None:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        _symlink_or_copy(image_dir, out_dir / "image_left")
        _symlink_or_copy(depth_dir, out_dir / "depth_left")
        _symlink_or_copy(pose_file, out_dir / "pose_left.txt")
        converted.append("/".join(rel_parts))
    return converted


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap a TartanAir subset and convert it to DPVO-style layout.")
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--converted-root", required=True)
    ap.add_argument("--subset-config", default=None)
    ap.add_argument("--source-api", default="auto", choices=["auto", "v2", "hf", "local_only"])
    args = ap.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    converted_root = Path(args.converted_root).expanduser().resolve()
    subset_cfg_path = Path(args.subset_config).expanduser().resolve() if args.subset_config else (
        Path(__file__).resolve().parents[2] / "configs" / "tartanair_subset_v1.yaml"
    )
    subset_cfg = yaml.safe_load(subset_cfg_path.read_text(encoding="utf-8")) or {}

    raw_root.mkdir(parents=True, exist_ok=True)
    converted_root.mkdir(parents=True, exist_ok=True)

    if args.source_api == "v2":
        _try_tartanair_v2_download(raw_root, subset_cfg)
    elif args.source_api == "hf":
        _try_hf_download(raw_root, subset_cfg)
    elif args.source_api == "auto":
        converted_probe = convert_existing_tree(raw_root, converted_root, subset_cfg)
        if not converted_probe:
            try:
                _try_hf_download(raw_root, subset_cfg)
            except Exception as exc:
                print(f"[bootstrap_tartanair_subset] WARNING: HF fallback download step skipped: {exc}")
        converted_probe = convert_existing_tree(raw_root, converted_root, subset_cfg)
        if not converted_probe:
            try:
                _try_tartanair_v2_download(raw_root, subset_cfg)
            except Exception as exc:
                print(f"[bootstrap_tartanair_subset] WARNING: tartanair V2 download step skipped: {exc}")

    converted = convert_existing_tree(raw_root, converted_root, subset_cfg)
    if not converted:
        help_lines = [
            f"No TartanAir sequences were found under '{raw_root}' and no converted sequences were produced.",
            "Install the official toolkit in the same Python env and rerun bootstrap:",
            "  /home/coder/DINOSLAM3/.venv_pyslam_integration_v2/bin/python -m pip install tartanair",
            "Then rerun:",
            "  refocus_vo/scripts/bootstrap_tartanair_subset.sh",
            "Or place an existing DPVO-style TartanAir subset under:",
            f"  {converted_root}",
        ]
        raise FileNotFoundError("\n".join(help_lines))
    manifest = {
        "raw_root": str(raw_root),
        "converted_root": str(converted_root),
        "converted_sequences": converted,
        "subset_config": str(subset_cfg_path),
    }
    manifest_path = converted_root / "subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"TartanAir bootstrap complete. Converted {len(converted)} sequences.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
