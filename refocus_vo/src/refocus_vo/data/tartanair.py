from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


DPVO_VALIDATION_SPLIT = {
    "abandonedfactory/abandonedfactory/Easy/P011",
    "abandonedfactory/abandonedfactory/Hard/P011",
    "abandonedfactory_night/abandonedfactory_night/Easy/P013",
    "abandonedfactory_night/abandonedfactory_night/Hard/P014",
    "amusement/amusement/Easy/P008",
    "amusement/amusement/Hard/P007",
    "carwelding/carwelding/Easy/P007",
    "endofworld/endofworld/Easy/P009",
    "gascola/gascola/Easy/P008",
    "gascola/gascola/Hard/P009",
    "hospital/hospital/Easy/P036",
    "hospital/hospital/Hard/P049",
    "japanesealley/japanesealley/Easy/P007",
    "japanesealley/japanesealley/Hard/P005",
    "neighborhood/neighborhood/Easy/P021",
    "neighborhood/neighborhood/Hard/P017",
    "ocean/ocean/Easy/P013",
    "ocean/ocean/Hard/P009",
    "office2/office2/Easy/P011",
    "office2/office2/Hard/P010",
    "office/office/Hard/P007",
    "oldtown/oldtown/Easy/P007",
    "oldtown/oldtown/Hard/P008",
    "seasidetown/seasidetown/Easy/P009",
    "seasonsforest/seasonsforest/Easy/P011",
    "seasonsforest/seasonsforest/Hard/P006",
    "seasonsforest_winter/seasonsforest_winter/Easy/P009",
    "seasonsforest_winter/seasonsforest_winter/Hard/P018",
    "soulcity/soulcity/Easy/P012",
    "soulcity/soulcity/Hard/P009",
    "westerndesert/westerndesert/Easy/P013",
    "westerndesert/westerndesert/Hard/P007",
}

TARTANAIR_DEPTH_SCALE = 5.0

DEFAULT_TARTANAIR_SUBSET_ENVS = (
    "abandonedfactory",
    "abandonedfactory_night",
    "carwelding",
    "hospital",
    "office",
    "office2",
    "japanesealley",
    "neighborhood",
)


def _read_pose_file(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, dtype=np.float64)
    if poses.ndim == 1:
        poses = poses.reshape(1, -1)
    if poses.shape[1] < 7:
        raise ValueError(f"Expected pose file with 7 columns, got {poses.shape} at {path}")
    poses = poses[:, :7]
    # Match DPVO's TartanAir convention: convert NED ordering to xyz and scale
    # translation consistently with the paired depth maps.
    poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]
    poses[:, :3] /= TARTANAIR_DEPTH_SCALE
    return poses


def pose_vector_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    if pose.shape[0] != 7:
        raise ValueError(f"Expected 7D pose vector, got {pose.shape}")
    tx, ty, tz, qx, qy, qz, qw = pose
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = [tx, ty, tz]
    return T


def matrix_to_pose_vector(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    qx, qy, qz, qw = R.from_matrix(pose[:3, :3]).as_quat()
    return np.asarray([pose[0, 3], pose[1, 3], pose[2, 3], qx, qy, qz, qw], dtype=np.float64)


@dataclass
class TartanAirSequence:
    root: Path
    environment: str
    environment_repeat: str
    difficulty: str
    trajectory: str
    image_files: list[Path]
    depth_files: list[Path]
    poses: np.ndarray
    intrinsics: np.ndarray
    relative_dir: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.environment}/{self.environment_repeat}/{self.difficulty}/{self.trajectory}"

    @property
    def num_frames(self) -> int:
        return len(self.image_files)

    @property
    def sequence_dir(self) -> Path:
        return self.root.joinpath(*self.relative_dir)


def tartanair_intrinsics() -> np.ndarray:
    return np.asarray([320.0, 320.0, 320.0, 240.0], dtype=np.float32)


def scale_intrinsics(
    intrinsics: np.ndarray,
    *,
    src_height: int,
    src_width: int,
    dst_height: int,
    dst_width: int,
) -> np.ndarray:
    fx, fy, cx, cy = [float(v) for v in np.asarray(intrinsics, dtype=np.float32)]
    sx = float(dst_width) / float(src_width)
    sy = float(dst_height) / float(src_height)
    return np.asarray([fx * sx, fy * sy, cx * sx, cy * sy], dtype=np.float32)


def read_tartanair_rgb(path: str | Path, image_size: tuple[int, int] | None = None) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if image_size is not None:
        h, w = int(image_size[0]), int(image_size[1])
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img


def read_tartanair_depth(path: str | Path, image_size: tuple[int, int] | None = None) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        depth = np.load(p).astype(np.float32)
    else:
        depth = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(str(path))
        depth = depth.astype(np.float32)
    depth = depth / np.float32(TARTANAIR_DEPTH_SCALE)
    if image_size is not None:
        h, w = int(image_size[0]), int(image_size[1])
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    return depth


def discover_tartanair_sequences(
    root: str | Path,
    *,
    environments: tuple[str, ...] | list[str] | None = None,
    difficulties: tuple[str, ...] | list[str] = ("Easy", "Hard"),
    include_validation: bool = False,
) -> list[TartanAirSequence]:
    root = Path(root).expanduser().resolve()
    if environments is None:
        env_filter = set(str(v) for v in DEFAULT_TARTANAIR_SUBSET_ENVS)
    elif len(environments) == 0:
        env_filter = set()
    else:
        env_filter = set(str(v) for v in environments)
    diff_filter = set(str(v) for v in difficulties)
    sequences: list[TartanAirSequence] = []

    for image_dir in sorted(root.glob("**/image_left")):
        seq_dir = image_dir.parent
        parts = seq_dir.relative_to(root).parts
        if len(parts) == 4:
            environment, env_repeat, difficulty, trajectory = parts[:4]
            if environment != env_repeat:
                continue
        elif len(parts) == 3:
            environment, difficulty, trajectory = parts[:3]
            env_repeat = environment
        else:
            continue
        if env_filter and environment not in env_filter:
            continue
        if diff_filter and difficulty not in diff_filter:
            continue

        key = f"{environment}/{env_repeat}/{difficulty}/{trajectory}"
        if not include_validation and key in DPVO_VALIDATION_SPLIT:
            continue

        depth_dir = seq_dir / "depth_left"
        pose_file = seq_dir / "pose_left.txt"
        if not depth_dir.exists() or not pose_file.exists():
            continue

        image_files = sorted(image_dir.glob("*.png"))
        depth_files = sorted(depth_dir.glob("*.npy"))
        if not image_files or len(image_files) != len(depth_files):
            continue

        poses = _read_pose_file(pose_file)
        n = min(len(image_files), len(depth_files), poses.shape[0])
        if n < 4:
            continue

        sequences.append(
            TartanAirSequence(
                root=root,
                environment=environment,
                environment_repeat=env_repeat,
                difficulty=difficulty,
                trajectory=trajectory,
                image_files=image_files[:n],
                depth_files=depth_files[:n],
                poses=poses[:n],
                intrinsics=tartanair_intrinsics(),
                relative_dir=tuple(parts),
            )
        )

    return sequences


def select_patchgraph_training_sequences(
    root: str | Path,
    *,
    environments: tuple[str, ...] | list[str] = DEFAULT_TARTANAIR_SUBSET_ENVS,
    difficulties: tuple[str, ...] | list[str] = ("Easy", "Hard"),
    max_trajectories_per_env_difficulty: int = 1,
) -> list[TartanAirSequence]:
    candidates = discover_tartanair_sequences(
        root,
        environments=tuple(environments),
        difficulties=tuple(difficulties),
        include_validation=False,
    )
    grouped: dict[tuple[str, str], list[TartanAirSequence]] = {}
    for seq in candidates:
        grouped.setdefault((seq.environment, seq.difficulty), []).append(seq)

    selected: list[TartanAirSequence] = []
    for env in environments:
        for diff in difficulties:
            bucket = sorted(grouped.get((str(env), str(diff)), []), key=lambda s: s.trajectory)
            selected.extend(bucket[: int(max_trajectories_per_env_difficulty)])
    return selected


@dataclass
class TartanAirWindow:
    sequence_key: str
    frame_indices: tuple[int, ...]
    images: torch.Tensor
    depths: torch.Tensor
    poses: torch.Tensor
    intrinsics: torch.Tensor


class TartanAirWindowDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        subset_environments: tuple[str, ...] | list[str] = DEFAULT_TARTANAIR_SUBSET_ENVS,
        difficulties: tuple[str, ...] | list[str] = ("Easy", "Hard"),
        max_trajectories_per_env_difficulty: int = 1,
        n_frames: int = 4,
        image_size: tuple[int, int] = (240, 320),
        max_windows: int | None = None,
        dev_ratio: float = 0.1,
        seed: int = 13,
    ):
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        self.n_frames = int(n_frames)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.split = str(split)
        self.seed = int(seed)

        sequences = select_patchgraph_training_sequences(
            self.root,
            environments=tuple(subset_environments),
            difficulties=tuple(difficulties),
            max_trajectories_per_env_difficulty=int(max_trajectories_per_env_difficulty),
        )
        if not sequences:
            raise FileNotFoundError(
                f"No TartanAir training sequences found under {self.root}. "
                "Run the TartanAir bootstrap/conversion first."
            )
        self.sequences = sequences
        self.windows = self._build_windows(max_windows=max_windows, dev_ratio=float(dev_ratio))

    def _build_windows(self, *, max_windows: int | None, dev_ratio: float) -> list[tuple[int, tuple[int, ...]]]:
        windows: list[tuple[int, tuple[int, ...]]] = []
        for seq_idx, seq in enumerate(self.sequences):
            max_start = seq.num_frames - self.n_frames
            for start in range(max(0, max_start + 1)):
                frame_indices = tuple(range(start, start + self.n_frames))
                windows.append((seq_idx, frame_indices))

        rng = np.random.default_rng(self.seed)
        rng.shuffle(windows)
        if max_windows is not None:
            windows = windows[: int(max_windows)]

        if self.split == "all":
            return windows
        cutoff = int(round((1.0 - dev_ratio) * len(windows)))
        if self.split == "train":
            return windows[:cutoff]
        if self.split in {"dev", "val", "validation"}:
            return windows[cutoff:]
        raise ValueError(f"Unsupported split: {self.split}")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        seq_idx, frame_indices = self.windows[int(index)]
        seq = self.sequences[seq_idx]

        images = []
        depths = []
        poses = []
        for frame_idx in frame_indices:
            images.append(read_tartanair_rgb(seq.image_files[frame_idx], self.image_size))
            depths.append(read_tartanair_depth(seq.depth_files[frame_idx], self.image_size))
            poses.append(seq.poses[frame_idx])

        images_np = np.stack(images, axis=0).astype(np.float32) / 255.0
        depths_np = np.stack(depths, axis=0).astype(np.float32)
        poses_np = np.stack(poses, axis=0).astype(np.float64)
        scaled_intrinsics = scale_intrinsics(
            seq.intrinsics,
            src_height=480,
            src_width=640,
            dst_height=self.image_size[0],
            dst_width=self.image_size[1],
        )

        return {
            "sequence_key": seq.key,
            "frame_indices": np.asarray(frame_indices, dtype=np.int64),
            "images": torch.from_numpy(images_np).permute(0, 3, 1, 2).float(),
            "depths": torch.from_numpy(depths_np).float(),
            "poses": torch.from_numpy(poses_np).float(),
            "intrinsics": torch.from_numpy(scaled_intrinsics).float(),
        }
