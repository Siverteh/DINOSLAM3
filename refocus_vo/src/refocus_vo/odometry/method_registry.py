from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import yaml


SUBTREE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METHOD_MANIFEST = SUBTREE_ROOT / "configs" / "dino_guided_methods.yaml"


@dataclass(frozen=True)
class DinoGuidedMethodConfig:
    method_id: str
    feature_type: str
    layers: tuple[int, ...]
    weights: Dict[str, float]
    keep_ratio: float
    mask_policy: str
    coarse: str
    refine: str
    min_refine_pixels: int


def _coerce_method_config(raw: dict) -> DinoGuidedMethodConfig:
    method_id = str(raw["method_id"]).strip()
    feature_type = str(raw.get("feature_type") or method_id.upper())
    layers = tuple(int(v) for v in raw.get("layers", []))
    weights = {str(k): float(v) for k, v in dict(raw.get("weights") or {}).items()}
    return DinoGuidedMethodConfig(
        method_id=method_id,
        feature_type=feature_type,
        layers=layers,
        weights=weights,
        keep_ratio=float(raw["keep_ratio"]),
        mask_policy=str(raw.get("mask_policy", "topk")),
        coarse=str(raw.get("coarse", "hybrid")),
        refine=str(raw.get("refine", "point_to_plane")),
        min_refine_pixels=int(raw.get("min_refine_pixels", 4096)),
    )


def load_dino_method_registry(
    manifest_path: str | Path | None = None,
) -> dict[str, DinoGuidedMethodConfig]:
    manifest = Path(manifest_path or DEFAULT_METHOD_MANIFEST).expanduser().resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"DINO method manifest not found: {manifest}")

    payload = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    methods = payload.get("dino_methods") or []
    registry = {}
    for item in methods:
        cfg = _coerce_method_config(dict(item))
        registry[cfg.method_id] = cfg

    if "dino_guided" not in registry and "dino_balanced_611" in registry:
        registry["dino_guided"] = registry["dino_balanced_611"]
    return registry


def get_dino_method_config(
    method_id: str,
    manifest_path: str | Path | None = None,
) -> DinoGuidedMethodConfig:
    registry = load_dino_method_registry(manifest_path)
    try:
        return registry[str(method_id)]
    except KeyError as exc:
        known = ", ".join(sorted(registry))
        raise KeyError(f"Unknown DINO-guided method_id '{method_id}'. Known ids: {known}") from exc


def iter_registered_method_ids(
    manifest_path: str | Path | None = None,
) -> Iterable[str]:
    registry = load_dino_method_registry(manifest_path)
    for method_id in registry:
        if method_id == "dino_guided":
            continue
        yield method_id
