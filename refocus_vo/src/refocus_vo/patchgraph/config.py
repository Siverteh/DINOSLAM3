from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PatchGraphConfig:
    method_id: str
    feature_type: str
    raw: dict[str, Any]

    @property
    def model(self) -> dict[str, Any]:
        return dict(self.raw.get("model", {}))

    @property
    def teacher(self) -> dict[str, Any]:
        return dict(self.raw.get("teacher", {}))

    @property
    def training(self) -> dict[str, Any]:
        return dict(self.raw.get("training", {}))

    @property
    def losses(self) -> dict[str, Any]:
        return dict(self.raw.get("losses", {}))

    @property
    def eval(self) -> dict[str, Any]:
        return dict(self.raw.get("eval", {}))


def load_patchgraph_config(path: str | Path) -> PatchGraphConfig:
    path = Path(path).expanduser().resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return PatchGraphConfig(
        method_id=str(data.get("method_id", "dino_patchgraph_vo_v1")),
        feature_type=str(data.get("feature_type", "DINO_PATCHGRAPH_VO_V1")),
        raw=data,
    )
