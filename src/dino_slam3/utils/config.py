from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml

def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if isinstance(cfg, dict) and cfg.get("base"):
        base_path = (path.parent / cfg["base"]).resolve()
        base_cfg = load_config(base_path)
        return _deep_update(base_cfg, {k: v for k, v in cfg.items() if k != "base"})
    return cfg

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
