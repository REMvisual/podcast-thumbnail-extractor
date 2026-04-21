"""Category config loader — CRUD over categories.json."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional


VALID_DETECTORS = {"face", "quality", "none"}

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "categories.json"


class CategoryError(ValueError):
    """Raised on invalid category operation."""


def _resolve_path(path: Optional[Path] = None) -> Path:
    if path is not None:
        return Path(path)
    env_path = os.environ.get("THUMBNAIL_EXTRACTOR_CONFIG_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_CONFIG_PATH


def load_config(path: Optional[Path] = None) -> dict[str, Any]:
    p = _resolve_path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _save_config(path: Path, cfg: dict[str, Any]) -> None:
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def list_categories(path: Optional[Path] = None) -> list[dict[str, Any]]:
    return load_config(path).get("categories", [])


def _validate_category(cat: dict[str, Any]) -> None:
    required = {"id", "label", "icon", "detector", "model_path",
                "training_data_path", "enabled", "builtin"}
    missing = required - cat.keys()
    if missing:
        raise CategoryError(f"category missing fields: {sorted(missing)}")
    if cat["detector"] not in VALID_DETECTORS:
        raise CategoryError(
            f"invalid detector {cat['detector']!r} — must be one of {sorted(VALID_DETECTORS)}"
        )


def save_category(path: Optional[Path], category: dict[str, Any]) -> None:
    _validate_category(category)
    p = _resolve_path(path)
    cfg = load_config(p)
    if any(c["id"] == category["id"] for c in cfg["categories"]):
        raise CategoryError(f"category id {category['id']!r} already exists")
    cfg["categories"].append(category)
    _save_config(p, cfg)


def update_category(path: Optional[Path], category_id: str, changes: dict[str, Any]) -> None:
    p = _resolve_path(path)
    cfg = load_config(p)
    for c in cfg["categories"]:
        if c["id"] == category_id:
            c.update(changes)
            if "detector" in changes:
                _validate_category(c)
            _save_config(p, cfg)
            return
    raise CategoryError(f"category id {category_id!r} not found")


def delete_category(path: Optional[Path], category_id: str) -> None:
    p = _resolve_path(path)
    cfg = load_config(p)
    target = next((c for c in cfg["categories"] if c["id"] == category_id), None)
    if target is None:
        raise CategoryError(f"category id {category_id!r} not found")
    if target.get("builtin"):
        raise CategoryError(f"cannot delete builtin category {category_id!r}")
    cfg["categories"] = [c for c in cfg["categories"] if c["id"] != category_id]
    _save_config(p, cfg)
