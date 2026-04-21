"""Tests for config_loader — JSON CRUD for categories."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config_loader import (  # noqa: E402
    CategoryError,
    delete_category,
    list_categories,
    load_config,
    save_category,
    update_category,
)


@pytest.fixture
def tmp_config(tmp_path):
    cfg = tmp_path / "categories.json"
    cfg.write_text(json.dumps({
        "version": 1,
        "categories": [
            {"id": "faces", "label": "Faces", "icon": "\U0001f60a", "detector": "face",
             "model_path": None, "training_data_path": None, "enabled": True, "builtin": True},
            {"id": "everything", "label": "Everything", "icon": "\U0001f3af", "detector": "none",
             "model_path": None, "training_data_path": None, "enabled": True, "builtin": True},
        ],
    }))
    return cfg


def test_load_config_returns_dict(tmp_config):
    cfg = load_config(tmp_config)
    assert cfg["version"] == 1
    assert len(cfg["categories"]) == 2


def test_list_categories_returns_list(tmp_config):
    cats = list_categories(tmp_config)
    assert len(cats) == 2
    assert cats[0]["id"] == "faces"


def test_save_new_category(tmp_config):
    new_cat = {
        "id": "firetrucks",
        "label": "Firetrucks",
        "icon": "\U0001f692",
        "detector": "quality",
        "model_path": "models/firetrucks.pth",
        "training_data_path": "training_data/firetrucks/",
        "enabled": True,
        "builtin": False,
    }
    save_category(tmp_config, new_cat)
    cats = list_categories(tmp_config)
    assert len(cats) == 3
    assert any(c["id"] == "firetrucks" for c in cats)


def test_save_category_rejects_duplicate_id(tmp_config):
    dup = {"id": "faces", "label": "Dup", "icon": "X", "detector": "face",
           "model_path": None, "training_data_path": None, "enabled": True, "builtin": False}
    with pytest.raises(CategoryError, match="already exists"):
        save_category(tmp_config, dup)


def test_save_category_rejects_invalid_detector(tmp_config):
    bad = {"id": "bad", "label": "Bad", "icon": "X", "detector": "rocket-science",
           "model_path": None, "training_data_path": None, "enabled": True, "builtin": False}
    with pytest.raises(CategoryError, match="detector"):
        save_category(tmp_config, bad)


def test_update_category_toggles_enabled(tmp_config):
    update_category(tmp_config, "faces", {"enabled": False})
    cats = list_categories(tmp_config)
    faces = next(c for c in cats if c["id"] == "faces")
    assert faces["enabled"] is False


def test_update_category_missing_id_raises(tmp_config):
    with pytest.raises(CategoryError, match="not found"):
        update_category(tmp_config, "nonexistent", {"enabled": False})


def test_delete_category_removes(tmp_config):
    save_category(tmp_config, {
        "id": "temp", "label": "Temp", "icon": "T", "detector": "none",
        "model_path": None, "training_data_path": None, "enabled": True, "builtin": False,
    })
    delete_category(tmp_config, "temp")
    cats = list_categories(tmp_config)
    assert not any(c["id"] == "temp" for c in cats)


def test_delete_category_refuses_builtin(tmp_config):
    with pytest.raises(CategoryError, match="builtin"):
        delete_category(tmp_config, "faces")


def test_load_config_path_from_env(tmp_config, monkeypatch):
    monkeypatch.setenv("THUMBNAIL_EXTRACTOR_CONFIG_PATH", str(tmp_config))
    cfg = load_config()
    assert cfg["version"] == 1
