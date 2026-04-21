"""
Model registry for auto-downloading and caching vision models.

Models are downloaded to HuggingFace cache on first use.
This module provides utilities for checking model availability
and managing disk usage.

Usage:
    info = get_model_info("clip")
    ensure_model_available("clip")
"""

from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a registered model."""
    name: str
    description: str
    size_mb: int
    pip_package: str
    auto_download: bool  # True = downloaded by library on first use


# Registry of models used by the vision library
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    "mediapipe_face": ModelInfo(
        name="MediaPipe Face Detection",
        description="Bundled with mediapipe package, no separate download needed",
        size_mb=30,
        pip_package="mediapipe>=0.10.9",
        auto_download=True,
    ),
    "clip_vit_b32": ModelInfo(
        name="OpenCLIP ViT-B/32",
        description="Zero-shot image classifier trained on 2B image-text pairs",
        size_mb=350,
        pip_package="open-clip-torch>=2.24.0",
        auto_download=True,
    ),
    "topiq_iaa": ModelInfo(
        name="TOPIQ Image Aesthetic Assessment",
        description="Learned aesthetic quality model trained on human ratings",
        size_mb=80,
        pip_package="pyiqa>=0.1.10",
        auto_download=True,
    ),
    "pyav_nvdec": ModelInfo(
        name="PyAV GPU Decoder (NVDEC)",
        description="Hardware-accelerated video decoding via NVDEC",
        size_mb=0,
        pip_package="av>=12.0.0",
        auto_download=False,
    ),
}


def get_model_info(model_key: str) -> Optional[ModelInfo]:
    """Get info about a registered model."""
    return MODEL_REGISTRY.get(model_key)


# Map pip package names to their actual import names
_IMPORT_MAP = {
    "mediapipe": "mediapipe",
    "open-clip-torch": "open_clip",
    "scenedetect": "scenedetect",
    "imagehash": "imagehash",
    "pyiqa": "pyiqa",
    "av": "av",
}


def check_dependency(pip_package: str) -> bool:
    """Check if a pip package is importable."""
    pkg_base = pip_package.split(">=")[0].split("==")[0]
    import_name = _IMPORT_MAP.get(pkg_base, pkg_base.replace("-", "_"))
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def get_status() -> Dict[str, Dict]:
    """Get status of all registered models."""
    status = {}
    for key, info in MODEL_REGISTRY.items():
        available = check_dependency(info.pip_package)
        status[key] = {
            "name": info.name,
            "available": available,
            "size_mb": info.size_mb,
            "package": info.pip_package,
        }
    return status
