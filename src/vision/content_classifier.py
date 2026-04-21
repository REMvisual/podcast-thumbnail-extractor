"""
Content classification using OpenCLIP zero-shot (primary) or edge density (fallback).

Replaces custom ResNet50 binary classifiers with a single CLIP model that
generalizes to any software UI without training.

Usage:
    classifier = get_content_classifier()
    category, scores = classifier.classify(frame_bgr)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Protocol, Tuple
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ContentCategory(Enum):
    """Frame content categories."""
    UI_INTERFACE = "UI/interface"
    ART_RENDER = "art/render"
    TALKING_HEAD = "talking_head"
    TRANSITION = "transition"
    UNKNOWN = "unknown"


# Default CLIP text prompts for each category
DEFAULT_PROMPTS = {
    ContentCategory.UI_INTERFACE: [
        "a screenshot of a software interface with menus, toolbars, or node graphs",
        "a computer screen showing code, settings panels, or digital audio workstation",
        "a screenshot of a 3D modeling or visual effects application",
    ],
    ContentCategory.ART_RENDER: [
        "abstract digital art, 3D rendered scene, or generative visuals",
        "colorful artistic visualization or creative digital output",
        "a rendered 3D scene with lighting and materials",
    ],
    ContentCategory.TALKING_HEAD: [
        "a person speaking to camera in a podcast or interview",
        "a person's face and upper body in a studio or home office",
        "a webcam view of someone talking",
    ],
    ContentCategory.TRANSITION: [
        "a black screen, blank frame, or title card",
        "a solid color background with text overlay",
        "a dark or mostly black video frame",
    ],
}


class ContentClassifierProtocol(Protocol):
    """Protocol for content classifiers."""
    def classify(self, frame: np.ndarray) -> Tuple[ContentCategory, Dict[ContentCategory, float]]: ...
    def classify_batch(self, frames: List[np.ndarray]) -> List[Tuple[ContentCategory, Dict[ContentCategory, float]]]: ...


class CLIPClassifier:
    """Zero-shot content classifier using OpenCLIP.

    Maps frames to categories via cosine similarity with text prompts.
    ViT-B-32 model (~350MB) auto-downloaded to HuggingFace cache on first use.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        custom_prompts: Optional[Dict[ContentCategory, List[str]]] = None,
    ):
        import open_clip
        import torch

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading OpenCLIP %s (%s) on %s...", model_name, pretrained, self._device)
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self._model = self._model.to(self._device).eval()
        self._tokenizer = open_clip.get_tokenizer(model_name)

        # Pre-encode text prompts
        prompts = custom_prompts or DEFAULT_PROMPTS
        self._categories = list(prompts.keys())
        self._text_features = self._encode_prompts(prompts)

        logger.info("OpenCLIP classifier ready with %d categories", len(self._categories))

    def _encode_prompts(self, prompts: Dict[ContentCategory, List[str]]):
        """Pre-encode all text prompts into averaged embeddings per category."""
        import torch

        category_features = []
        for cat in self._categories:
            texts = prompts[cat]
            tokens = self._tokenizer(texts).to(self._device)
            with torch.no_grad():
                feats = self._model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                avg_feat = feats.mean(dim=0)
                avg_feat = avg_feat / avg_feat.norm()
            category_features.append(avg_feat)

        return torch.stack(category_features)

    def _preprocess_frame(self, frame: np.ndarray):
        """Convert BGR numpy frame to preprocessed tensor."""
        from PIL import Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        return self._preprocess(pil_img).unsqueeze(0).to(self._device)

    def classify(self, frame: np.ndarray) -> Tuple[ContentCategory, Dict[ContentCategory, float]]:
        """Classify a single frame.

        Args:
            frame: BGR numpy array.

        Returns:
            (best_category, {category: score}) where scores are cosine similarities.
        """
        import torch

        img_tensor = self._preprocess_frame(frame)

        with torch.no_grad():
            img_features = self._model.encode_image(img_tensor)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            similarities = (img_features @ self._text_features.T).squeeze(0)
            probs = similarities.softmax(dim=0)

        scores = {}
        for i, cat in enumerate(self._categories):
            scores[cat] = probs[i].item()

        best_idx = probs.argmax().item()
        return self._categories[best_idx], scores

    def classify_batch(self, frames: List[np.ndarray]) -> List[Tuple[ContentCategory, Dict[ContentCategory, float]]]:
        """Classify multiple frames in a single forward pass.

        Args:
            frames: List of BGR numpy arrays.

        Returns:
            List of (best_category, scores) per frame.
        """
        import torch

        if not frames:
            return []

        tensors = [self._preprocess_frame(f) for f in frames]
        batch = torch.cat(tensors, dim=0)

        with torch.no_grad():
            img_features = self._model.encode_image(batch)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            similarities = img_features @ self._text_features.T
            probs = similarities.softmax(dim=-1)

        results = []
        for i in range(len(frames)):
            scores = {}
            for j, cat in enumerate(self._categories):
                scores[cat] = probs[i][j].item()
            best_idx = probs[i].argmax().item()
            results.append((self._categories[best_idx], scores))

        return results


class EdgeDensityClassifier:
    """Fallback classifier using edge density and basic heuristics.

    Used when OpenCLIP is unavailable.
    """

    EDGE_DENSITY_SCREEN = 0.12
    EDGE_DENSITY_LOW = 0.03
    BRIGHTNESS_BLACK_THRESHOLD = 30

    def classify(self, frame: np.ndarray) -> Tuple[ContentCategory, Dict[ContentCategory, float]]:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        mean_brightness = np.mean(gray)

        scores = {cat: 0.0 for cat in ContentCategory}

        # Near-black frame = transition
        if mean_brightness < self.BRIGHTNESS_BLACK_THRESHOLD:
            scores[ContentCategory.TRANSITION] = 0.9
            return ContentCategory.TRANSITION, scores

        # High edge density = likely UI
        if edge_density > self.EDGE_DENSITY_SCREEN:
            scores[ContentCategory.UI_INTERFACE] = 0.5 + edge_density
            scores[ContentCategory.ART_RENDER] = 0.3
            return ContentCategory.UI_INTERFACE, scores

        # Low edge density = could be talking head or art
        if edge_density < self.EDGE_DENSITY_LOW:
            scores[ContentCategory.TALKING_HEAD] = 0.5
            return ContentCategory.TALKING_HEAD, scores

        scores[ContentCategory.UNKNOWN] = 0.5
        return ContentCategory.UNKNOWN, scores

    def classify_batch(self, frames: List[np.ndarray]) -> List[Tuple[ContentCategory, Dict[ContentCategory, float]]]:
        return [self.classify(f) for f in frames]


class ResNetLegacyClassifier:
    """Wrapper around the legacy ResNet50 models for backward compatibility.

    Wraps the existing ThumbnailQualityScorer .pth models so they can be
    used through the unified classifier interface during migration.
    """

    def __init__(self, model_paths: Optional[Dict[str, str]] = None):
        """
        Args:
            model_paths: Dict of model_type -> path, e.g. {"ui": "models/ui_model.pth"}.
        """
        self._models = {}
        self._transforms = {}
        self._device = None

        if model_paths:
            import torch
            from torchvision import models as tv_models, transforms
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            for model_type, path in model_paths.items():
                try:
                    from thumbnailextractor.thumbnail_extractor_complete import ThumbnailQualityScorer
                    model = ThumbnailQualityScorer().to(self._device)
                    checkpoint = torch.load(path, map_location=self._device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.eval()
                    self._models[model_type] = model
                    self._transforms[model_type] = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])
                    logger.info("Loaded legacy %s model from %s", model_type, path)
                except Exception as e:
                    logger.warning("Failed to load legacy %s model: %s", model_type, e)

    def classify(self, frame: np.ndarray) -> Tuple[ContentCategory, Dict[ContentCategory, float]]:
        """Classify using legacy models. Returns UI_INTERFACE or ART_RENDER based on scores."""
        if not self._models:
            return ContentCategory.UNKNOWN, {cat: 0.0 for cat in ContentCategory}

        from PIL import Image
        import torch

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        scores = {cat: 0.0 for cat in ContentCategory}

        for model_type, model in self._models.items():
            transform = self._transforms[model_type]
            tensor = transform(pil_img).unsqueeze(0).to(self._device)
            with torch.no_grad():
                score = model(tensor).item()

            if model_type == "ui":
                scores[ContentCategory.UI_INTERFACE] = score
            elif model_type == "art":
                scores[ContentCategory.ART_RENDER] = score

        best = max(scores, key=scores.get)
        return best, scores

    def classify_batch(self, frames: List[np.ndarray]) -> List[Tuple[ContentCategory, Dict[ContentCategory, float]]]:
        return [self.classify(f) for f in frames]


# Singleton cache
_classifier_instance: Optional[ContentClassifierProtocol] = None


def get_content_classifier(backend: str = "auto") -> ContentClassifierProtocol:
    """Get content classifier instance (singleton).

    Args:
        backend: "clip", "edge_density", "legacy", or "auto"
                 (auto tries CLIP -> edge_density fallback).

    Returns:
        Content classifier implementing ContentClassifierProtocol.
    """
    global _classifier_instance

    if _classifier_instance is not None:
        return _classifier_instance

    if backend == "auto":
        try:
            _classifier_instance = CLIPClassifier()
            return _classifier_instance
        except ImportError:
            logger.warning("OpenCLIP not available, falling back to edge density classifier")
            _classifier_instance = EdgeDensityClassifier()
            return _classifier_instance

    if backend == "clip":
        _classifier_instance = CLIPClassifier()
    elif backend == "edge_density":
        _classifier_instance = EdgeDensityClassifier()
    elif backend == "legacy":
        import os
        paths = {}
        for name in ["ui", "art"]:
            p = os.path.join("models", f"{name}_model.pth")
            if os.path.exists(p):
                paths[name] = p
        _classifier_instance = ResNetLegacyClassifier(model_paths=paths if paths else None)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return _classifier_instance
