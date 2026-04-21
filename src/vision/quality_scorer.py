"""
Frame quality scoring with face-aware signals.

Combines:
- Laplacian sharpness (baseline)
- Brightness / contrast (baseline)
- Face quality: blink detection, expression quality (Phase 1)
- TOPIQ aesthetic score (Phase 5 - optional)

Usage:
    scorer = get_quality_scorer()
    score = scorer.score(frame_bgr, face_detections)
"""

from dataclasses import dataclass
from typing import List, Optional, Protocol
import cv2
import numpy as np
import logging

from .face_detector import FaceDetection

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Detailed quality score breakdown."""
    total: float  # Combined score 0-1
    sharpness: float  # Laplacian variance normalized
    brightness: float  # Brightness quality 0-1
    contrast: float  # Contrast quality 0-1
    face_quality: Optional[float] = None  # Face sub-score 0-1
    aesthetic: Optional[float] = None  # TOPIQ score (Phase 5)


class QualityScorerProtocol(Protocol):
    """Protocol for quality scorer implementations."""
    def score(
        self,
        frame: np.ndarray,
        face_detections: Optional[List[FaceDetection]] = None,
    ) -> QualityScore: ...


class HeuristicQualityScorer:
    """Quality scoring using traditional CV heuristics + face-aware signals.

    Face quality sub-scorer:
    - Penalizes blinks (eye aspect ratio too flat)
    - Penalizes mid-word expressions (mouth too open)
    - Rewards centered, large faces
    """

    # Weight configuration
    WEIGHT_SHARPNESS = 0.4
    WEIGHT_BRIGHTNESS = 0.15
    WEIGHT_CONTRAST = 0.15
    WEIGHT_FACE = 0.3  # Only used when faces present

    # Face quality thresholds
    EAR_BLINK_THRESHOLD = 0.15  # Below this = likely blinking
    MAR_OPEN_THRESHOLD = 0.08   # Above this = mouth too open (mid-word)

    def score(
        self,
        frame: np.ndarray,
        face_detections: Optional[List[FaceDetection]] = None,
    ) -> QualityScore:
        """Score frame quality.

        Args:
            frame: BGR numpy array.
            face_detections: Optional face detection results for face-aware scoring.

        Returns:
            QualityScore with breakdown.
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(laplacian_var / 1000.0, 1.0)

        # Brightness (prefer ~50%)
        brightness_raw = np.mean(gray) / 255.0
        brightness = 1.0 - abs(brightness_raw - 0.5) * 2

        # Contrast (standard deviation)
        contrast = min(np.std(gray) / 128.0, 1.0)

        # Face quality sub-score
        face_quality = None
        if face_detections:
            face_quality = self._score_face_quality(face_detections)

        # Combine scores
        if face_quality is not None:
            total = (
                sharpness * self.WEIGHT_SHARPNESS
                + brightness * self.WEIGHT_BRIGHTNESS
                + contrast * self.WEIGHT_CONTRAST
                + face_quality * self.WEIGHT_FACE
            )
        else:
            # No faces: redistribute face weight to sharpness
            total = (
                sharpness * (self.WEIGHT_SHARPNESS + self.WEIGHT_FACE)
                + brightness * self.WEIGHT_BRIGHTNESS
                + contrast * self.WEIGHT_CONTRAST
            )

        total = max(0.0, min(1.0, total))

        return QualityScore(
            total=total,
            sharpness=sharpness,
            brightness=brightness,
            contrast=contrast,
            face_quality=face_quality,
        )

    def _score_face_quality(self, detections: List[FaceDetection]) -> float:
        """Score face quality from detection results.

        Evaluates:
        - Size: larger face area = better thumbnail
        - Blink: penalize closed/squinting eyes
        - Expression: penalize mid-word open mouth
        - Confidence: higher detection confidence = clearer face
        """
        if not detections:
            return 0.0

        # Use the largest face (primary subject)
        primary = max(detections, key=lambda d: d.face_area_ratio)

        # Base: confidence score
        score = primary.confidence

        # Size bonus: larger faces are better thumbnails
        size_bonus = min(primary.face_area_ratio * 5, 0.3)
        score += size_bonus

        # Blink penalty
        ear = primary.eye_aspect_ratio
        if ear is not None and ear < self.EAR_BLINK_THRESHOLD:
            score *= 0.5  # Heavy penalty for blinks

        # Mid-word penalty
        mar = primary.mouth_aspect_ratio
        if mar is not None and mar > self.MAR_OPEN_THRESHOLD:
            score *= 0.7  # Moderate penalty for open mouth

        return max(0.0, min(1.0, score))


class TOPIQScorer:
    """Quality scoring using TOPIQ learned aesthetic model + face-aware signals.

    Final score = TOPIQ aesthetic (0.7) + face quality (0.2) + Laplacian sanity check (0.1)

    TOPIQ is trained on millions of human quality ratings and scores visual
    aesthetics (composition, color harmony) -- not just technical quality.
    ~10ms/frame on GPU, ~80MB model.
    """

    WEIGHT_AESTHETIC = 0.7
    WEIGHT_FACE = 0.2
    WEIGHT_SHARPNESS = 0.1

    def __init__(self):
        import pyiqa
        import torch

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading TOPIQ IAA model on %s...", self._device)
        self._metric = pyiqa.create_metric('topiq_iaa', device=self._device)
        self._heuristic = HeuristicQualityScorer()
        logger.info("TOPIQ scorer ready")

    def score(
        self,
        frame: np.ndarray,
        face_detections: Optional[List[FaceDetection]] = None,
    ) -> QualityScore:
        """Score frame using TOPIQ aesthetic + face quality + sharpness sanity.

        Args:
            frame: BGR numpy array.
            face_detections: Optional face detections for face quality sub-score.

        Returns:
            QualityScore with aesthetic field populated.
        """
        import torch
        from PIL import Image

        # Get heuristic sub-scores
        heuristic = self._heuristic.score(frame, face_detections)

        # TOPIQ aesthetic score
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # pyiqa expects a tensor or path
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
            ])
            tensor = transform(pil_img).unsqueeze(0).to(self._device)

            with torch.no_grad():
                aesthetic_raw = self._metric(tensor).item()

            # TOPIQ IAA outputs ~0-1 range
            aesthetic = max(0.0, min(1.0, aesthetic_raw))
        except Exception as e:
            logger.warning("TOPIQ scoring failed: %s, using heuristic", e)
            return heuristic

        # Combine: aesthetic (70%) + face quality (20%) + sharpness sanity (10%)
        face_quality = heuristic.face_quality if heuristic.face_quality is not None else aesthetic

        total = (
            aesthetic * self.WEIGHT_AESTHETIC
            + face_quality * self.WEIGHT_FACE
            + heuristic.sharpness * self.WEIGHT_SHARPNESS
        )
        total = max(0.0, min(1.0, total))

        return QualityScore(
            total=total,
            sharpness=heuristic.sharpness,
            brightness=heuristic.brightness,
            contrast=heuristic.contrast,
            face_quality=heuristic.face_quality,
            aesthetic=aesthetic,
        )


# Singleton cache
_scorer_instance: Optional[QualityScorerProtocol] = None


def get_quality_scorer(backend: str = "auto") -> QualityScorerProtocol:
    """Get quality scorer instance (singleton).

    Args:
        backend: "heuristic", "topiq", or "auto" (try TOPIQ, fall back to heuristic).

    Returns:
        Quality scorer implementing QualityScorerProtocol.
    """
    global _scorer_instance

    if _scorer_instance is not None:
        return _scorer_instance

    if backend == "auto":
        try:
            _scorer_instance = TOPIQScorer()
            return _scorer_instance
        except ImportError:
            logger.info("pyiqa not available, using heuristic quality scorer")
            _scorer_instance = HeuristicQualityScorer()
            return _scorer_instance
        except Exception as e:
            logger.warning("TOPIQ init failed (%s), using heuristic", e)
            _scorer_instance = HeuristicQualityScorer()
            return _scorer_instance

    if backend == "heuristic":
        _scorer_instance = HeuristicQualityScorer()
    elif backend == "topiq":
        _scorer_instance = TOPIQScorer()
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return _scorer_instance
