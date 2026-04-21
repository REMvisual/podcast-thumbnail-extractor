"""
Face detection with MediaPipe (primary) and Haar Cascade (fallback).

Usage:
    detector = get_face_detector()  # auto-selects best available
    detections = detector.detect(frame_bgr)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """Single face detection result."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    landmarks: Optional[List[Tuple[float, float]]] = None  # normalized (x, y)
    face_area_ratio: float = 0.0  # fraction of frame area

    @property
    def eye_aspect_ratio(self) -> Optional[float]:
        """Eye aspect ratio for blink detection. Returns None if no landmarks."""
        if not self.landmarks or len(self.landmarks) < 6:
            return None
        # MediaPipe face detection returns 6 keypoints:
        # 0: right eye, 1: left eye, 2: nose tip,
        # 3: mouth center, 4: right ear, 5: left ear
        right_eye = self.landmarks[0]
        left_eye = self.landmarks[1]
        # Approximate EAR from inter-eye distance vs face width
        eye_dist = np.sqrt(
            (right_eye[0] - left_eye[0]) ** 2 +
            (right_eye[1] - left_eye[1]) ** 2
        )
        # Normal eye dist is ~30% of face width; squinting reduces y-diff
        y_diff = abs(right_eye[1] - left_eye[1])
        return y_diff / max(eye_dist, 1e-6)

    @property
    def mouth_aspect_ratio(self) -> Optional[float]:
        """Mouth openness estimate. Returns None if no landmarks."""
        if not self.landmarks or len(self.landmarks) < 4:
            return None
        nose = self.landmarks[2]
        mouth = self.landmarks[3]
        # Distance from nose to mouth relative to face height
        return abs(mouth[1] - nose[1])


class FaceDetectorProtocol(Protocol):
    """Protocol for face detector implementations."""
    def detect(self, frame: np.ndarray) -> List[FaceDetection]: ...
    def release(self) -> None: ...


class MediaPipeFaceDetector:
    """Face detection using MediaPipe Face Detection.

    Advantages over Haar:
    - Handles side profiles up to ~75 degrees (vs ~30)
    - Returns confidence scores + 6 landmarks
    - <5ms/frame on CPU
    """

    def __init__(self, min_confidence: float = 0.5):
        """
        Args:
            min_confidence: Minimum detection confidence (0-1).
        """
        import mediapipe as mp

        # MediaPipe >= 0.10.14 uses tasks API
        base_options = mp.tasks.BaseOptions(
            model_asset_path=self._find_model_path()
        )
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_confidence,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        self._detector = mp.tasks.vision.FaceDetector.create_from_options(options)
        self._mp = mp
        logger.info("MediaPipe face detector initialized (tasks API, conf=%.2f)", min_confidence)

    @staticmethod
    def _find_model_path() -> str:
        """Find or download the BlazeFace model."""
        import os
        import mediapipe as mp

        # Check if bundled model exists in mediapipe package
        mp_dir = os.path.dirname(mp.__file__)
        candidates = [
            os.path.join(mp_dir, "modules", "face_detection", "face_detection_short_range.tflite"),
            os.path.join(mp_dir, "modules", "face_detection", "face_detection_full_range_sparse.tflite"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c

        # Download the model if not found
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "blaze_face_short_range.tflite")

        if not os.path.exists(model_path):
            logger.info("Downloading BlazeFace model...")
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
            urllib.request.urlretrieve(url, model_path)
            logger.info("BlazeFace model downloaded to %s", model_path)

        return model_path

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces in a BGR or RGB frame.

        Args:
            frame: numpy array (H, W, 3) in BGR or RGB format.

        Returns:
            List of FaceDetection results.
        """
        # MediaPipe tasks API expects RGB via mp.Image
        if len(frame.shape) == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w = frame.shape[:2]
        total_pixels = h * w

        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=rgb,
        )
        results = self._detector.detect(mp_image)
        detections = []

        for det in results.detections:
            bb = det.bounding_box
            x = bb.origin_x
            y = bb.origin_y
            bw = bb.width
            bh = bb.height

            # Clamp to frame bounds
            x = max(0, x)
            y = max(0, y)
            bw = min(bw, w - x)
            bh = min(bh, h - y)

            # Extract keypoints as normalized landmarks
            landmarks = []
            if det.keypoints:
                for kp in det.keypoints:
                    landmarks.append((kp.x, kp.y))

            face_area = bw * bh
            confidence = det.categories[0].score if det.categories else 0.0

            detections.append(FaceDetection(
                bbox=(x, y, bw, bh),
                confidence=confidence,
                landmarks=landmarks if landmarks else None,
                face_area_ratio=face_area / total_pixels,
            ))

        return detections

    def release(self) -> None:
        if hasattr(self, '_detector') and self._detector:
            self._detector.close()


class HaarCascadeFaceDetector:
    """Fallback face detector using OpenCV Haar Cascades.

    Used when MediaPipe is unavailable. Singleton cascade to avoid
    per-frame reinitialization (fixes bug in thumbnail_extractor_complete.py L383).
    """

    def __init__(self, min_size: Tuple[int, int] = (60, 60)):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self._cascade = cv2.CascadeClassifier(cascade_path)
        self._min_size = min_size
        if self._cascade.empty():
            logger.warning("Failed to load Haar cascade")
            self._cascade = None
        else:
            logger.info("Haar cascade face detector initialized (fallback)")

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        if self._cascade is None:
            return []

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        h, w = gray.shape[:2]
        total_pixels = h * w

        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self._min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        detections = []
        if len(faces) > 0:
            for (x, y, fw, fh) in faces:
                detections.append(FaceDetection(
                    bbox=(x, y, fw, fh),
                    confidence=0.7,  # Haar doesn't provide confidence
                    landmarks=None,
                    face_area_ratio=(fw * fh) / total_pixels,
                ))
        return detections

    def release(self) -> None:
        pass


# Singleton cache
_detector_instance: Optional[FaceDetectorProtocol] = None


def get_face_detector(backend: str = "auto") -> FaceDetectorProtocol:
    """Get face detector instance (singleton).

    Args:
        backend: "mediapipe", "haar", or "auto" (try mediapipe, fall back to haar).

    Returns:
        Face detector implementing FaceDetectorProtocol.
    """
    global _detector_instance

    if _detector_instance is not None:
        return _detector_instance

    if backend == "auto":
        try:
            _detector_instance = MediaPipeFaceDetector()
            return _detector_instance
        except ImportError:
            logger.warning("MediaPipe not available, falling back to Haar cascade")
            _detector_instance = HaarCascadeFaceDetector()
            return _detector_instance

    if backend == "mediapipe":
        _detector_instance = MediaPipeFaceDetector()
    elif backend == "haar":
        _detector_instance = HaarCascadeFaceDetector()
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return _detector_instance
