"""Vendored vision primitives — face detection, quality scoring, content classification."""

from .face_detector import FaceDetection, get_face_detector
from .quality_scorer import QualityScore, get_quality_scorer
from .frame_sampler import (
    SceneBoundary,
    SampledFrame,
    detect_scene_boundaries,
    adaptive_sample,
    deduplicate_frames,
)
from .content_classifier import ContentCategory, get_content_classifier
from .gpu_decoder import DecodedFrame, get_video_decoder
from .batch_pipeline import BatchProcessor, FrameResult

__all__ = [
    "FaceDetection",
    "get_face_detector",
    "QualityScore",
    "get_quality_scorer",
    "SceneBoundary",
    "SampledFrame",
    "detect_scene_boundaries",
    "adaptive_sample",
    "deduplicate_frames",
    "ContentCategory",
    "get_content_classifier",
    "DecodedFrame",
    "get_video_decoder",
    "BatchProcessor",
    "FrameResult",
]
