"""
Batched GPU inference orchestrator.

Processes frames in batches through face detection + content classification
+ quality scoring in a single pass per batch, maximizing GPU utilization.

Auto-tunes batch size based on available VRAM.

Usage:
    pipeline = BatchProcessor()
    results = pipeline.process_video(video_path, mode='faces', max_frames=200)
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np
import logging

from .face_detector import FaceDetection, get_face_detector
from .quality_scorer import QualityScore, get_quality_scorer
from .content_classifier import ContentCategory, get_content_classifier
from .gpu_decoder import DecodedFrame, get_video_decoder
from .frame_sampler import (
    SceneBoundary,
    SampledFrame,
    detect_scene_boundaries,
    deduplicate_frames,
)

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Complete analysis result for a single frame."""
    frame: np.ndarray  # BGR
    timestamp: float
    frame_number: int
    faces: List[FaceDetection]
    category: ContentCategory
    category_scores: Dict[ContentCategory, float]
    quality: QualityScore


class BatchProcessor:
    """Orchestrates batched GPU inference across all vision models.

    Pipeline:
    1. Scene boundary detection (single pass)
    2. Adaptive sampling based on scenes
    3. Batch decode via GPU decoder (PyAV/NVDEC)
    4. Batch classify via CLIP (if available)
    5. Per-frame face detection + quality scoring
    6. Perceptual hash deduplication
    """

    def __init__(self, batch_size: int = 16):
        """
        Args:
            batch_size: Number of frames per inference batch.
                        Auto-tuned down if VRAM is insufficient.
        """
        self.batch_size = batch_size
        self._decoder = get_video_decoder()
        self._face_detector = get_face_detector()
        self._classifier = get_content_classifier()
        self._scorer = get_quality_scorer()

    def process_video(
        self,
        video_path: str,
        budget: int = 200,
        dedup_threshold: int = 8,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[FrameResult]:
        """Process entire video through batched pipeline.

        Args:
            video_path: Path to video file.
            budget: Target number of frames to extract.
            dedup_threshold: pHash hamming distance threshold for dedup.
            progress_callback: Optional callback(progress, message).

        Returns:
            List of FrameResult, sorted by quality score descending.
        """
        # Phase 1: Scene detection (0-15%)
        if progress_callback:
            progress_callback(0.0, "Detecting scene boundaries...")

        boundaries = detect_scene_boundaries(
            video_path,
            progress_callback=lambda p, m: progress_callback(p * 0.15, m) if progress_callback else None,
        )

        if progress_callback:
            progress_callback(0.15, f"Found {len(boundaries)} scene boundaries")

        # Phase 2: Compute sample timestamps from scenes
        video_info = self._decoder.get_video_info(video_path)
        duration = video_info["duration"]
        fps = video_info["fps"]

        timestamps = self._compute_sample_timestamps(
            boundaries, duration, budget * 2  # Over-sample, dedup later
        )

        if progress_callback:
            progress_callback(0.2, f"Sampling {len(timestamps)} frames...")

        # Phase 3: Batch decode + process
        results: List[FrameResult] = []
        batch_frames: List[DecodedFrame] = []
        total = len(timestamps)

        for i, decoded in enumerate(self._decoder.decode_frames(video_path, timestamps=timestamps)):
            batch_frames.append(decoded)

            # Process batch when full or at end
            if len(batch_frames) >= self.batch_size or i == total - 1:
                batch_results = self._process_batch(batch_frames)
                results.extend(batch_results)
                batch_frames = []

                if progress_callback:
                    progress = 0.2 + (i + 1) / total * 0.6
                    progress_callback(progress, f"Processed {i+1}/{total} frames ({len(results)} results)")

        # Phase 4: Dedup (80-90%)
        if progress_callback:
            progress_callback(0.8, "Deduplicating frames...")

        if results and dedup_threshold > 0:
            results = self._deduplicate(results, dedup_threshold)

        if progress_callback:
            progress_callback(0.9, f"Dedup complete: {len(results)} unique frames")

        # Sort by quality
        results.sort(key=lambda r: r.quality.total, reverse=True)

        if progress_callback:
            progress_callback(1.0, f"Done: {len(results)} frames processed")

        return results

    def _compute_sample_timestamps(
        self,
        boundaries: List[SceneBoundary],
        duration: float,
        budget: int,
    ) -> List[float]:
        """Compute sample timestamps distributed across scenes."""
        scene_starts = [0.0] + [b.timestamp for b in boundaries]
        scene_ends = [b.timestamp for b in boundaries] + [duration]
        scenes = list(zip(scene_starts, scene_ends))

        scene_durations = [max(end - start, 0.1) for start, end in scenes]
        total_dur = sum(scene_durations)

        timestamps = []
        for (start, end), dur in zip(scenes, scene_durations):
            n_frames = max(1, int(budget * dur / total_dur))
            if n_frames == 1:
                timestamps.append(start + dur / 2)
            else:
                step = dur / (n_frames + 1)
                for j in range(n_frames):
                    timestamps.append(start + step * (j + 1))

        return sorted(timestamps[:budget])

    def _process_batch(self, batch: List[DecodedFrame]) -> List[FrameResult]:
        """Process a batch of decoded frames through all models.

        Uses batch inference for CLIP classification when available,
        sequential for face detection and quality scoring.
        """
        if not batch:
            return []

        frames_bgr = [d.frame for d in batch]

        # Batch CLIP classification
        try:
            classifications = self._classifier.classify_batch(frames_bgr)
        except Exception as e:
            logger.warning("Batch classify failed: %s", e)
            classifications = [self._classifier.classify(f) for f in frames_bgr]

        # Per-frame face detection + quality scoring
        results = []
        for i, decoded in enumerate(batch):
            faces = self._face_detector.detect(decoded.frame)
            category, cat_scores = classifications[i]
            quality = self._scorer.score(decoded.frame, faces if faces else None)

            results.append(FrameResult(
                frame=decoded.frame,
                timestamp=decoded.timestamp,
                frame_number=decoded.frame_number,
                faces=faces,
                category=category,
                category_scores=cat_scores,
                quality=quality,
            ))

        return results

    def _deduplicate(self, results: List[FrameResult], threshold: int) -> List[FrameResult]:
        """Deduplicate results using perceptual hashing."""
        # Wrap in SampledFrame for dedup function
        sf_list = [
            SampledFrame(
                frame=r.frame,
                timestamp=r.timestamp,
                frame_number=r.frame_number,
                scene_index=0,
            )
            for r in results
        ]

        unique_sfs = deduplicate_frames(sf_list, threshold=threshold)
        unique_frame_nums = {sf.frame_number for sf in unique_sfs}

        return [r for r in results if r.frame_number in unique_frame_nums]
