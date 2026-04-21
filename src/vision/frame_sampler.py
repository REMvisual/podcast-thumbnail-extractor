"""
Intelligent temporal sampling + perceptual hash deduplication.

Three-pass pipeline:
  Pass 1: Scene boundary detection via histogram diff (PySceneDetect)
  Pass 2: Adaptive frame allocation per scene
  Pass 3: pHash deduplication

Usage:
    boundaries = detect_scene_boundaries(video_path)
    frames = adaptive_sample(video_path, boundaries, budget=200)
    unique = deduplicate_frames(frames, threshold=8)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class SceneBoundary:
    """Detected scene boundary."""
    timestamp: float  # seconds
    frame_number: int
    score: float  # histogram diff score


@dataclass
class SampledFrame:
    """A sampled frame with metadata."""
    frame: np.ndarray  # BGR numpy array
    timestamp: float
    frame_number: int
    scene_index: int  # which scene this belongs to
    phash: Optional[str] = None  # perceptual hash (set during dedup)


def detect_scene_boundaries(
    video_path: str | Path,
    threshold: float = 27.0,
    min_scene_len: int = 15,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> List[SceneBoundary]:
    """Detect scene boundaries using histogram difference.

    Wraps PySceneDetect ContentDetector for fast scene change detection.
    Falls back to uniform boundaries if PySceneDetect is unavailable.

    Args:
        video_path: Path to video file.
        threshold: Scene change threshold (lower = more sensitive).
        min_scene_len: Minimum frames between scene changes.
        progress_callback: Optional callback(progress, message).

    Returns:
        List of SceneBoundary at each scene change point.
    """
    video_path = str(video_path)

    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector

        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
        )

        if progress_callback:
            progress_callback(0.0, "Detecting scene boundaries...")

        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()

        boundaries = []
        for i, (start, end) in enumerate(scene_list):
            boundaries.append(SceneBoundary(
                timestamp=start.get_seconds(),
                frame_number=start.get_frames(),
                score=0.0,  # ContentDetector doesn't expose per-boundary scores
            ))

        if progress_callback:
            progress_callback(1.0, f"Found {len(boundaries)} scene boundaries")

        logger.info("Scene detection: %d boundaries in %s", len(boundaries), video_path)
        return boundaries

    except ImportError:
        logger.warning("PySceneDetect not available, using histogram fallback")
        return _histogram_fallback(video_path, threshold=0.3, progress_callback=progress_callback)


def _histogram_fallback(
    video_path: str,
    threshold: float = 0.3,
    progress_callback: Optional[Callable] = None,
) -> List[SceneBoundary]:
    """Simple histogram-diff scene detection without PySceneDetect."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Sample every 0.5 seconds for speed
    step = max(1, int(fps * 0.5))

    prev_hist = None
    boundaries = []
    frame_idx = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)

        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > threshold:
                boundaries.append(SceneBoundary(
                    timestamp=frame_idx / fps,
                    frame_number=frame_idx,
                    score=diff,
                ))

        prev_hist = hist
        frame_idx += step

        if progress_callback and total_frames > 0:
            progress_callback(min(frame_idx / total_frames, 1.0), "Scanning for scenes...")

    cap.release()
    logger.info("Histogram fallback: %d boundaries", len(boundaries))
    return boundaries


def adaptive_sample(
    video_path: str | Path,
    boundaries: List[SceneBoundary],
    budget: int = 200,
    min_frames_per_scene: int = 2,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> List[SampledFrame]:
    """Allocate frame budget across scenes proportionally to duration.

    Scenes with more visual change (shorter duration = rapid cuts) get
    proportionally more frames per second. Static talking-head scenes
    get fewer frames.

    Args:
        video_path: Path to video file.
        boundaries: Scene boundaries from detect_scene_boundaries().
        budget: Total number of frames to extract.
        min_frames_per_scene: Minimum frames per scene.
        progress_callback: Optional callback(progress, message).

    Returns:
        List of SampledFrame with extracted frame data.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Build scene intervals
    scene_starts = [0.0] + [b.timestamp for b in boundaries]
    scene_ends = [b.timestamp for b in boundaries] + [duration]
    scenes = list(zip(scene_starts, scene_ends))

    # Allocate frames per scene proportionally to duration
    scene_durations = [max(end - start, 0.1) for start, end in scenes]
    total_duration = sum(scene_durations)

    allocations = []
    for dur in scene_durations:
        alloc = max(min_frames_per_scene, int(budget * dur / total_duration))
        allocations.append(alloc)

    # Trim to budget
    while sum(allocations) > budget:
        # Reduce the largest allocation
        max_idx = allocations.index(max(allocations))
        allocations[max_idx] -= 1

    # Extract frames
    sampled = []
    extracted = 0
    total_to_extract = sum(allocations)

    for scene_idx, ((start, end), n_frames) in enumerate(zip(scenes, allocations)):
        if n_frames <= 0:
            continue

        scene_duration = end - start
        if scene_duration <= 0:
            continue

        # Evenly space frames within scene
        if n_frames == 1:
            timestamps = [start + scene_duration / 2]
        else:
            step = scene_duration / (n_frames + 1)
            timestamps = [start + step * (i + 1) for i in range(n_frames)]

        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            sampled.append(SampledFrame(
                frame=frame,
                timestamp=ts,
                frame_number=int(ts * fps),
                scene_index=scene_idx,
            ))
            extracted += 1

            if progress_callback:
                progress_callback(
                    extracted / total_to_extract,
                    f"Sampling frame {extracted}/{total_to_extract}",
                )

    cap.release()
    logger.info("Adaptive sampling: %d frames from %d scenes", len(sampled), len(scenes))
    return sampled


def deduplicate_frames(
    frames: List[SampledFrame],
    threshold: int = 8,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> List[SampledFrame]:
    """Remove near-duplicate frames using perceptual hashing.

    Args:
        frames: List of SampledFrame to deduplicate.
        threshold: Hamming distance threshold. Lower = stricter dedup.
        progress_callback: Optional callback(progress, message).

    Returns:
        Filtered list with duplicates removed.
    """
    if not frames:
        return []

    try:
        import imagehash
        from PIL import Image
    except ImportError:
        logger.warning("imagehash not available, skipping deduplication")
        return frames

    unique: List[SampledFrame] = []
    seen_hashes: List = []

    for i, sf in enumerate(frames):
        # Convert BGR numpy to PIL
        rgb = cv2.cvtColor(sf.frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        phash = imagehash.phash(pil_img)
        sf.phash = str(phash)

        # Check against all seen hashes
        is_dup = False
        for seen in seen_hashes:
            if phash - seen < threshold:
                is_dup = True
                break

        if not is_dup:
            unique.append(sf)
            seen_hashes.append(phash)

        if progress_callback:
            progress_callback(
                (i + 1) / len(frames),
                f"Deduplicating {i+1}/{len(frames)}",
            )

    logger.info("Deduplication: %d -> %d frames (removed %d)",
                len(frames), len(unique), len(frames) - len(unique))
    return unique
