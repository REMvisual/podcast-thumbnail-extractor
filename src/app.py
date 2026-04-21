#!/usr/bin/env python3
"""
Thumbnail Asset Extractor - Complete Working System
Single file implementation with embedded web interface

Requirements:
    pip install flask opencv-python pillow numpy rembg

Usage:
    python thumbnail_extractor_complete.py
    Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_file, send_from_directory, Response
from flask_cors import CORS
import cv2
import os
import sys
import numpy as np
from PIL import Image
import base64
import uuid
import zipfile
from io import BytesIO
from datetime import datetime
import shutil
import threading
import time
import json
from collections import OrderedDict
from rembg import remove

# Make `src/` importable when running this script directly.
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from vision.face_detector import get_face_detector
from vision.quality_scorer import get_quality_scorer
from vision.content_classifier import ContentCategory, get_content_classifier
from vision.frame_sampler import detect_scene_boundaries, adaptive_sample, deduplicate_frames
from vision.batch_pipeline import BatchProcessor

app = Flask(__name__)
CORS(app)

# Configuration — all paths env-var-overridable for portability
UPLOAD_FOLDER = os.environ.get('THUMBNAIL_EXTRACTOR_UPLOAD_DIR', './uploads')
OUTPUT_FOLDER = os.environ.get('THUMBNAIL_EXTRACTOR_OUTPUT_DIR', './outputs')
MODEL_FOLDER = os.environ.get('THUMBNAIL_EXTRACTOR_MODEL_DIR', './models')
PORT = int(os.environ.get('THUMBNAIL_EXTRACTOR_PORT', '5000'))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Shared vision singletons (lazy-initialized)
_face_detector = None
_quality_scorer = None
_content_classifier = None


def _get_face_detector():
    global _face_detector
    if _face_detector is None:
        _face_detector = get_face_detector()
    return _face_detector


def _get_quality_scorer():
    global _quality_scorer
    if _quality_scorer is None:
        # Use heuristic scorer for speed -- TOPIQ adds ~80ms/frame
        # and gives 1.0 to nearly all valid content frames anyway
        _quality_scorer = get_quality_scorer(backend="heuristic")
    return _quality_scorer


def _get_content_classifier():
    global _content_classifier
    if _content_classifier is None:
        _content_classifier = get_content_classifier()
    return _content_classifier

# ============= QUEUE & SSE INFRASTRUCTURE =============

_job_queue = OrderedDict()          # job_id -> job dict
_queue_lock = threading.Lock()
_sse_clients = []                   # list of queue.Queue for SSE clients
_sse_clients_lock = threading.Lock()
_cancellation_flags = {}            # job_id -> bool
_worker_thread = None
_worker_running = False


class CancelledException(Exception):
    """Raised when a job is cancelled mid-processing."""
    pass


class ThumbnailProgressEmitter:
    """Emits progress events for a specific job to all SSE clients."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.start_time = datetime.now()

    def emit(self, progress: float, message: str, assets_found: int = 0,
             frames_processed: int = 0, total_frames: int = 0):
        event = {
            'type': 'job_progress',
            'job_id': self.job_id,
            'progress': round(progress, 1),
            'message': message,
            'assets_found': assets_found,
            'frames_processed': frames_processed,
            'total_frames': total_frames
        }
        _broadcast_sse(event)
        # Also update job dict
        with _queue_lock:
            if self.job_id in _job_queue:
                _job_queue[self.job_id]['progress'] = round(progress, 1)
                _job_queue[self.job_id]['progress_message'] = message


def _broadcast_sse(event: dict):
    """Send an SSE event to all connected clients."""
    with _sse_clients_lock:
        dead = []
        for q in _sse_clients:
            try:
                q.put_nowait(event)
            except Exception:
                dead.append(q)
        for q in dead:
            _sse_clients.remove(q)


def _get_job_dict(job) -> dict:
    """Return a serializable copy of a job (without frame data)."""
    return {k: v for k, v in job.items() if k != 'video_path_internal'}


def _update_job_status(job_id: str, status: str, **kwargs):
    """Update job status and broadcast change."""
    with _queue_lock:
        if job_id not in _job_queue:
            return
        job = _job_queue[job_id]
        job['status'] = status
        for k, v in kwargs.items():
            job[k] = v

    event_type_map = {
        'processing': 'job_started',
        'completed': 'job_completed',
        'error': 'job_error',
        'cancelled': 'job_cancelled',
    }
    event = {'type': event_type_map.get(status, 'job_updated'), 'job_id': job_id}

    with _queue_lock:
        event['job'] = _get_job_dict(_job_queue[job_id])

    _broadcast_sse(event)


def _process_job(job):
    """Process a single extraction job."""
    job_id = job['id']
    emitter = ThumbnailProgressEmitter(job_id)

    try:
        _update_job_status(job_id, 'processing', started_at=datetime.now().isoformat(), progress=0)

        video_path = job['video_path']
        mode = job['mode']
        remove_bg = job['remove_bg']
        asset_count = job['asset_count']

        # Extract frames with progress
        assets = extract_and_score_frames(
            video_path, mode=mode, max_frames=asset_count * 10,
            emitter=emitter, job_id=job_id
        )

        if not assets:
            _update_job_status(job_id, 'error', error_message='No assets found in video',
                               completed_at=datetime.now().isoformat())
            try:
                os.remove(video_path)
            except OSError:
                pass
            return

        # Keep top N
        top_assets = assets[:asset_count]

        # Save assets
        video_id = job_id
        output_dir = os.path.join(OUTPUT_FOLDER, video_id)
        results = []

        emitter.emit(95, f'Saving {len(top_assets)} assets...', len(top_assets), 0, 0)

        for i, asset in enumerate(top_assets):
            # Check cancellation during save phase too
            if _cancellation_flags.get(job_id):
                raise CancelledException()

            filepath = save_asset(asset['frame'], asset['id'], output_dir, remove_bg=remove_bg)
            preview = frame_to_base64(asset['frame'])
            file_ext = 'png' if remove_bg else 'jpg'

            results.append({
                'id': asset['id'],
                'score': asset['score'],
                'timestamp': asset['timestamp'],
                'type': asset['type'],
                'preview': preview,
                'download_url': f"/api/download/{video_id}/{asset['id']}.{file_ext}"
            })

        # Clean up video file
        try:
            os.remove(video_path)
        except OSError:
            pass

        result_data = {
            'video_id': video_id,
            'mode': mode,
            'assets': results,
            'asset_count': len(results),
            'top_score': max(r['score'] for r in results) if results else 0
        }

        best_score = max(r['score'] for r in results) if results else 0
        summary = f"{len(results)} assets extracted (best: {best_score:.2f})" if results else "0 assets"
        _update_job_status(job_id, 'completed', completed_at=datetime.now().isoformat(),
                           progress=100, result=result_data, progress_message=summary)

    except CancelledException:
        _update_job_status(job_id, 'cancelled', completed_at=datetime.now().isoformat(),
                           progress_message='Cancelled by user')
        try:
            os.remove(job.get('video_path', ''))
        except OSError:
            pass

    except Exception as e:
        import traceback
        traceback.print_exc()
        _update_job_status(job_id, 'error', error_message=str(e),
                           completed_at=datetime.now().isoformat())
        try:
            os.remove(job.get('video_path', ''))
        except OSError:
            pass


def _queue_worker():
    """Background worker thread — processes jobs sequentially."""
    global _worker_running
    _worker_running = True

    while _worker_running:
        next_job = None
        with _queue_lock:
            for jid, job in _job_queue.items():
                if job['status'] == 'queued':
                    next_job = job
                    break

        if next_job:
            _process_job(next_job)
        else:
            time.sleep(0.5)


def _ensure_worker():
    """Start the worker thread if not already running."""
    global _worker_thread
    if _worker_thread is None or not _worker_thread.is_alive():
        _worker_thread = threading.Thread(target=_queue_worker, daemon=True)
        _worker_thread.start()


# ============= LEGACY MODEL COMPATIBILITY =============
# ResNet50 custom models replaced by shared vision library (OpenCLIP + MediaPipe).
# Legacy model upload/remove endpoints preserved for backward compatibility
# but now no-op (return success without loading).

# ============= VIDEO PROCESSING FUNCTIONS =============

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def detect_faces(frame):
    """Detect faces in frame using shared vision library (MediaPipe/Haar).

    Args:
        frame: RGB numpy array

    Returns:
        (has_faces: bool, num_faces: int)
    """
    try:
        # Convert RGB to BGR for shared detector
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        detections = _get_face_detector().detect(frame_bgr)
        return len(detections) > 0, len(detections)
    except Exception as e:
        print(f"Face detection error: {e}")
        return False, 0


def detect_faces_detailed(frame_bgr):
    """Detect faces with full detection details.

    Args:
        frame_bgr: BGR numpy array

    Returns:
        List of FaceDetection objects with confidence, landmarks, etc.
    """
    try:
        return _get_face_detector().detect(frame_bgr)
    except Exception as e:
        print(f"Face detection error: {e}")
        return []


def score_frame_quality(frame, face_detections=None):
    """
    Score frame quality using shared vision library.

    Uses face-aware quality scoring when face_detections provided:
    - Penalizes blinks (eye aspect ratio)
    - Penalizes mid-word expressions (mouth aspect ratio)
    - Rewards large, high-confidence faces

    Args:
        frame: RGB numpy array
        face_detections: Optional list of FaceDetection from shared detector

    Returns:
        float between 0 and 1
    """
    try:
        # Convert RGB to BGR for shared scorer
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        score = _get_quality_scorer().score(frame_bgr, face_detections)
        return score.total
    except Exception as e:
        print(f"Quality scoring error: {e}")
        return 0.0


def classify_content(frame_bgr):
    """Classify frame content using shared vision library (OpenCLIP/edge density).

    Args:
        frame_bgr: BGR numpy array

    Returns:
        (category: ContentCategory, scores: dict)
    """
    try:
        return _get_content_classifier().classify(frame_bgr)
    except Exception as e:
        print(f"Content classification error: {e}")
        return ContentCategory.UNKNOWN, {}


def extract_and_score_frames(video_path, mode='faces', max_frames=200,
                              emitter=None, job_id=None):
    """
    Extract frames from video using 3-pass pipeline and score them.

    Pass 1 (0-20%):  Scene boundary detection via histogram diff
    Pass 2 (20-80%): Adaptive sampling + face/content detection + quality scoring
    Pass 3 (80-90%): Perceptual hash deduplication
    Scoring (90-100%): Final sort

    Args:
        video_path: Path to video file
        mode: 'faces', 'screens_ui', 'screens_art', 'screens_mixed', or 'both'
        max_frames: Maximum frames to extract
        emitter: Optional ThumbnailProgressEmitter for real-time progress
        job_id: Optional job ID for cancellation checks

    Returns:
        List of assets with scores and metadata
    """
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(video_path)}")
    print(f"   Mode: {mode}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames_raw / fps if fps > 0 else 0
    cap.release()

    print(f"   Duration: {duration:.1f}s ({total_frames_raw} frames)")
    print(f"   FPS: {fps:.1f}")

    # --- PASS 1: Scene boundary detection (0-20% progress) ---
    if emitter:
        emitter.emit(2, "Pass 1: Detecting scene boundaries...", 0, 0, total_frames_raw)

    if job_id and _cancellation_flags.get(job_id):
        raise CancelledException()

    def scene_progress(p, msg):
        if emitter:
            emitter.emit(p * 18 + 2, f"Pass 1: {msg}", 0, 0, total_frames_raw)

    boundaries = detect_scene_boundaries(video_path, progress_callback=scene_progress)
    print(f"   Scene boundaries: {len(boundaries)} detected")

    if emitter:
        emitter.emit(20, f"Pass 1 complete: {len(boundaries)} scenes", 0, 0, total_frames_raw)

    # --- PASS 2: Adaptive sampling + detection (20-80% progress) ---
    if job_id and _cancellation_flags.get(job_id):
        raise CancelledException()

    # Budget: ~5 frames per minute for full coverage.
    # Temporal selection handles even spread, so we don't need massive over-sampling.
    duration_minutes = max(duration / 60.0, 1.0)
    sample_budget = max(int(duration_minutes * 5), max_frames, 200)

    print(f"   Sample budget: {sample_budget} frames ({duration_minutes:.1f} min video)")

    def sample_progress(p, msg):
        if emitter:
            emitter.emit(20 + p * 20, f"Pass 2a: {msg}", 0, 0, total_frames_raw)

    sampled_frames = adaptive_sample(
        video_path, boundaries, budget=sample_budget,
        progress_callback=sample_progress,
    )
    print(f"   Sampled frames: {len(sampled_frames)}")

    if emitter:
        emitter.emit(40, f"Classifying {len(sampled_frames)} frames...", 0, 0, total_frames_raw)

    # --- Batch CLIP classification (much faster than per-frame) ---
    BATCH_SIZE = 16
    classifier = _get_content_classifier()
    all_classifications = []

    for batch_start in range(0, len(sampled_frames), BATCH_SIZE):
        if job_id and _cancellation_flags.get(job_id):
            raise CancelledException()

        batch_end = min(batch_start + BATCH_SIZE, len(sampled_frames))
        batch_bgr = [sf.frame for sf in sampled_frames[batch_start:batch_end]]

        try:
            batch_results = classifier.classify_batch(batch_bgr)
        except Exception:
            batch_results = [classifier.classify(f) for f in batch_bgr]

        all_classifications.extend(batch_results)

        if emitter and batch_start % (BATCH_SIZE * 4) == 0:
            progress = 40 + (batch_start / len(sampled_frames)) * 25
            emitter.emit(progress, f"Pass 2b: Classified {batch_start}/{len(sampled_frames)} frames",
                        0, batch_start, len(sampled_frames))

    print(f"   Classification done: {len(all_classifications)} results")

    if emitter:
        emitter.emit(65, "Scoring and filtering frames...", 0, 0, 0)

    # --- Per-frame: face detection + quality scoring + mode filtering ---
    assets = []
    needs_faces = mode in ('faces', 'both')

    for i, (sf, (category, cat_scores)) in enumerate(zip(sampled_frames, all_classifications)):
        if job_id and _cancellation_flags.get(job_id):
            raise CancelledException()

        frame_bgr = sf.frame
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        timestamp_str = format_timestamp(sf.timestamp)

        # Face detection only when needed (saves ~3ms/frame)
        face_dets = []
        has_face = False
        num_faces = 0
        if needs_faces:
            face_dets = detect_faces_detailed(frame_bgr)
            has_face = len(face_dets) > 0
            num_faces = len(face_dets)

        # Quality scoring (heuristic, <1ms/frame)
        quality_score = score_frame_quality(frame_rgb, face_detections=face_dets if face_dets else None)
        # Get sharpness sub-score for screen modes (heuristic total penalizes art via brightness/contrast)
        sharpness_score = _get_quality_scorer().score(frame_bgr).sharpness

        # Mode-specific filtering
        include_frame = False
        frame_type = "other"

        if mode == 'faces' or mode == 'both':
            if has_face:
                include_frame = True
                frame_type = f"{num_faces} face(s)" if num_faces > 1 else "face"
                best_conf = max((d.confidence for d in face_dets), default=0)
                quality_score = min(1.0, quality_score * (1.0 + best_conf * 0.3))

        if mode in ['screens_ui', 'screens_art', 'screens_mixed', 'both']:
            is_ui = category == ContentCategory.UI_INTERFACE
            is_art = category == ContentCategory.ART_RENDER

            if mode == 'screens_ui' and is_ui:
                include_frame = True
                frame_type = "UI/interface"
                clip_conf = cat_scores.get(ContentCategory.UI_INTERFACE, 0)
                # Normalize CLIP softmax (0.25=random across 4 cats) to 0-1 range
                clip_norm = min(1.0, max(0.0, (clip_conf - 0.25) / 0.40))
                # Sharpness + normalized CLIP (heuristic total penalizes art)
                quality_score = sharpness_score * 0.5 + clip_norm * 0.5

            elif mode == 'screens_art' and is_art:
                include_frame = True
                frame_type = "art/render"
                clip_conf = cat_scores.get(ContentCategory.ART_RENDER, 0)
                clip_norm = min(1.0, max(0.0, (clip_conf - 0.25) / 0.40))
                quality_score = sharpness_score * 0.5 + clip_norm * 0.5

            elif mode == 'screens_mixed' and (is_ui or is_art):
                include_frame = True
                frame_type = "UI + art" if (is_ui and is_art) else ("UI/interface" if is_ui else "art/render")
                best_clip = max(
                    cat_scores.get(ContentCategory.UI_INTERFACE, 0),
                    cat_scores.get(ContentCategory.ART_RENDER, 0),
                )
                clip_norm = min(1.0, max(0.0, (best_clip - 0.25) / 0.40))
                quality_score = sharpness_score * 0.5 + clip_norm * 0.5

            elif mode == 'both' and (is_ui or is_art):
                if frame_type == "other":
                    frame_type = "screen"
                else:
                    frame_type = "face + screen"
                include_frame = True

        if mode == 'both' and category != ContentCategory.TRANSITION:
            include_frame = True

        quality_score = min(quality_score, 1.0)

        if include_frame:
            asset_id = str(uuid.uuid4())
            assets.append({
                'id': asset_id,
                'frame': frame_rgb,
                'score': round(quality_score, 2),
                'timestamp': timestamp_str,
                'type': frame_type,
                'frame_number': sf.frame_number,
                '_sampled_frame': sf,
            })

        if i % 50 == 0 and emitter:
            progress = 65 + (i / len(sampled_frames)) * 15
            emitter.emit(progress, f"Pass 2c: Filtered {i}/{len(sampled_frames)}, {len(assets)} matched",
                        len(assets), i, len(sampled_frames))

    print(f"   Assets after classification: {len(assets)}")

    if emitter:
        emitter.emit(80, f"Pass 2 complete: {len(assets)} assets found", len(assets), 0, 0)

    # --- PASS 3: Perceptual hash deduplication (80-90% progress) ---
    if job_id and _cancellation_flags.get(job_id):
        raise CancelledException()

    if len(assets) > 0:
        if emitter:
            emitter.emit(82, "Pass 3: Deduplicating similar frames...", len(assets), 0, 0)

        # Build SampledFrame list for dedup, maintaining index correspondence
        from vision.frame_sampler import SampledFrame
        sf_list = []
        for asset in assets:
            sf = asset.get('_sampled_frame')
            if sf:
                sf_bgr = SampledFrame(
                    frame=cv2.cvtColor(asset['frame'], cv2.COLOR_RGB2BGR),
                    timestamp=sf.timestamp,
                    frame_number=sf.frame_number,
                    scene_index=sf.scene_index,
                )
                sf_list.append(sf_bgr)
            else:
                sf_list.append(None)

        def dedup_progress(p, msg):
            if emitter:
                emitter.emit(82 + p * 8, f"Pass 3: {msg}", len(assets), 0, 0)

        # Filter to non-None for dedup
        valid_indices = [i for i, sf in enumerate(sf_list) if sf is not None]
        valid_sfs = [sf_list[i] for i in valid_indices]

        if valid_sfs:
            unique_sfs = deduplicate_frames(valid_sfs, threshold=8, progress_callback=dedup_progress)
            # Find which indices survived dedup (by frame_number match)
            unique_frame_numbers = {sf.frame_number for sf in unique_sfs}
            keep_indices = set()
            for i in valid_indices:
                if sf_list[i].frame_number in unique_frame_numbers:
                    keep_indices.add(i)
            # Also keep assets without _sampled_frame
            for i, sf in enumerate(sf_list):
                if sf is None:
                    keep_indices.add(i)

            before_count = len(assets)
            assets = [assets[i] for i in sorted(keep_indices)]
            print(f"   After dedup: {before_count} -> {len(assets)} assets")
        else:
            print(f"   No valid frames for dedup, keeping all {len(assets)} assets")

    # Clean up internal references
    for asset in assets:
        asset.pop('_sampled_frame', None)

    # --- PASS 4: Temporally diverse selection (90-95% progress) ---
    # Instead of just "top N by score", divide the video into N time buckets
    # and pick the best frame from each bucket. This ensures coverage across
    # the entire video duration.
    if emitter:
        emitter.emit(90, f"Pass 4: Selecting best frames across timeline...", len(assets), 0, 0)

    # target_count is the user's requested asset count, not the over-sampled max_frames
    # max_frames is asset_count * 10 (over-sample), but we want to select asset_count
    target_asset_count = min(len(assets), max_frames // 10) if max_frames > 0 else len(assets)
    target_asset_count = max(target_asset_count, 1)
    assets = _select_temporally_diverse(assets, target_asset_count, duration)

    if emitter:
        emitter.emit(95, f"Selected {len(assets)} assets spanning full video", len(assets), 0, 0)

    print(f"* Extraction complete!")
    print(f"   Total assets found: {len(assets)}")
    print(f"   Top score: {assets[0]['score'] if assets else 0:.2f}")
    print(f"{'='*60}\n")

    return assets


def _select_temporally_diverse(assets, target_count, video_duration):
    """Select frames spread across the full video duration.

    Algorithm:
    1. Parse timestamps to seconds
    2. Divide video into `target_count` equal time buckets
    3. From each bucket, pick the highest-scored frame
    4. If a bucket is empty, skip it (some segments may have no matching content)
    5. Sort final selection by timestamp for consistent display

    Args:
        assets: List of asset dicts with 'timestamp' (HH:MM:SS) and 'score'.
        target_count: Desired number of frames to return.
        video_duration: Total video duration in seconds.

    Returns:
        Temporally diverse subset of assets, sorted by score descending.
    """
    if not assets or target_count <= 0:
        return assets

    def _ts_to_seconds(ts_str):
        parts = ts_str.split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return 0

    # If fewer assets than target, return all
    if len(assets) <= target_count:
        assets.sort(key=lambda x: x['score'], reverse=True)
        return assets

    # Find actual time range of assets
    asset_times = [(_ts_to_seconds(a['timestamp']), a) for a in assets]
    min_time = min(t for t, _ in asset_times)
    max_time = max(t for t, _ in asset_times)

    print(f"   Temporal select: {len(assets)} candidates, target={target_count}")
    print(f"   Asset time range: {min_time}s - {max_time}s, video duration: {video_duration:.0f}s")
    time_span = max(max_time - min_time, 1)

    # Use the larger of video_duration or asset time span
    effective_duration = max(video_duration, max_time + 1) if video_duration > 0 else time_span + 1

    # Create time buckets
    bucket_size = effective_duration / target_count
    buckets = [[] for _ in range(target_count)]

    for ts_sec, asset in asset_times:
        bucket_idx = min(int(ts_sec / bucket_size), target_count - 1)
        buckets[bucket_idx].append(asset)

    # Pick best from each bucket
    selected = []
    for bucket in buckets:
        if bucket:
            best = max(bucket, key=lambda a: a['score'])
            selected.append(best)

    # If we got fewer than target (empty buckets), fill with remaining best-scored
    if len(selected) < target_count:
        selected_ids = {a['id'] for a in selected}
        remaining = [a for a in assets if a['id'] not in selected_ids]
        remaining.sort(key=lambda a: a['score'], reverse=True)

        # Fill empty buckets with best remaining, preferring temporal gaps
        for asset in remaining:
            if len(selected) >= target_count:
                break
            selected.append(asset)

    # Sort by timestamp for chronological display
    selected.sort(key=lambda x: x['timestamp'])

    filled_buckets = sum(1 for b in buckets if b)
    print(f"   Temporal selection: {len(assets)} candidates -> {len(selected)} selected "
          f"({filled_buckets}/{target_count} time buckets filled)")

    return selected


def remove_background(frame):
    """
    Remove background from frame using rembg
    
    Args:
        frame: RGB numpy array
    
    Returns:
        RGBA PIL Image with transparent background
    """
    try:
        print(f"   Removing background...")
        # Convert numpy array to PIL Image
        img = Image.fromarray(frame)
        
        # Remove background (returns RGBA image)
        output = remove(img)
        
        print(f"   * Background removed successfully")
        return output
    except Exception as e:
        print(f"   ERROR: Background removal error: {e}")
        import traceback
        traceback.print_exc()
        # Return original image converted to RGBA if removal fails
        return Image.fromarray(frame).convert('RGBA')


def save_asset(frame, asset_id, output_dir, remove_bg=False):
    """
    Save frame as JPG image
    
    Args:
        frame: RGB numpy array
        asset_id: Unique identifier
        output_dir: Directory to save to
        remove_bg: If True, remove background and save as PNG
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"   Saving asset {asset_id} (remove_bg={remove_bg})")
    
    if remove_bg:
        # Remove background and save as PNG with transparency
        filepath = os.path.join(output_dir, f"{asset_id}.png")
        print(f"   > Removing background and saving as PNG...")
        img_no_bg = remove_background(frame)
        img_no_bg.save(filepath, format='PNG')
        print(f"   * Saved: {filepath}")
    else:
        # Save as JPG
        filepath = os.path.join(output_dir, f"{asset_id}.jpg")
        print(f"   > Saving as JPG...")
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"   * Saved: {filepath}")
    
    return filepath


def frame_to_base64(frame, remove_bg_preview=False):
    """
    Convert frame to base64 for preview
    
    NOTE: Currently shows original frame with background for speed.
    Set remove_bg_preview=True to show transparent previews (slower, larger files)
    """
    try:
        # Convert to PIL Image
        img = Image.fromarray(frame)
        
        # OPTIONAL: Remove background in preview (currently disabled for performance)
        # if remove_bg_preview:
        #     img = remove_background(frame)
        
        # Resize for preview (smaller filesize)
        img.thumbnail((400, 300), Image.LANCZOS)
        
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        print(f"Base64 conversion error: {e}")
        return None


# ============= API ENDPOINTS =============

from config_loader import (
    CategoryError,
    delete_category as _delete_category,
    list_categories,
    save_category,
    update_category,
)


@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')


@app.route('/api/categories', methods=['GET'])
def api_list_categories():
    return jsonify({"categories": list_categories()})


@app.route('/api/categories', methods=['POST'])
def api_create_category():
    try:
        save_category(None, request.get_json(force=True))
    except CategoryError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True}), 201


@app.route('/api/categories/<cat_id>', methods=['PUT'])
def api_update_category(cat_id):
    try:
        update_category(None, cat_id, request.get_json(force=True))
    except CategoryError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True})


@app.route('/api/categories/<cat_id>', methods=['DELETE'])
def api_delete_category(cat_id):
    try:
        _delete_category(None, cat_id)
    except CategoryError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True})


@app.route('/api/process', methods=['POST'])
def process_video():
    """Process uploaded video and extract assets"""
    try:
        # Get uploaded video
        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded'}), 400
        
        video_file = request.files['video']
        mode = request.form.get('mode', 'faces')
        remove_bg = request.form.get('remove_bg', 'false').lower() == 'true'
        asset_count = int(request.form.get('asset_count', '10'))
        
        if video_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save video
        video_id = str(uuid.uuid4())
        video_ext = os.path.splitext(video_file.filename)[1]
        video_path = os.path.join(UPLOAD_FOLDER, f"{video_id}{video_ext}")
        video_file.save(video_path)
        
        print(f"\nNew processing request")
        print(f"   File: {video_file.filename}")
        print(f"   Mode: {mode}")
        print(f"   Asset count: {asset_count}")
        print(f"   Remove BG: {remove_bg}")
        print(f"   Size: {os.path.getsize(video_path) / 1024 / 1024:.2f} MB")
        
        # Extract and score frames
        assets = extract_and_score_frames(video_path, mode=mode, max_frames=asset_count * 10)
        
        if not assets:
            os.remove(video_path)
            return jsonify({'error': 'No assets found in video'}), 400
        
        # Keep top N (user selected)
        top_assets = assets[:asset_count]
        
        # Save assets and create response
        output_dir = os.path.join(OUTPUT_FOLDER, video_id)
        results = []
        
        for asset in top_assets:
            # Save full resolution (with or without background removal)
            filepath = save_asset(asset['frame'], asset['id'], output_dir, remove_bg=remove_bg)
            
            # Create preview (base64)
            preview = frame_to_base64(asset['frame'])
            
            # Determine file extension
            file_ext = 'png' if remove_bg else 'jpg'
            
            results.append({
                'id': asset['id'],
                'score': asset['score'],
                'timestamp': asset['timestamp'],
                'type': asset['type'],
                'preview': preview,
                'download_url': f"/api/download/{video_id}/{asset['id']}.{file_ext}"
            })
        
        # Clean up video file
        os.remove(video_path)
        
        print(f"* Processing complete! Returning {len(results)} assets")
        
        return jsonify({
            'success': True,
            'mode': mode,
            'assets': results,
            'video_id': video_id
        })
    
    except Exception as e:
        print(f"ERROR: Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<video_id>/<filename>')
def download_asset(video_id, filename):
    """Download individual asset (JPG or PNG)"""
    try:
        filepath = os.path.join(OUTPUT_FOLDER, video_id, filename)
        
        if os.path.exists(filepath):
            # Determine mimetype based on extension
            if filename.endswith('.png'):
                mimetype = 'image/png'
            else:
                mimetype = 'image/jpeg'
            
            return send_file(
                filepath,
                mimetype=mimetype,
                as_attachment=True,
                download_name=filename
            )
        else:
            return jsonify({'error': 'Asset not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download-all/<video_id>')
def download_all_assets(video_id):
    """Download all assets as ZIP"""
    try:
        output_dir = os.path.join(OUTPUT_FOLDER, video_id)
        
        if not os.path.exists(output_dir):
            return jsonify({'error': 'Assets not found'}), 404
        
        # Create ZIP in memory
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for filename in os.listdir(output_dir):
                if filename.endswith(('.jpg', '.png')):
                    filepath = os.path.join(output_dir, filename)
                    zf.write(filepath, arcname=filename)
        
        memory_file.seek(0)
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'thumbnail_assets_{video_id}.zip'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    from vision.model_registry import get_status
    model_status = get_status()
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'opencv_version': cv2.__version__,
        'vision_models': model_status,
        # Legacy fields for backward compatibility
        'ui_model_loaded': model_status.get('clip_vit_b32', {}).get('available', False),
        'art_model_loaded': model_status.get('clip_vit_b32', {}).get('available', False),
        'face_model_loaded': model_status.get('mediapipe_face', {}).get('available', False),
    })


@app.route('/api/upload-model', methods=['POST'])
def upload_custom_model():
    """Legacy endpoint: custom .pth models replaced by shared vision library.
    Accepts upload for backward compatibility but uses OpenCLIP/MediaPipe instead."""
    return jsonify({
        'success': True,
        'message': 'Custom models deprecated. Using shared vision library (OpenCLIP + MediaPipe).',
        'model_type': request.form.get('model_type', 'ui'),
        'model_active': True
    })


@app.route('/api/model-status')
def model_status():
    """Get current model status."""
    from vision.model_registry import get_status
    status = get_status()
    return jsonify({
        # Legacy fields
        'ui_model_loaded': status.get('clip_vit_b32', {}).get('available', False),
        'art_model_loaded': status.get('clip_vit_b32', {}).get('available', False),
        'face_model_loaded': status.get('mediapipe_face', {}).get('available', False),
        # New detailed status
        'vision_models': status,
    })


@app.route('/api/remove-model', methods=['POST'])
def remove_custom_model():
    """Legacy endpoint: no-op since models are now managed by shared vision library."""
    model_type = 'ui'
    if request.is_json:
        model_type = request.json.get('model_type', 'ui')
    else:
        model_type = request.form.get('model_type', 'ui')

    return jsonify({
        'success': True,
        'message': f'Models now managed by shared vision library. Legacy {model_type.upper()} model endpoint deprecated.',
        'model_type': model_type
    })


# ============= QUEUE API ENDPOINTS =============

@app.route('/api/queue/add', methods=['POST'])
def queue_add():
    """Upload video + config, add to queue, return job_id immediately."""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded'}), 400

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        mode = request.form.get('mode', 'faces')
        remove_bg = request.form.get('remove_bg', 'false').lower() == 'true'
        asset_count = int(request.form.get('asset_count', '10'))

        job_id = str(uuid.uuid4())
        video_ext = os.path.splitext(video_file.filename)[1]
        video_path = os.path.join(UPLOAD_FOLDER, f"{job_id}{video_ext}")
        video_file.save(video_path)

        file_size_mb = round(os.path.getsize(video_path) / 1024 / 1024, 2)

        job = {
            'id': job_id,
            'filename': video_file.filename,
            'video_path': video_path,
            'mode': mode,
            'remove_bg': remove_bg,
            'asset_count': asset_count,
            'status': 'queued',
            'progress': 0,
            'progress_message': '',
            'created_at': datetime.now().isoformat(),
            'started_at': None,
            'completed_at': None,
            'error_message': None,
            'result': None,
            'file_size_mb': file_size_mb
        }

        with _queue_lock:
            _job_queue[job_id] = job
        _cancellation_flags[job_id] = False

        _broadcast_sse({'type': 'job_added', 'job': _get_job_dict(job)})

        print(f"[Queue] Added job {job_id[:8]}... - {video_file.filename} ({mode}, {asset_count} assets)")

        return jsonify({'success': True, 'job_id': job_id, 'job': _get_job_dict(job)})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/queue', methods=['GET'])
def queue_list():
    """Get all jobs and their current states."""
    with _queue_lock:
        jobs = [_get_job_dict(j) for j in _job_queue.values()]
    return jsonify({'success': True, 'jobs': jobs})


@app.route('/api/queue/start', methods=['POST'])
def queue_start():
    """Start processing all queued jobs."""
    _ensure_worker()
    return jsonify({'success': True, 'message': 'Worker started'})


@app.route('/api/queue/<job_id>/cancel', methods=['POST'])
def queue_cancel(job_id):
    """Cancel a queued or processing job."""
    with _queue_lock:
        if job_id not in _job_queue:
            return jsonify({'error': 'Job not found'}), 404
        job = _job_queue[job_id]

    if job['status'] == 'queued':
        _update_job_status(job_id, 'cancelled', completed_at=datetime.now().isoformat(),
                           progress_message='Cancelled before processing')
        try:
            os.remove(job.get('video_path', ''))
        except OSError:
            pass
    elif job['status'] == 'processing':
        _cancellation_flags[job_id] = True
    else:
        return jsonify({'error': f'Cannot cancel job in status: {job["status"]}'}), 400

    return jsonify({'success': True, 'message': f'Job {job_id} cancellation requested'})


@app.route('/api/queue/<job_id>/remove', methods=['POST'])
def queue_remove(job_id):
    """Remove a completed/cancelled/error job from the queue."""
    with _queue_lock:
        if job_id not in _job_queue:
            return jsonify({'error': 'Job not found'}), 404
        job = _job_queue[job_id]
        if job['status'] in ('queued', 'processing'):
            return jsonify({'error': 'Cannot remove active job. Cancel it first.'}), 400
        del _job_queue[job_id]

    _cancellation_flags.pop(job_id, None)
    _broadcast_sse({'type': 'job_removed', 'job_id': job_id})

    return jsonify({'success': True})


@app.route('/api/queue/<job_id>/results', methods=['GET'])
def queue_results(job_id):
    """Get results for a completed job."""
    with _queue_lock:
        if job_id not in _job_queue:
            return jsonify({'error': 'Job not found'}), 404
        job = _job_queue[job_id]

    if job['status'] != 'completed' or not job.get('result'):
        return jsonify({'error': 'No results available'}), 400

    return jsonify({'success': True, 'result': job['result']})


@app.route('/api/queue/<job_id>/retry', methods=['POST'])
def queue_retry(job_id):
    """Retry a failed job."""
    with _queue_lock:
        if job_id not in _job_queue:
            return jsonify({'error': 'Job not found'}), 404
        job = _job_queue[job_id]
        if job['status'] not in ('error', 'cancelled'):
            return jsonify({'error': 'Can only retry failed/cancelled jobs'}), 400

        # Re-upload the video if it was cleaned up
        if not os.path.exists(job.get('video_path', '')):
            return jsonify({'error': 'Video file no longer available. Please re-add.'}), 400

        job['status'] = 'queued'
        job['progress'] = 0
        job['progress_message'] = ''
        job['error_message'] = None
        job['started_at'] = None
        job['completed_at'] = None
        job['result'] = None

    _cancellation_flags[job_id] = False
    _broadcast_sse({'type': 'job_added', 'job': _get_job_dict(job)})
    _ensure_worker()

    return jsonify({'success': True, 'message': 'Job re-queued'})


@app.route('/api/sse/queue')
def sse_queue():
    """SSE stream for all queue events."""
    def generate():
        q = __import__('queue').Queue(maxsize=200)
        with _sse_clients_lock:
            _sse_clients.append(q)
        try:
            while True:
                try:
                    event = q.get(timeout=15)
                    yield f"data: {json.dumps(event)}\n\n"
                except Exception:
                    # Heartbeat on timeout
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        except GeneratorExit:
            pass
        finally:
            with _sse_clients_lock:
                if q in _sse_clients:
                    _sse_clients.remove(q)

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
            'Access-Control-Allow-Origin': '*'
        }
    )


# ============= HTML TEMPLATE =============

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Thumbnail Asset Extractor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 100%);
            color: #ffffff;
            min-height: 100vh;
            padding: 2rem;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #ff6b35, #f7931e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .subtitle {
            color: #a0a0b8;
            margin-bottom: 3rem;
            font-size: 1.1rem;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
        }
        
        .card h2 {
            margin-bottom: 1.5rem;
            font-size: 1.5rem;
        }
        
        .upload-zone {
            border: 2px dashed rgba(255, 107, 53, 0.5);
            border-radius: 12px;
            padding: 3rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: rgba(255, 107, 53, 0.05);
        }
        
        .upload-zone:hover {
            border-color: #ff6b35;
            background: rgba(255, 107, 53, 0.1);
        }
        
        .upload-zone.dragover {
            border-color: #f7931e;
            background: rgba(247, 147, 30, 0.2);
            transform: scale(1.02);
        }
        
        .radio-group {
            display: flex;
            gap: 1rem;
            margin: 1.5rem 0;
        }
        
        .radio-option {
            flex: 1;
        }
        
        .radio-option input[type="radio"] {
            display: none;
        }
        
        .radio-label {
            display: block;
            padding: 1.5rem;
            background: rgba(255, 255, 255, 0.05);
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s;
            text-align: center;
        }
        
        .radio-option input[type="radio"]:checked + .radio-label {
            background: rgba(255, 107, 53, 0.2);
            border-color: #ff6b35;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #ff6b35, #f7931e);
            color: white;
            border: none;
            padding: 1.25rem 3rem;
            font-size: 1.1rem;
            font-weight: 700;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn-primary:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(255, 107, 53, 0.3);
        }
        
        .btn-primary:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .processing {
            text-align: center;
            padding: 3rem;
        }
        
        .spinner {
            width: 60px;
            height: 60px;
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-top: 4px solid #ff6b35;
            border-radius: 50%;
            margin: 0 auto 1.5rem;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
        }
        
        .asset-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            overflow: hidden;
            transition: all 0.3s;
        }
        
        .asset-card:hover {
            border-color: #ff6b35;
            transform: translateY(-4px);
        }
        
        .asset-image {
            width: 100%;
            height: 180px;
            object-fit: cover;
            background: rgba(0, 0, 0, 0.3);
        }
        
        .asset-info {
            padding: 1rem;
        }
        
        .asset-score {
            display: inline-block;
            background: linear-gradient(135deg, #ff6b35, #f7931e);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .download-btn {
            width: 100%;
            background: #ff6b35;
            color: white;
            border: none;
            padding: 0.75rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s;
            margin-top: 0.5rem;
            border-radius: 6px;
        }
        
        .download-btn:hover {
            background: #f7931e;
        }
        
        video {
            max-width: 100%;
            max-height: 400px;
            border-radius: 12px;
            margin-bottom: 1rem;
        }
        
        .status-message {
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        
        .status-success {
            background: rgba(74, 222, 128, 0.1);
            border: 1px solid rgba(74, 222, 128, 0.3);
            color: #4ade80;
        }
        
        .status-error {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #ef4444;
        }
        
        /* Slider styling */
        input[type="range"] {
            -webkit-appearance: none;
            appearance: none;
            width: 100%;
            height: 8px;
            background: linear-gradient(90deg, #ff6b35, #f7931e);
            border-radius: 4px;
            outline: none;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 24px;
            height: 24px;
            background: white;
            border: 3px solid #ff6b35;
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
            transition: all 0.2s;
        }
        
        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.1);
            box-shadow: 0 4px 12px rgba(255, 107, 53, 0.5);
        }
        
        input[type="range"]::-moz-range-thumb {
            width: 24px;
            height: 24px;
            background: white;
            border: 3px solid #ff6b35;
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
            transition: all 0.2s;
        }
        
        input[type="range"]::-moz-range-thumb:hover {
            transform: scale(1.1);
            box-shadow: 0 4px 12px rgba(255, 107, 53, 0.5);
        }

        /* Queue styles */
        .btn-outline {
            background: transparent;
            color: #ff6b35;
            border: 2px solid #ff6b35;
            padding: 1.25rem 2.5rem;
            font-size: 1.1rem;
            font-weight: 700;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-outline:hover:not(:disabled) {
            background: rgba(255, 107, 53, 0.1);
            transform: translateY(-2px);
        }
        .btn-outline:disabled { opacity: 0.5; cursor: not-allowed; }

        .btn-sm {
            padding: 0.4rem 0.8rem;
            font-size: 0.8rem;
            border-radius: 6px;
            border: none;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.2s;
        }
        .btn-sm:hover { opacity: 0.85; }

        .action-buttons {
            display: flex;
            gap: 1rem;
            justify-content: center;
            margin: 2rem 0;
        }

        .queue-panel {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            margin-bottom: 2rem;
            overflow: hidden;
        }
        .queue-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.25rem 1.5rem;
            background: rgba(255, 255, 255, 0.05);
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }
        .queue-header h2 { margin: 0; font-size: 1.25rem; }
        .queue-empty {
            padding: 3rem;
            text-align: center;
            color: #a0a0b8;
            font-size: 0.95rem;
        }
        .queue-item {
            padding: 1.25rem 1.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            transition: all 0.3s;
        }
        .queue-item:last-child { border-bottom: none; }
        .queue-item-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .queue-item-status {
            width: 28px;
            height: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            flex-shrink: 0;
        }
        .queue-item-name {
            flex: 1;
            font-weight: 600;
            font-size: 0.95rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .queue-item-tags {
            display: flex;
            gap: 0.4rem;
            flex-shrink: 0;
        }
        .tag {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
        }
        .tag-mode { background: rgba(99, 102, 241, 0.25); color: #818cf8; }
        .tag-count { background: rgba(255, 107, 53, 0.25); color: #ff6b35; }
        .tag-size { background: rgba(255, 255, 255, 0.1); color: #a0a0b8; }
        .queue-item-actions {
            display: flex;
            gap: 0.4rem;
            flex-shrink: 0;
            margin-left: 0.5rem;
        }

        /* Status-specific styles */
        .queue-item.processing {
            border-left: 3px solid #ff6b35;
            background: rgba(255, 107, 53, 0.04);
            animation: pulse-border 2s ease-in-out infinite;
        }
        @keyframes pulse-border {
            0%, 100% { box-shadow: inset 0 0 0 0 rgba(255,107,53,0); }
            50% { box-shadow: inset 0 0 20px 0 rgba(255,107,53,0.06); }
        }
        .queue-item.completed { border-left: 3px solid #4ade80; background: rgba(74, 222, 128, 0.03); }
        .queue-item.error { border-left: 3px solid #ef4444; background: rgba(239, 68, 68, 0.03); }
        .queue-item.cancelled { border-left: 3px solid #6b7280; opacity: 0.6; }

        .progress-bar-track {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            margin-top: 0.75rem;
            overflow: hidden;
        }
        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #ff6b35, #f7931e);
            border-radius: 3px;
            transition: width 0.4s ease;
        }
        .progress-text {
            font-size: 0.8rem;
            color: #a0a0b8;
            margin-top: 0.4rem;
        }
        .queue-item-detail {
            font-size: 0.85rem;
            color: #a0a0b8;
            margin-top: 0.5rem;
        }
        .queue-item-detail.error-msg { color: #ef4444; }

        .results-toggle {
            margin-top: 0.75rem;
        }
        .results-toggle-btn {
            background: rgba(74, 222, 128, 0.15);
            color: #4ade80;
            border: 1px solid rgba(74, 222, 128, 0.3);
            padding: 0.4rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 600;
            transition: all 0.2s;
        }
        .results-toggle-btn:hover { background: rgba(74, 222, 128, 0.25); }
        .inline-results {
            margin-top: 1rem;
            display: none;
        }
        .inline-results.open { display: block; }
        .inline-results .results-grid {
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 1rem;
        }
        .inline-results .asset-image { height: 120px; }

        .spin-slow {
            animation: spin 2s linear infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Thumbnail Asset Extractor</h1>
        <p class="subtitle">AI-powered frame extraction for podcast thumbnails</p>
        
        <div id="status"></div>
        
        <!-- Custom Model Section -->
        <div class="card">
            <h2>🧠 Custom Models (Optional)</h2>
            <div style="margin-bottom: 1.5rem; color: #a0a0b8;">
                Train AI models on YOUR preferences! Upload specialized models for different content types.
            </div>
            
            <!-- UI Model -->
            <div style="margin-bottom: 1.5rem; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <strong>💻 UI Model</strong>
                    <span id="ui-model-status-text" style="font-size: 0.9rem; color: #a0a0b8;">Not loaded</span>
                </div>
                <div style="display: flex; gap: 1rem;">
                    <input type="file" id="ui-model-input" accept=".pth" style="display: none;">
                    <button 
                        onclick="document.getElementById('ui-model-input').click()" 
                        class="btn-primary" 
                        style="padding: 0.5rem 1rem; font-size: 0.85rem;"
                    >
                        📤 Upload UI Model
                    </button>
                    <button 
                        id="remove-ui-model-btn"
                        onclick="removeModel('ui')" 
                        class="btn-primary" 
                        style="padding: 0.5rem 1rem; font-size: 0.85rem; background: #ef4444; display: none;"
                    >
                        🗑️ Remove
                    </button>
                </div>
                <div style="font-size: 0.8rem; color: #a0a0b8; margin-top: 0.5rem;">
                    Train on: UI interfaces, node graphs, software screenshots
                </div>
            </div>
            
            <!-- Art Model -->
            <div style="margin-bottom: 1.5rem; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <strong>🎨 Art Model</strong>
                    <span id="art-model-status-text" style="font-size: 0.9rem; color: #a0a0b8;">Not loaded</span>
                </div>
                <div style="display: flex; gap: 1rem;">
                    <input type="file" id="art-model-input" accept=".pth" style="display: none;">
                    <button 
                        onclick="document.getElementById('art-model-input').click()" 
                        class="btn-primary" 
                        style="padding: 0.5rem 1rem; font-size: 0.85rem;"
                    >
                        📤 Upload Art Model
                    </button>
                    <button 
                        id="remove-art-model-btn"
                        onclick="removeModel('art')" 
                        class="btn-primary" 
                        style="padding: 0.5rem 1rem; font-size: 0.85rem; background: #ef4444; display: none;"
                    >
                        🗑️ Remove
                    </button>
                </div>
                <div style="font-size: 0.8rem; color: #a0a0b8; margin-top: 0.5rem;">
                    Train on: Renders, outputs, fullscreen art, visuals
                </div>
            </div>
            
            <!-- Face Model (Optional) -->
            <div style="margin-bottom: 1.5rem; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <strong>😊 Face Model</strong>
                    <span id="face-model-status-text" style="font-size: 0.9rem; color: #a0a0b8;">Not loaded</span>
                </div>
                <div style="display: flex; gap: 1rem;">
                    <input type="file" id="face-model-input" accept=".pth" style="display: none;">
                    <button 
                        onclick="document.getElementById('face-model-input').click()" 
                        class="btn-primary" 
                        style="padding: 0.5rem 1rem; font-size: 0.85rem;"
                    >
                        📤 Upload Face Model
                    </button>
                    <button 
                        id="remove-face-model-btn"
                        onclick="removeModel('face')" 
                        class="btn-primary" 
                        style="padding: 0.5rem 1rem; font-size: 0.85rem; background: #ef4444; display: none;"
                    >
                        🗑️ Remove
                    </button>
                </div>
                <div style="font-size: 0.8rem; color: #a0a0b8; margin-top: 0.5rem;">
                    Train on: Face shots, expressions you prefer
                </div>
            </div>
            
            <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(255,107,53,0.1); border: 1px solid rgba(255,107,53,0.3); border-radius: 8px;">
                <strong>📚 Training Guide</strong><br>
                <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #a0a0b8;">
                    See <code>DUAL_MODEL_TRAINING_GUIDE.md</code> for step-by-step instructions.
                    Learn how to train separate models for UI and Art content!
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>2. Upload Video</h2>
            <div id="upload-zone" class="upload-zone">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📹</div>
                <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">Drop your video here</div>
                <div style="color: #a0a0b8; font-size: 0.9rem;">or click to browse</div>
                <input type="file" id="video-input" accept="video/*" style="display: none;">
            </div>
            <div id="video-preview"></div>
        </div>
        
        <div class="card">
            <h2>3. Choose Mode</h2>
            <div class="radio-group">
                <div class="radio-option">
                    <input type="radio" id="faces" name="mode" value="faces" checked>
                    <label for="faces" class="radio-label">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">😊</div>
                        <div style="font-weight: 700;">Faces</div>
                        <div style="font-size: 0.85rem; color: #a0a0b8; margin-top: 0.5rem;">Extract talking heads</div>
                    </label>
                </div>
                <div class="radio-option">
                    <input type="radio" id="screens_ui" name="mode" value="screens_ui">
                    <label for="screens_ui" class="radio-label">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">💻</div>
                        <div style="font-weight: 700;">Screens: UI</div>
                        <div style="font-size: 0.85rem; color: #a0a0b8; margin-top: 0.5rem;">Interfaces, nodes, software</div>
                    </label>
                </div>
                <div class="radio-option">
                    <input type="radio" id="screens_art" name="mode" value="screens_art">
                    <label for="screens_art" class="radio-label">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎨</div>
                        <div style="font-weight: 700;">Screens: Art</div>
                        <div style="font-size: 0.85rem; color: #a0a0b8; margin-top: 0.5rem;">Renders, outputs, visuals</div>
                    </label>
                </div>
            </div>
            
            <div class="radio-group" style="margin-top: 1rem;">
                <div class="radio-option">
                    <input type="radio" id="screens_mixed" name="mode" value="screens_mixed">
                    <label for="screens_mixed" class="radio-label">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🖥️</div>
                        <div style="font-weight: 700;">Screens: Mixed</div>
                        <div style="font-size: 0.85rem; color: #a0a0b8; margin-top: 0.5rem;">50/50 UI + Art blend</div>
                    </label>
                </div>
                <div class="radio-option">
                    <input type="radio" id="both" name="mode" value="both">
                    <label for="both" class="radio-label">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎯</div>
                        <div style="font-weight: 700;">Everything</div>
                        <div style="font-size: 0.85rem; color: #a0a0b8; margin-top: 0.5rem;">Faces + UI + Art</div>
                    </label>
                </div>
                <div class="radio-option" style="opacity: 0; pointer-events: none;">
                    <!-- Spacer for grid alignment -->
                </div>
            </div>
            
            <!-- Background Removal Option (Only for Faces) -->
            <div id="bg-removal-option" style="margin-top: 1.5rem; padding: 1.5rem; background: rgba(255, 107, 53, 0.1); border: 1px solid rgba(255, 107, 53, 0.3); border-radius: 12px; display: none;">
                <label style="display: flex; align-items: center; cursor: pointer;">
                    <input type="checkbox" id="remove-bg" style="width: 20px; height: 20px; margin-right: 1rem; cursor: pointer;">
                    <div>
                        <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 0.25rem;">✨ Remove Background</div>
                        <div style="font-size: 0.85rem; color: #a0a0b8;">
                            Extract faces with transparent background (PNG format)
                        </div>
                    </div>
                </label>
            </div>
            
            <!-- Number of Assets Slider -->
            <div style="margin-top: 1.5rem; padding: 1.5rem; background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px;">
                <label style="display: block; margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                        <div style="font-weight: 700; font-size: 1.1rem;">📊 Number of Assets to Extract</div>
                        <div id="asset-count-display" style="font-size: 1.5rem; font-weight: 700; color: #ff6b35;">10</div>
                    </div>
                    <input 
                        type="range" 
                        id="asset-count-slider" 
                        min="5" 
                        max="50" 
                        value="10" 
                        step="5"
                        style="width: 100%; height: 8px; background: linear-gradient(90deg, #ff6b35, #f7931e); border-radius: 4px; cursor: pointer; -webkit-appearance: none; appearance: none;"
                    >
                    <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.85rem; color: #a0a0b8;">
                        <span>5 (Fast)</span>
                        <span>25 (Balanced)</span>
                        <span>50 (Comprehensive)</span>
                    </div>
                </label>
                <div style="font-size: 0.85rem; color: #a0a0b8; margin-top: 0.75rem;">
                    More assets = more options but slower processing
                </div>
            </div>
        </div>
        
        <div class="action-buttons">
            <button id="add-queue-btn" class="btn-outline" disabled>+ Add to Queue</button>
            <button id="extract-now-btn" class="btn-primary" disabled>Extract Now</button>
        </div>

        <div id="queue-panel" class="queue-panel">
            <div class="queue-header">
                <h2>Extraction Queue (<span id="queue-count">0</span>)</h2>
                <div style="display: flex; gap: 0.5rem;">
                    <button id="start-all-btn" class="btn-sm" style="background: #ff6b35; color: white; display: none;">Start All</button>
                    <button id="clear-done-btn" class="btn-sm" style="background: rgba(255,255,255,0.1); color: #a0a0b8; display: none;">Clear Done</button>
                </div>
            </div>
            <div id="queue-items">
                <div class="queue-empty">Configure a video above and add it to the queue</div>
            </div>
        </div>
    </div>
    
    <script>
        let videoFile = null;

        const uploadZone = document.getElementById('upload-zone');
        const videoInput = document.getElementById('video-input');
        const videoPreview = document.getElementById('video-preview');
        const addQueueBtn = document.getElementById('add-queue-btn');
        const extractNowBtn = document.getElementById('extract-now-btn');
        const statusDiv = document.getElementById('status');

        // Upload handling
        uploadZone.onclick = () => videoInput.click();

        uploadZone.ondragover = (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        };

        uploadZone.ondragleave = () => {
            uploadZone.classList.remove('dragover');
        };

        uploadZone.ondrop = (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('video/')) {
                handleVideoUpload(file);
            }
        };

        videoInput.onchange = (e) => {
            const file = e.target.files[0];
            if (file) handleVideoUpload(file);
        };

        function handleVideoUpload(file) {
            videoFile = file;
            const url = URL.createObjectURL(file);

            videoPreview.innerHTML = `
                <video controls src="${url}"></video>
                <div style="color: #a0a0b8; margin-top: 1rem;">
                    ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)
                </div>
            `;

            addQueueBtn.disabled = false;
            extractNowBtn.disabled = false;
            showStatus('Video uploaded successfully! Ready to add to queue.', 'success');
        }

        // Show/hide background removal option based on mode
        const modeRadios = document.querySelectorAll('input[name="mode"]');
        const bgRemovalOption = document.getElementById('bg-removal-option');

        modeRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                if (e.target.value === 'faces' || e.target.value === 'both') {
                    bgRemovalOption.style.display = 'block';
                } else {
                    bgRemovalOption.style.display = 'none';
                }
            });
        });

        // Asset count slider
        const assetCountSlider = document.getElementById('asset-count-slider');
        const assetCountDisplay = document.getElementById('asset-count-display');

        assetCountSlider.addEventListener('input', (e) => {
            assetCountDisplay.textContent = e.target.value;
        });

        function showStatus(message, type) {
            statusDiv.innerHTML = `
                <div class="status-message status-${type}">
                    ${type === 'success' ? '&#10003;' : '&#9888;'} ${message}
                </div>
            `;
            setTimeout(() => { statusDiv.innerHTML = ''; }, 5000);
        }

        // ============= QUEUE MANAGER =============
        const QueueManager = {
            jobs: [],
            eventSource: null,
            expandedResultId: null,

            init() {
                this.connect();
                this.loadExisting();
            },

            connect() {
                if (this.eventSource) this.eventSource.close();
                this.eventSource = new EventSource('/api/sse/queue');
                this.eventSource.onmessage = (e) => {
                    try { this.handleEvent(JSON.parse(e.data)); }
                    catch (err) { console.error('SSE parse error', err); }
                };
                this.eventSource.onerror = () => {
                    this.eventSource.close();
                    setTimeout(() => this.connect(), 3000);
                };
            },

            async loadExisting() {
                try {
                    const resp = await fetch('/api/queue');
                    const data = await resp.json();
                    if (data.success) {
                        this.jobs = data.jobs;
                        this.render();
                    }
                } catch (err) {
                    console.error('Failed to load queue', err);
                }
            },

            handleEvent(data) {
                if (data.type === 'heartbeat') return;

                if (data.type === 'job_added') {
                    const idx = this.jobs.findIndex(j => j.id === data.job.id);
                    if (idx >= 0) this.jobs[idx] = data.job;
                    else this.jobs.push(data.job);
                }
                else if (data.type === 'job_started' || data.type === 'job_completed' ||
                         data.type === 'job_error' || data.type === 'job_cancelled' ||
                         data.type === 'job_updated') {
                    const idx = this.jobs.findIndex(j => j.id === data.job_id);
                    if (idx >= 0 && data.job) this.jobs[idx] = data.job;
                }
                else if (data.type === 'job_progress') {
                    const job = this.jobs.find(j => j.id === data.job_id);
                    if (job) {
                        job.progress = data.progress;
                        job.progress_message = data.message;
                        // Fast update: just update progress bar + text, no full re-render
                        const bar = document.getElementById(`progress-fill-${data.job_id}`);
                        const txt = document.getElementById(`progress-text-${data.job_id}`);
                        if (bar) bar.style.width = data.progress + '%';
                        if (txt) txt.textContent = data.message;
                        this.updateCounts();
                        return;
                    }
                }
                else if (data.type === 'job_removed') {
                    this.jobs = this.jobs.filter(j => j.id !== data.job_id);
                }

                this.render();
            },

            async addJob(andStart) {
                if (!videoFile) { showStatus('Upload a video first', 'error'); return; }

                const mode = document.querySelector('input[name="mode"]:checked').value;
                const removeBg = document.getElementById('remove-bg')?.checked || false;
                const assetCount = parseInt(assetCountSlider.value);

                addQueueBtn.disabled = true;
                extractNowBtn.disabled = true;

                const formData = new FormData();
                formData.append('video', videoFile);
                formData.append('mode', mode);
                formData.append('remove_bg', removeBg.toString());
                formData.append('asset_count', assetCount.toString());

                try {
                    const resp = await fetch('/api/queue/add', { method: 'POST', body: formData });
                    const data = await resp.json();

                    if (data.success) {
                        showStatus(`Added "${videoFile.name}" to queue`, 'success');
                        this.resetForm();

                        if (andStart) {
                            await fetch('/api/queue/start', { method: 'POST' });
                        }
                    } else {
                        showStatus('Error: ' + (data.error || 'Failed to add'), 'error');
                        addQueueBtn.disabled = false;
                        extractNowBtn.disabled = false;
                    }
                } catch (err) {
                    showStatus('Error: ' + err.message, 'error');
                    addQueueBtn.disabled = false;
                    extractNowBtn.disabled = false;
                }
            },

            resetForm() {
                videoFile = null;
                videoInput.value = '';
                videoPreview.innerHTML = '';
                addQueueBtn.disabled = true;
                extractNowBtn.disabled = true;
                // Keep mode & slider as-is for convenience
            },

            async startAll() {
                try {
                    await fetch('/api/queue/start', { method: 'POST' });
                    showStatus('Processing started', 'success');
                } catch (err) {
                    showStatus('Error starting: ' + err.message, 'error');
                }
            },

            async cancelJob(id) {
                try {
                    await fetch(`/api/queue/${id}/cancel`, { method: 'POST' });
                } catch (err) {
                    showStatus('Error: ' + err.message, 'error');
                }
            },

            async removeJob(id) {
                try {
                    await fetch(`/api/queue/${id}/remove`, { method: 'POST' });
                } catch (err) {
                    showStatus('Error: ' + err.message, 'error');
                }
            },

            async retryJob(id) {
                try {
                    const resp = await fetch(`/api/queue/${id}/retry`, { method: 'POST' });
                    const data = await resp.json();
                    if (!data.success) showStatus('Error: ' + (data.error || 'Retry failed'), 'error');
                } catch (err) {
                    showStatus('Error: ' + err.message, 'error');
                }
            },

            clearDone() {
                const done = this.jobs.filter(j => ['completed','cancelled','error'].includes(j.status));
                done.forEach(j => this.removeJob(j.id));
            },

            toggleResults(jobId) {
                if (this.expandedResultId === jobId) {
                    this.expandedResultId = null;
                } else {
                    this.expandedResultId = jobId;
                }
                this.render();
            },

            async loadResults(jobId) {
                try {
                    const resp = await fetch(`/api/queue/${jobId}/results`);
                    const data = await resp.json();
                    if (data.success) {
                        const job = this.jobs.find(j => j.id === jobId);
                        if (job) job.result = data.result;
                        this.render();
                    }
                } catch (err) {
                    console.error('Failed to load results', err);
                }
            },

            updateCounts() {
                const total = this.jobs.length;
                const queued = this.jobs.filter(j => j.status === 'queued').length;
                const done = this.jobs.filter(j => ['completed','cancelled','error'].includes(j.status)).length;

                document.getElementById('queue-count').textContent = total;
                document.getElementById('start-all-btn').style.display = queued > 0 ? 'inline-block' : 'none';
                document.getElementById('clear-done-btn').style.display = done > 0 ? 'inline-block' : 'none';
            },

            render() {
                const container = document.getElementById('queue-items');
                this.updateCounts();

                if (this.jobs.length === 0) {
                    container.innerHTML = '<div class="queue-empty">Configure a video above and add it to the queue</div>';
                    return;
                }

                container.innerHTML = this.jobs.map(job => this.renderJob(job)).join('');
            },

            renderJob(job) {
                const statusIcons = {
                    queued: '<span style="color:#6b7280">&#9202;</span>',
                    processing: '<span class="spin-slow" style="color:#ff6b35;display:inline-block">&#9673;</span>',
                    completed: '<span style="color:#4ade80">&#10003;</span>',
                    error: '<span style="color:#ef4444">&#10007;</span>',
                    cancelled: '<span style="color:#6b7280">&#8212;</span>'
                };

                const modeLabels = {
                    faces: 'Faces', screens_ui: 'UI', screens_art: 'Art',
                    screens_mixed: 'Mixed', both: 'All'
                };

                let progressHtml = '';
                if (job.status === 'processing') {
                    progressHtml = `
                        <div class="progress-bar-track">
                            <div class="progress-bar-fill" id="progress-fill-${job.id}" style="width:${job.progress || 0}%"></div>
                        </div>
                        <div class="progress-text" id="progress-text-${job.id}">${job.progress_message || 'Starting...'}</div>
                    `;
                }

                let detailHtml = '';
                if (job.status === 'queued') {
                    detailHtml = '<div class="queue-item-detail">Waiting...</div>';
                } else if (job.status === 'error') {
                    detailHtml = `<div class="queue-item-detail error-msg">${job.error_message || 'Unknown error'}</div>`;
                } else if (job.status === 'completed') {
                    const r = job.result;
                    const summary = r ? `${r.asset_count} assets (Best: ${r.top_score?.toFixed(2) || '?'})` : (job.progress_message || 'Done');
                    detailHtml = `<div class="queue-item-detail" style="color:#4ade80">${summary}</div>`;
                } else if (job.status === 'cancelled') {
                    detailHtml = '<div class="queue-item-detail">Cancelled</div>';
                }

                // Action buttons
                let actions = '';
                if (job.status === 'queued') {
                    actions = `<button class="btn-sm" style="background:#ef4444;color:white" onclick="QueueManager.cancelJob('${job.id}')">Cancel</button>`;
                } else if (job.status === 'processing') {
                    actions = `<button class="btn-sm" style="background:#ef4444;color:white" onclick="QueueManager.cancelJob('${job.id}')">Cancel</button>`;
                } else if (job.status === 'completed') {
                    actions = `
                        <a href="/api/download-all/${job.id}" download><button class="btn-sm" style="background:#4ade80;color:#000">ZIP</button></a>
                        <button class="btn-sm" style="background:rgba(255,255,255,0.1);color:#a0a0b8" onclick="QueueManager.removeJob('${job.id}')">Remove</button>
                    `;
                } else if (job.status === 'error') {
                    actions = `
                        <button class="btn-sm" style="background:#ff6b35;color:white" onclick="QueueManager.retryJob('${job.id}')">Retry</button>
                        <button class="btn-sm" style="background:rgba(255,255,255,0.1);color:#a0a0b8" onclick="QueueManager.removeJob('${job.id}')">Remove</button>
                    `;
                } else if (job.status === 'cancelled') {
                    actions = `<button class="btn-sm" style="background:rgba(255,255,255,0.1);color:#a0a0b8" onclick="QueueManager.removeJob('${job.id}')">Remove</button>`;
                }

                // Results toggle for completed jobs
                let resultsHtml = '';
                if (job.status === 'completed') {
                    const isOpen = this.expandedResultId === job.id;
                    resultsHtml = `
                        <div class="results-toggle">
                            <button class="results-toggle-btn" onclick="QueueManager.toggleResults('${job.id}')">
                                ${isOpen ? '&#9660; Hide Results' : '&#9654; View Results'}
                            </button>
                        </div>
                    `;

                    if (isOpen && job.result && job.result.assets) {
                        resultsHtml += `
                            <div class="inline-results open">
                                <div class="results-grid">
                                    ${job.result.assets.map(a => `
                                        <div class="asset-card">
                                            <img src="${a.preview}" alt="Asset" class="asset-image" loading="lazy">
                                            <div class="asset-info">
                                                <div class="asset-score">Score: ${a.score.toFixed(2)}</div>
                                                <div style="font-size:0.75rem;color:#a0a0b8">
                                                    ${a.timestamp} | ${a.type}
                                                </div>
                                                <a href="${a.download_url}" download>
                                                    <button class="download-btn">Download</button>
                                                </a>
                                            </div>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        `;
                    } else if (isOpen && (!job.result || !job.result.assets)) {
                        // Need to load results
                        this.loadResults(job.id);
                        resultsHtml += '<div class="inline-results open"><div class="queue-item-detail">Loading results...</div></div>';
                    }
                }

                return `
                    <div class="queue-item ${job.status}">
                        <div class="queue-item-header">
                            <div class="queue-item-status">${statusIcons[job.status] || '?'}</div>
                            <div class="queue-item-name" title="${job.filename}">${job.filename}</div>
                            <div class="queue-item-tags">
                                <span class="tag tag-mode">${modeLabels[job.mode] || job.mode}</span>
                                <span class="tag tag-count">${job.asset_count}</span>
                                <span class="tag tag-size">${job.file_size_mb} MB</span>
                            </div>
                            <div class="queue-item-actions">${actions}</div>
                        </div>
                        ${progressHtml}
                        ${detailHtml}
                        ${resultsHtml}
                    </div>
                `;
            }
        };

        // Wire up buttons
        addQueueBtn.onclick = () => QueueManager.addJob(false);
        extractNowBtn.onclick = () => QueueManager.addJob(true);
        document.getElementById('start-all-btn').onclick = () => QueueManager.startAll();
        document.getElementById('clear-done-btn').onclick = () => QueueManager.clearDone();

        // Init queue manager
        QueueManager.init();

        // ============= CUSTOM MODEL FUNCTIONS =============

        const uiModelInput = document.getElementById('ui-model-input');
        const artModelInput = document.getElementById('art-model-input');
        const faceModelInput = document.getElementById('face-model-input');

        const uiModelStatus = document.getElementById('ui-model-status-text');
        const artModelStatus = document.getElementById('art-model-status-text');
        const faceModelStatus = document.getElementById('face-model-status-text');

        const removeUiModelBtn = document.getElementById('remove-ui-model-btn');
        const removeArtModelBtn = document.getElementById('remove-art-model-btn');
        const removeFaceModelBtn = document.getElementById('remove-face-model-btn');

        async function checkModelStatus() {
            try {
                const response = await fetch('/api/model-status');
                const data = await response.json();

                if (data.ui_model_loaded) {
                    uiModelStatus.textContent = 'Loaded';
                    uiModelStatus.style.color = '#4ade80';
                    removeUiModelBtn.style.display = 'inline-block';
                } else {
                    uiModelStatus.textContent = 'Not loaded';
                    uiModelStatus.style.color = '#a0a0b8';
                    removeUiModelBtn.style.display = 'none';
                }

                if (data.art_model_loaded) {
                    artModelStatus.textContent = 'Loaded';
                    artModelStatus.style.color = '#4ade80';
                    removeArtModelBtn.style.display = 'inline-block';
                } else {
                    artModelStatus.textContent = 'Not loaded';
                    artModelStatus.style.color = '#a0a0b8';
                    removeArtModelBtn.style.display = 'none';
                }

                if (data.face_model_loaded) {
                    faceModelStatus.textContent = 'Loaded';
                    faceModelStatus.style.color = '#4ade80';
                    removeFaceModelBtn.style.display = 'inline-block';
                } else {
                    faceModelStatus.textContent = 'Not loaded';
                    faceModelStatus.style.color = '#a0a0b8';
                    removeFaceModelBtn.style.display = 'none';
                }
            } catch (error) {
                console.error('Error checking model status:', error);
            }
        }

        async function uploadModel(file, modelType) {
            if (!file) return;

            if (!file.name.endsWith('.pth')) {
                showStatus('Error: Model must be a .pth file', 'error');
                return;
            }

            showStatus(`Uploading ${modelType.toUpperCase()} model...`, 'success');

            const formData = new FormData();
            formData.append('model', file);
            formData.append('model_type', modelType);

            try {
                const response = await fetch('/api/upload-model', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.success) {
                    showStatus(`${modelType.toUpperCase()} model loaded!`, 'success');
                    checkModelStatus();
                } else {
                    showStatus('Error: ' + (data.error || 'Failed to load model'), 'error');
                }
            } catch (error) {
                showStatus('Error uploading model: ' + error.message, 'error');
            }
        }

        uiModelInput.onchange = (e) => uploadModel(e.target.files[0], 'ui');
        artModelInput.onchange = (e) => uploadModel(e.target.files[0], 'art');
        faceModelInput.onchange = (e) => uploadModel(e.target.files[0], 'face');

        async function removeModel(modelType) {
            if (!confirm(`Remove ${modelType.toUpperCase()} model and use default scoring?`)) {
                return;
            }

            try {
                const response = await fetch('/api/remove-model', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model_type: modelType })
                });

                const data = await response.json();

                if (data.success) {
                    showStatus(`${modelType.toUpperCase()} model removed.`, 'success');
                    checkModelStatus();
                } else {
                    showStatus('Error: ' + (data.error || 'Failed to remove model'), 'error');
                }
            } catch (error) {
                showStatus('Error: ' + error.message, 'error');
            }
        }

        checkModelStatus();
    </script>
</body>
</html>'''


# Create index.html file
with open('index.html', 'w', encoding='utf-8') as f:
    f.write(HTML_TEMPLATE)


# ============= MAIN =============

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Thumbnail Asset Extractor - Modernized Pipeline")
    print("="*60)
    print(f"\n* OpenCV version: {cv2.__version__}")
    print(f"* Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"* Output folder: {os.path.abspath(OUTPUT_FOLDER)}")

    # Report vision library status
    from vision.model_registry import get_status
    for key, info in get_status().items():
        status = "available" if info['available'] else "not installed"
        print(f"* {info['name']}: {status}")

    print(f"\n> Server starting on: http://localhost:5000")
    print(f"   Open this URL in your browser to use the tool")
    print("\n" + "="*60 + "\n")

    app.run(debug=False, host='0.0.0.0', port=PORT, threaded=True)
