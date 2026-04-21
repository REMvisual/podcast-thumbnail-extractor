"""
Hardware-accelerated video decoding using PyAV with NVDEC.

Falls back to OpenCV VideoCapture when PyAV or NVDEC is unavailable.

Usage:
    decoder = get_video_decoder()
    for frame_bgr, timestamp, frame_num in decoder.decode_frames(path, timestamps):
        process(frame_bgr)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Protocol
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class DecodedFrame:
    """A decoded video frame."""
    frame: np.ndarray  # BGR numpy array
    timestamp: float  # seconds
    frame_number: int


class VideoDecoderProtocol(Protocol):
    """Protocol for video decoder implementations."""
    def decode_frames(
        self,
        video_path: str,
        timestamps: Optional[List[float]] = None,
        start: float = 0.0,
        end: Optional[float] = None,
        step: float = 1.0,
    ) -> Generator[DecodedFrame, None, None]: ...

    def get_video_info(self, video_path: str) -> dict: ...


class PyAVDecoder:
    """Hardware-accelerated video decoder using PyAV.

    Attempts NVDEC (CUDA) hardware decoding first, falls back to
    software decoding if GPU decode is unavailable.
    """

    def __init__(self, use_hw: bool = True):
        """
        Args:
            use_hw: Attempt hardware (NVDEC) decoding.
        """
        import av
        self._av = av
        self._use_hw = use_hw
        self._hw_available = None
        logger.info("PyAV decoder initialized (hw=%s)", use_hw)

    def _check_hw(self) -> bool:
        """Check if NVDEC hardware decoding is available."""
        if self._hw_available is not None:
            return self._hw_available

        try:
            import av
            # Check for CUDA/NVDEC codec availability
            codecs = [c.name for c in av.codecs_available if 'cuvid' in c.name or 'nvdec' in c.name]
            self._hw_available = len(codecs) > 0
            if self._hw_available:
                logger.info("NVDEC hardware decode available: %s", codecs)
            else:
                logger.info("NVDEC not available, using software decode")
        except Exception:
            self._hw_available = False

        return self._hw_available

    def get_video_info(self, video_path: str) -> dict:
        """Get video metadata."""
        container = self._av.open(video_path)
        stream = container.streams.video[0]
        info = {
            "duration": float(container.duration / self._av.time_base) if container.duration else 0,
            "fps": float(stream.average_rate) if stream.average_rate else 30.0,
            "width": stream.width,
            "height": stream.height,
            "codec": stream.codec_context.name,
            "frame_count": stream.frames or 0,
            "hw_decode": self._use_hw and self._check_hw(),
        }
        container.close()
        return info

    def decode_frames(
        self,
        video_path: str,
        timestamps: Optional[List[float]] = None,
        start: float = 0.0,
        end: Optional[float] = None,
        step: float = 1.0,
    ) -> Generator[DecodedFrame, None, None]:
        """Decode frames at specified timestamps or intervals.

        Args:
            video_path: Path to video file.
            timestamps: Specific timestamps to decode (overrides start/end/step).
            start: Start time in seconds.
            end: End time in seconds (None = entire video).
            step: Interval between frames in seconds (when timestamps is None).

        Yields:
            DecodedFrame objects with BGR numpy arrays.
        """
        try:
            # Attempt hardware decode
            if self._use_hw and self._check_hw():
                yield from self._decode_hw(video_path, timestamps, start, end, step)
            else:
                yield from self._decode_sw(video_path, timestamps, start, end, step)
        except Exception as e:
            logger.warning("PyAV decode failed, trying software: %s", e)
            yield from self._decode_sw(video_path, timestamps, start, end, step)

    def _decode_sw(self, video_path, timestamps, start, end, step):
        """Software decode path."""
        container = self._av.open(video_path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        time_base = stream.time_base

        if timestamps:
            # Seek to each timestamp
            for ts in sorted(timestamps):
                target_pts = int(ts / time_base)
                container.seek(target_pts, stream=stream)

                for frame in container.decode(stream):
                    frame_ts = float(frame.pts * time_base) if frame.pts else 0
                    frame_num = int(frame_ts * fps)
                    bgr = frame.to_ndarray(format='bgr24')
                    yield DecodedFrame(frame=bgr, timestamp=frame_ts, frame_number=frame_num)
                    break  # Only need the first frame after seek
        else:
            # Sequential decode with step
            duration = float(container.duration / 1000000) if container.duration else 0
            if end is None:
                end = duration

            current = start
            while current <= end:
                target_pts = int(current / time_base)
                container.seek(target_pts, stream=stream)

                for frame in container.decode(stream):
                    frame_ts = float(frame.pts * time_base) if frame.pts else current
                    frame_num = int(frame_ts * fps)
                    bgr = frame.to_ndarray(format='bgr24')
                    yield DecodedFrame(frame=bgr, timestamp=frame_ts, frame_number=frame_num)
                    break

                current += step

        container.close()

    def _decode_hw(self, video_path, timestamps, start, end, step):
        """Hardware-accelerated decode path using NVDEC."""
        # Try to open with hardware codec
        try:
            codec_name = None
            # Probe for codec
            probe = self._av.open(video_path)
            src_codec = probe.streams.video[0].codec_context.name
            probe.close()

            hw_codecs = {
                'h264': 'h264_cuvid',
                'hevc': 'hevc_cuvid',
                'vp9': 'vp9_cuvid',
                'mpeg4': 'mpeg4_cuvid',
            }
            codec_name = hw_codecs.get(src_codec)

            if not codec_name:
                logger.info("No NVDEC codec for %s, using software", src_codec)
                yield from self._decode_sw(video_path, timestamps, start, end, step)
                return

            container = self._av.open(video_path, options={'hwaccel': 'cuda'})
            stream = container.streams.video[0]
            stream.codec_context.codec = self._av.codec.Codec(codec_name, 'r')

        except Exception as e:
            logger.info("NVDEC init failed (%s), falling back to software", e)
            yield from self._decode_sw(video_path, timestamps, start, end, step)
            return

        fps = float(stream.average_rate) if stream.average_rate else 30.0
        time_base = stream.time_base

        try:
            if timestamps:
                for ts in sorted(timestamps):
                    target_pts = int(ts / time_base)
                    container.seek(target_pts, stream=stream)
                    for frame in container.decode(stream):
                        frame_ts = float(frame.pts * time_base) if frame.pts else 0
                        frame_num = int(frame_ts * fps)
                        bgr = frame.to_ndarray(format='bgr24')
                        yield DecodedFrame(frame=bgr, timestamp=frame_ts, frame_number=frame_num)
                        break
            else:
                duration = float(container.duration / 1000000) if container.duration else 0
                if end is None:
                    end = duration
                current = start
                while current <= end:
                    target_pts = int(current / time_base)
                    container.seek(target_pts, stream=stream)
                    for frame in container.decode(stream):
                        frame_ts = float(frame.pts * time_base) if frame.pts else current
                        frame_num = int(frame_ts * fps)
                        bgr = frame.to_ndarray(format='bgr24')
                        yield DecodedFrame(frame=bgr, timestamp=frame_ts, frame_number=frame_num)
                        break
                    current += step
        finally:
            container.close()


class OpenCVDecoder:
    """Fallback decoder using OpenCV VideoCapture (CPU only)."""

    def get_video_info(self, video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        info = {
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1),
            "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "codec": "opencv",
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "hw_decode": False,
        }
        cap.release()
        return info

    def decode_frames(
        self,
        video_path: str,
        timestamps: Optional[List[float]] = None,
        start: float = 0.0,
        end: Optional[float] = None,
        step: float = 1.0,
    ) -> Generator[DecodedFrame, None, None]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        if timestamps:
            for ts in sorted(timestamps):
                cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
                ret, frame = cap.read()
                if ret and frame is not None:
                    yield DecodedFrame(
                        frame=frame,
                        timestamp=ts,
                        frame_number=int(ts * fps),
                    )
        else:
            if end is None:
                end = duration
            current = start
            while current <= end:
                cap.set(cv2.CAP_PROP_POS_MSEC, current * 1000)
                ret, frame = cap.read()
                if ret and frame is not None:
                    yield DecodedFrame(
                        frame=frame,
                        timestamp=current,
                        frame_number=int(current * fps),
                    )
                current += step

        cap.release()


# Singleton
_decoder_instance: Optional[VideoDecoderProtocol] = None


def get_video_decoder(backend: str = "auto") -> VideoDecoderProtocol:
    """Get video decoder instance (singleton).

    Args:
        backend: "pyav", "opencv", or "auto" (try PyAV first).

    Returns:
        Video decoder implementing VideoDecoderProtocol.
    """
    global _decoder_instance

    if _decoder_instance is not None:
        return _decoder_instance

    if backend == "auto":
        try:
            _decoder_instance = PyAVDecoder()
            return _decoder_instance
        except ImportError:
            logger.warning("PyAV not available, falling back to OpenCV decoder")
            _decoder_instance = OpenCVDecoder()
            return _decoder_instance

    if backend == "pyav":
        _decoder_instance = PyAVDecoder()
    elif backend == "opencv":
        _decoder_instance = OpenCVDecoder()
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return _decoder_instance
