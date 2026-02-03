"""
Video Capture Abstraction.

Provides a unified interface for capturing frames from 
cameras, video files, or RTSP streams.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Result from frame capture."""
    
    frame: Optional[np.ndarray]
    frame_number: int
    timestamp_ms: float
    success: bool
    
    @property
    def is_valid(self) -> bool:
        """Check if frame was captured successfully."""
        return self.success and self.frame is not None


class VideoSource:
    """
    Video capture abstraction supporting multiple sources.
    
    Example:
        # Webcam
        source = VideoSource(0)
        
        # Video file
        source = VideoSource("video.mp4")
        
        # RTSP stream
        source = VideoSource("rtsp://192.168.1.100:554/stream")
        
        with source:
            while True:
                result = source.read()
                if not result.is_valid:
                    break
                process(result.frame)
    """
    
    def __init__(
        self,
        source: Union[int, str],
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        buffer_size: int = 1
    ):
        """
        Initialize video source.
        
        Args:
            source: Camera index, video file path, or RTSP URL.
            width: Desired frame width (None = use default).
            height: Desired frame height (None = use default).
            fps: Desired FPS (None = use default).
            buffer_size: OpenCV buffer size (1 = minimal latency).
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count: int = 0
        self._is_opened: bool = False
    
    def open(self) -> bool:
        """
        Open the video source.
        
        Returns:
            True if source opened successfully.
        """
        if self._is_opened:
            return True
        
        self._cap = cv2.VideoCapture(self.source)
        
        if not self._cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            return False
        
        # Set buffer size for lower latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        
        # Set resolution if specified
        if self.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        self._is_opened = True
        self._frame_count = 0
        
        # Log actual settings
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(
            f"Opened video source: {self.source} "
            f"({actual_width}x{actual_height} @ {actual_fps:.1f}fps)"
        )
        
        return True
    
    def read(self) -> FrameResult:
        """
        Read a frame from the source.
        
        Returns:
            FrameResult with frame data and metadata.
        """
        if not self._is_opened or self._cap is None:
            return FrameResult(
                frame=None,
                frame_number=0,
                timestamp_ms=0.0,
                success=False
            )
        
        success, frame = self._cap.read()
        timestamp = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        
        if success:
            self._frame_count += 1
        
        return FrameResult(
            frame=frame,
            frame_number=self._frame_count,
            timestamp_ms=timestamp,
            success=success
        )
    
    def close(self) -> None:
        """Release the video source."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_opened = False
        logger.info(f"Closed video source: {self.source}")
    
    @property
    def is_opened(self) -> bool:
        """Check if source is currently open."""
        return self._is_opened
    
    @property
    def frame_count(self) -> int:
        """Number of frames read so far."""
        return self._frame_count
    
    def get_properties(self) -> dict:
        """Get current video properties."""
        if not self._is_opened or self._cap is None:
            return {}
        
        return {
            "width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self._cap.get(cv2.CAP_PROP_FPS),
            "frame_count": self._frame_count,
            "backend": self._cap.getBackendName()
        }
    
    def __enter__(self) -> "VideoSource":
        self.open()
        return self
    
    def __exit__(self, *args) -> None:
        self.close()
