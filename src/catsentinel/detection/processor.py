"""
Detection Pipeline Processor.

Orchestrates the capture -> inference -> notification pipeline
with ROI filtering and visualization support.
"""

import asyncio
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..engines.base import Detection, InferenceEngine, InferenceResult
from ..notifications.base import Notifier, NotificationPayload
from ..utils.config import ROIConfig

logger = logging.getLogger(__name__)


class DetectionProcessor:
    """
    Detection pipeline orchestrator.
    
    Handles:
        - ROI (Region of Interest) filtering
        - Detection visualization
        - Async notification dispatch
    
    Example:
        processor = DetectionProcessor(
            engine=engine,
            notifiers=[telegram_notifier],
            roi=roi_config
        )
        
        result, annotated = await processor.process_frame(frame)
    """
    
    # Visualization colors (BGR)
    BOX_COLOR = (0, 255, 0)  # Green
    TEXT_COLOR = (255, 255, 255)  # White
    TEXT_BG_COLOR = (0, 0, 0)  # Black
    ROI_COLOR = (255, 0, 255)  # Magenta
    
    def __init__(
        self,
        engine: InferenceEngine,
        notifiers: Optional[List[Notifier]] = None,
        roi: Optional[ROIConfig] = None,
        draw_boxes: bool = True,
        draw_fps: bool = True
    ):
        """
        Initialize detection processor.
        
        Args:
            engine: Inference engine to use.
            notifiers: List of notification dispatchers.
            roi: Region of Interest configuration.
            draw_boxes: Draw detection boxes on frame.
            draw_fps: Draw FPS counter on frame.
        """
        self.engine = engine
        self.notifiers = notifiers or []
        self.roi = roi
        self.draw_boxes = draw_boxes
        self.draw_fps = draw_fps
        
        self._loop = None
    
    def process_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[InferenceResult, np.ndarray]:
        """
        Process a single frame through the detection pipeline.
        
        Args:
            frame: Input BGR image.
        
        Returns:
            Tuple of (InferenceResult, annotated_frame).
        """
        # Apply ROI if configured
        inference_frame = self._apply_roi(frame)
        
        # Run inference
        result = self.engine.predict(inference_frame)
        
        # Adjust bounding boxes if ROI was applied
        if self.roi and self.roi.enabled:
            result = self._adjust_detections_for_roi(result)
        
        # Draw annotations
        annotated = self._annotate_frame(frame, result)
        
        return result, annotated
    
    async def process_frame_async(
        self,
        frame: np.ndarray
    ) -> Tuple[InferenceResult, np.ndarray]:
        """
        Process frame and dispatch notifications asynchronously.
        
        Args:
            frame: Input BGR image.
        
        Returns:
            Tuple of (InferenceResult, annotated_frame).
        """
        result, annotated = self.process_frame(frame)
        
        # Send notifications if detections found
        if result.detections:
            await self._dispatch_notifications(result, frame)
        
        return result, annotated
    
    def _apply_roi(self, frame: np.ndarray) -> np.ndarray:
        """Extract ROI from frame if configured."""
        if not self.roi or not self.roi.enabled:
            return frame
        
        x, y = self.roi.x, self.roi.y
        w, h = self.roi.width, self.roi.height
        
        # Clamp to frame bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        return frame[y:y+h, x:x+w]
    
    def _adjust_detections_for_roi(
        self,
        result: InferenceResult
    ) -> InferenceResult:
        """Adjust detection coordinates to original frame."""
        if not self.roi or not self.roi.enabled:
            return result
        
        adjusted_detections = []
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            adjusted = Detection(
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                bbox=(
                    x1 + self.roi.x,
                    y1 + self.roi.y,
                    x2 + self.roi.x,
                    y2 + self.roi.y
                )
            )
            adjusted_detections.append(adjusted)
        
        return InferenceResult(
            detections=adjusted_detections,
            preprocessing_time_ms=result.preprocessing_time_ms,
            inference_time_ms=result.inference_time_ms,
            postprocessing_time_ms=result.postprocessing_time_ms
        )
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        result: InferenceResult
    ) -> np.ndarray:
        """Draw annotations on frame."""
        annotated = frame.copy()
        
        # Draw ROI rectangle if configured
        if self.roi and self.roi.enabled:
            cv2.rectangle(
                annotated,
                (self.roi.x, self.roi.y),
                (self.roi.x + self.roi.width, self.roi.y + self.roi.height),
                self.ROI_COLOR,
                2
            )
        
        # Draw detection boxes
        if self.draw_boxes:
            for det in result.detections:
                self._draw_detection(annotated, det)
        
        # Draw FPS counter
        if self.draw_fps:
            fps_text = f"FPS: {result.fps:.1f} | Inference: {result.inference_time_ms:.1f}ms"
            cv2.putText(
                annotated,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.TEXT_COLOR,
                2
            )
        
        return annotated
    
    def _draw_detection(
        self,
        frame: np.ndarray,
        detection: Detection
    ) -> None:
        """Draw a single detection box with label."""
        x1, y1, x2, y2 = detection.bbox
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.BOX_COLOR, 2)
        
        # Draw label background
        label = f"{detection.class_name}: {detection.confidence:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            frame,
            (x1, y1 - label_h - baseline - 5),
            (x1 + label_w, y1),
            self.TEXT_BG_COLOR,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.TEXT_COLOR,
            1
        )
    
    async def _dispatch_notifications(
        self,
        result: InferenceResult,
        frame: np.ndarray
    ) -> None:
        """Send notifications to all configured notifiers."""
        # Build payload
        cat_detections = [d for d in result.detections if d.class_name == "cat"]
        if not cat_detections:
            return
        
        best_conf = max(d.confidence for d in cat_detections)
        payload = NotificationPayload(
            message=f"Cat detected on table! ({len(cat_detections)} detection(s), confidence: {best_conf:.1%})",
            image=frame,
            detection_count=len(cat_detections),
            confidence=best_conf,
            metadata={
                "inference_ms": result.inference_time_ms,
                "engine": self.engine.model_name
            }
        )
        
        # Dispatch to all notifiers concurrently
        tasks = [
            notifier.send_alert(payload)
            for notifier in self.notifiers
            if notifier.enabled and notifier.can_send
        ]
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
