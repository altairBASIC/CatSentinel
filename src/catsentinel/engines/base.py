"""
Abstract Base Class for Inference Engines - Strategy Pattern Implementation.

This module defines the contract that all inference engines must follow,
enabling seamless swapping between different YOLO versions (v11, v26, etc.)
without modifying the video capture or notification logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np


@dataclass
class Detection:
    """Represents a single detection result."""
    
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    
    def __repr__(self) -> str:
        return (
            f"Detection(class={self.class_name}, "
            f"conf={self.confidence:.2f}, bbox={self.bbox})"
        )


@dataclass
class InferenceResult:
    """Container for inference results with timing metadata."""
    
    detections: List[Detection]
    preprocessing_time_ms: float
    inference_time_ms: float
    postprocessing_time_ms: float
    
    @property
    def total_time_ms(self) -> float:
        """Total processing time in milliseconds."""
        return (
            self.preprocessing_time_ms + 
            self.inference_time_ms + 
            self.postprocessing_time_ms
        )
    
    @property
    def fps(self) -> float:
        """Estimated FPS based on total processing time."""
        if self.total_time_ms > 0:
            return 1000.0 / self.total_time_ms
        return 0.0


@dataclass
class EngineInfo:
    """Metadata about the inference engine for benchmarking."""
    
    engine_name: str
    model_name: str
    model_path: str
    framework: str
    version: str
    device: str
    precision: str  # fp32, fp16, int8


class InferenceEngine(ABC):
    """
    Abstract Base Class for all inference engines.
    
    Implements the Strategy Pattern to allow runtime swapping of
    different YOLO versions while maintaining a consistent interface.
    
    Usage:
        engine = YOLOv11Engine(model_path="yolo11n.pt")
        engine.load_model()
        result = engine.predict(frame)
    
    To add a new engine (e.g., YOLOv26):
        1. Create yolov26_engine.py
        2. Inherit from InferenceEngine
        3. Implement load_model(), predict(), and get_engine_info()
        4. Register in the engine factory
    """
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        target_classes: Optional[List[str]] = None,
        device: str = "cuda"
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the model weights file.
            confidence_threshold: Minimum confidence for detections.
            target_classes: List of class names to detect (None = all).
            device: Device to run inference on ('cuda', 'cpu').
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes or ["cat"]
        self.device = device
        self._model: Any = None
        self._is_loaded: bool = False
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self._is_loaded
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the human-readable model name."""
        pass
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Load the model weights into memory.
        
        This method should:
            1. Load the model from self.model_path
            2. Move the model to the specified device
            3. Perform any warmup inference if needed
            4. Set self._is_loaded = True
        
        Raises:
            FileNotFoundError: If model file doesn't exist.
            RuntimeError: If model loading fails.
        """
        pass
    
    @abstractmethod
    def predict(self, frame: np.ndarray) -> InferenceResult:
        """
        Run inference on a single frame.
        
        Args:
            frame: Input image as numpy array (BGR format, HWC).
        
        Returns:
            InferenceResult containing detections and timing info.
        
        Raises:
            RuntimeError: If model is not loaded.
        """
        pass
    
    @abstractmethod
    def get_engine_info(self) -> EngineInfo:
        """
        Return metadata about this engine for benchmarking.
        
        Returns:
            EngineInfo dataclass with engine details.
        """
        pass
    
    def unload_model(self) -> None:
        """
        Unload the model from memory.
        
        Override this method if special cleanup is needed.
        """
        self._model = None
        self._is_loaded = False
    
    def __enter__(self) -> "InferenceEngine":
        """Context manager entry - loads the model."""
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - unloads the model."""
        self.unload_model()
