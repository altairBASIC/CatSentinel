"""
YOLOv11 Inference Engine Implementation.

Uses the Ultralytics library for YOLOv11 inference.
This is the primary engine for benchmarking on legacy hardware (GTX 1080).
"""

import time
from pathlib import Path
from typing import List

import numpy as np

from .base import Detection, EngineInfo, InferenceEngine, InferenceResult


class YOLOv11Engine(InferenceEngine):
    """
    YOLOv11 inference engine using Ultralytics.
    
    This implementation serves as the baseline for comparing
    against newer YOLO versions on legacy hardware.
    
    Example:
        engine = YOLOv11Engine(
            model_path="models/yolo11n.pt",
            confidence_threshold=0.5,
            target_classes=["cat"]
        )
        with engine:
            result = engine.predict(frame)
            print(f"Inference: {result.inference_time_ms:.2f}ms")
    """
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        target_classes: List[str] = None,
        device: str = "cuda",
        half_precision: bool = True
    ):
        """
        Initialize YOLOv11 engine.
        
        Args:
            model_path: Path to YOLO11 weights (.pt file).
            confidence_threshold: Detection confidence threshold.
            target_classes: Classes to detect (default: ["cat"]).
            device: Inference device ('cuda' or 'cpu').
            half_precision: Use FP16 for faster inference on GPU.
        """
        super().__init__(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            target_classes=target_classes,
            device=device
        )
        self.half_precision = half_precision and device == "cuda"
        self._class_names: dict = {}
        self._framework_version: str = ""
    
    @property
    def model_name(self) -> str:
        """Return the model name."""
        return f"YOLOv11 ({Path(self.model_path).stem})"
    
    def load_model(self) -> None:
        """
        Load YOLOv11 model using Ultralytics.
        
        Raises:
            FileNotFoundError: If model file doesn't exist.
            ImportError: If ultralytics is not installed.
        """
        if self._is_loaded:
            return
        
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            from ultralytics import YOLO
            import ultralytics
            self._framework_version = ultralytics.__version__
        except ImportError as e:
            raise ImportError(
                "Ultralytics not installed. Run: pip install ultralytics"
            ) from e
        
        # Load model
        self._model = YOLO(str(model_path))
        
        # Move to device (half precision is handled in predict())
        self._model.to(self.device)
        
        # Cache class names
        self._class_names = self._model.names
        
        # Warmup inference (use half precision if enabled)
        dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model.predict(dummy_input, verbose=False, half=self.half_precision)
        
        self._is_loaded = True
    
    def predict(self, frame: np.ndarray) -> InferenceResult:
        """
        Run YOLOv11 inference on a frame.
        
        Args:
            frame: BGR image as numpy array.
        
        Returns:
            InferenceResult with detections and timing.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocessing timing
        t0 = time.perf_counter()
        # Ultralytics handles preprocessing internally
        preprocessing_time = (time.perf_counter() - t0) * 1000
        
        # Inference timing
        t1 = time.perf_counter()
        results = self._model.predict(
            frame,
            conf=self.confidence_threshold,
            verbose=False,
            half=self.half_precision,
            classes=self._get_target_class_ids()
        )
        inference_time = (time.perf_counter() - t1) * 1000
        
        # Postprocessing timing
        t2 = time.perf_counter()
        detections = self._parse_results(results)
        postprocessing_time = (time.perf_counter() - t2) * 1000
        
        return InferenceResult(
            detections=detections,
            preprocessing_time_ms=preprocessing_time,
            inference_time_ms=inference_time,
            postprocessing_time_ms=postprocessing_time
        )
    
    def _get_target_class_ids(self) -> List[int]:
        """Get class IDs for target classes."""
        if not self.target_classes:
            return None
        
        target_ids = []
        for class_name in self.target_classes:
            for class_id, name in self._class_names.items():
                if name.lower() == class_name.lower():
                    target_ids.append(class_id)
                    break
        
        return target_ids if target_ids else None
    
    def _parse_results(self, results) -> List[Detection]:
        """Parse Ultralytics results into Detection objects."""
        detections = []
        
        for result in results:
            if result.boxes is None:
                continue
            
            boxes = result.boxes
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())
                bbox = tuple(map(int, boxes.xyxy[i].tolist()))
                
                detections.append(Detection(
                    class_id=class_id,
                    class_name=self._class_names.get(class_id, "unknown"),
                    confidence=confidence,
                    bbox=bbox
                ))
        
        return detections
    
    def get_engine_info(self) -> EngineInfo:
        """Return engine metadata for benchmarking."""
        return EngineInfo(
            engine_name="YOLOv11Engine",
            model_name=self.model_name,
            model_path=self.model_path,
            framework="ultralytics",
            version=self._framework_version,
            device=self.device,
            precision="fp16" if self.half_precision else "fp32"
        )
