"""
RF-DETR Inference Engine Implementation.

Uses the Roboflow RF-DETR library for transformer-based object detection.
This engine allows benchmarking DETR-style models against YOLO on legacy hardware.

Note: RF-DETR uses a DINOv2 vision transformer backbone, which is more
memory-intensive than convolutional models. Recommended to use Nano/Small
variants on GPUs with limited VRAM (e.g., GTX 1080 with 8GB).
"""

import time
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
from PIL import Image

from .base import Detection, EngineInfo, InferenceEngine, InferenceResult


# Model size type for type hints
ModelSize = Literal["nano", "small"]


class RFDETREngine(InferenceEngine):
    """
    RF-DETR inference engine using the rfdetr library.
    
    RF-DETR is a real-time DETR variant that achieves SOTA results
    on COCO while maintaining competitive inference speeds.
    
    Example:
        engine = RFDETREngine(
            model_size="nano",
            confidence_threshold=0.5,
            target_classes=["cat"]
        )
        with engine:
            result = engine.predict(frame)
            print(f"Inference: {result.inference_time_ms:.2f}ms")
    
    Supported model sizes (optimized for legacy hardware):
        - nano: Fastest, lowest accuracy (~4.5M params)
        - small: Good balance (~12M params)
    """
    
    # COCO class names mapping (subset - cat is class 15)
    COCO_CLASSES = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
        5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
        10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
        14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
        20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
        25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
        30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
        34: "baseball bat", 35: "baseball glove", 36: "skateboard",
        37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
        41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
        46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
        51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
        56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
        60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
        65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
        69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
        74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
        78: "hair drier", 79: "toothbrush"
    }
    
    def __init__(
        self,
        model_size: ModelSize = "nano",
        confidence_threshold: float = 0.5,
        target_classes: Optional[List[str]] = None,
        device: str = "cuda"
    ):
        """
        Initialize RF-DETR engine.
        
        Args:
            model_size: Size of RF-DETR model ("nano" or "small").
            confidence_threshold: Detection confidence threshold.
            target_classes: Classes to detect (default: ["cat"]).
            device: Inference device ('cuda' or 'cpu').
        """
        # RF-DETR doesn't use a model path file like YOLO
        super().__init__(
            model_path=f"rfdetr-{model_size}",
            confidence_threshold=confidence_threshold,
            target_classes=target_classes,
            device=device
        )
        self.model_size = model_size
        self._framework_version: str = ""
    
    @property
    def model_name(self) -> str:
        """Return the model name."""
        return f"RF-DETR ({self.model_size})"
    
    def load_model(self) -> None:
        """
        Load RF-DETR model.
        
        Raises:
            ImportError: If rfdetr is not installed.
            ValueError: If invalid model size.
        """
        if self._is_loaded:
            return
        
        try:
            import rfdetr
            self._framework_version = getattr(rfdetr, "__version__", "unknown")
            
            # Import the appropriate model class
            if self.model_size == "nano":
                from rfdetr import RFDETRNano
                self._model = RFDETRNano()
            elif self.model_size == "small":
                from rfdetr import RFDETRSmall
                self._model = RFDETRSmall()
            else:
                raise ValueError(
                    f"Invalid model size: {self.model_size}. "
                    f"Supported: nano, small"
                )
        except ImportError as e:
            raise ImportError(
                "rfdetr not installed. Run: pip install rfdetr"
            ) from e
        
        # Warmup inference
        dummy_input = Image.fromarray(
            np.zeros((640, 640, 3), dtype=np.uint8)
        )
        self._model.predict(dummy_input, threshold=self.confidence_threshold)
        
        self._is_loaded = True
    
    def predict(self, frame: np.ndarray) -> InferenceResult:
        """
        Run RF-DETR inference on a frame.
        
        Args:
            frame: BGR image as numpy array (OpenCV format).
        
        Returns:
            InferenceResult with detections and timing.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocessing: BGR numpy -> RGB PIL
        t0 = time.perf_counter()
        rgb_frame = frame[:, :, ::-1]  # BGR to RGB
        pil_image = Image.fromarray(rgb_frame)
        preprocessing_time = (time.perf_counter() - t0) * 1000
        
        # Inference
        t1 = time.perf_counter()
        sv_detections = self._model.predict(
            pil_image,
            threshold=self.confidence_threshold
        )
        inference_time = (time.perf_counter() - t1) * 1000
        
        # Postprocessing: supervision.Detections -> our Detection objects
        t2 = time.perf_counter()
        detections = self._parse_detections(sv_detections)
        postprocessing_time = (time.perf_counter() - t2) * 1000
        
        return InferenceResult(
            detections=detections,
            preprocessing_time_ms=preprocessing_time,
            inference_time_ms=inference_time,
            postprocessing_time_ms=postprocessing_time
        )
    
    def _parse_detections(self, sv_detections) -> List[Detection]:
        """
        Parse supervision.Detections into our Detection objects.
        
        Args:
            sv_detections: supervision.Detections object from RF-DETR.
        
        Returns:
            List of Detection objects.
        """
        detections = []
        
        # Get target class IDs
        target_ids = self._get_target_class_ids()
        
        # supervision.Detections has: xyxy, confidence, class_id
        if sv_detections is None or len(sv_detections) == 0:
            return detections
        
        for i in range(len(sv_detections)):
            class_id = int(sv_detections.class_id[i])
            
            # Filter by target classes if specified
            if target_ids and class_id not in target_ids:
                continue
            
            confidence = float(sv_detections.confidence[i])
            bbox = tuple(map(int, sv_detections.xyxy[i].tolist()))
            class_name = self.COCO_CLASSES.get(class_id, "unknown")
            
            detections.append(Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox=bbox
            ))
        
        return detections
    
    def _get_target_class_ids(self) -> Optional[List[int]]:
        """Get class IDs for target classes."""
        if not self.target_classes:
            return None
        
        # Build reverse mapping
        name_to_id = {name.lower(): id for id, name in self.COCO_CLASSES.items()}
        
        target_ids = []
        for class_name in self.target_classes:
            class_id = name_to_id.get(class_name.lower())
            if class_id is not None:
                target_ids.append(class_id)
        
        return target_ids if target_ids else None
    
    def get_engine_info(self) -> EngineInfo:
        """Return engine metadata for benchmarking."""
        return EngineInfo(
            engine_name="RFDETREngine",
            model_name=self.model_name,
            model_path=self.model_path,
            framework="rfdetr",
            version=self._framework_version,
            device=self.device,
            precision="fp32"  # RF-DETR handles precision internally
        )
