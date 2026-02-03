"""
Timing Decorators for Benchmarking.

Provides decorators to measure execution time of functions
with high precision for inference benchmarking.
"""

import functools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TimingStats:
    """Statistics for timed function calls."""
    
    function_name: str
    calls: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    times: List[float] = field(default_factory=list)
    
    @property
    def avg_time_ms(self) -> float:
        """Average execution time."""
        return self.total_time_ms / self.calls if self.calls > 0 else 0.0
    
    @property
    def fps(self) -> float:
        """Estimated FPS based on average time."""
        return 1000.0 / self.avg_time_ms if self.avg_time_ms > 0 else 0.0
    
    def add_timing(self, time_ms: float) -> None:
        """Record a new timing measurement."""
        self.calls += 1
        self.total_time_ms += time_ms
        self.min_time_ms = min(self.min_time_ms, time_ms)
        self.max_time_ms = max(self.max_time_ms, time_ms)
        self.times.append(time_ms)
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.calls = 0
        self.total_time_ms = 0.0
        self.min_time_ms = float('inf')
        self.max_time_ms = 0.0
        self.times.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/export."""
        return {
            "function": self.function_name,
            "calls": self.calls,
            "avg_ms": round(self.avg_time_ms, 3),
            "min_ms": round(self.min_time_ms, 3) if self.calls > 0 else 0,
            "max_ms": round(self.max_time_ms, 3),
            "total_ms": round(self.total_time_ms, 3),
            "fps": round(self.fps, 2)
        }


# Global registry for timing statistics
_timing_registry: Dict[str, TimingStats] = {}


def get_timing_stats(name: Optional[str] = None) -> Dict[str, TimingStats]:
    """
    Get timing statistics.
    
    Args:
        name: Specific function name, or None for all.
    
    Returns:
        Dictionary of TimingStats objects.
    """
    if name:
        return {name: _timing_registry.get(name)}
    return _timing_registry.copy()


def reset_timing_stats(name: Optional[str] = None) -> None:
    """Reset timing statistics."""
    if name and name in _timing_registry:
        _timing_registry[name].reset()
    else:
        for stats in _timing_registry.values():
            stats.reset()


def measure_time(
    name: Optional[str] = None,
    log_level: int = logging.DEBUG,
    log_interval: int = 1
) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        name: Custom name for the metric (default: function name).
        log_level: Logging level for timing output.
        log_interval: Log every N calls.
    
    Example:
        @measure_time(name="preprocessing")
        def preprocess(image):
            ...
    """
    def decorator(func: Callable) -> Callable:
        metric_name = name or func.__name__
        
        if metric_name not in _timing_registry:
            _timing_registry[metric_name] = TimingStats(metric_name)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            stats = _timing_registry[metric_name]
            stats.add_timing(elapsed_ms)
            
            if stats.calls % log_interval == 0:
                logger.log(
                    log_level,
                    f"{metric_name}: {elapsed_ms:.2f}ms "
                    f"(avg: {stats.avg_time_ms:.2f}ms, calls: {stats.calls})"
                )
            
            return result
        
        return wrapper
    return decorator


def measure_inference(func: Callable) -> Callable:
    """
    Specialized decorator for inference timing.
    
    Logs at INFO level every 100 calls for inference methods.
    """
    return measure_time(
        name=f"inference_{func.__name__}",
        log_level=logging.INFO,
        log_interval=100
    )(func)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    
    engine_name: str
    model_name: str
    total_frames: int
    total_time_seconds: float
    avg_inference_ms: float
    min_inference_ms: float
    max_inference_ms: float
    avg_fps: float
    vram_peak_mb: float
    vram_avg_mb: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "engine": self.engine_name,
            "model": self.model_name,
            "frames": self.total_frames,
            "duration_s": round(self.total_time_seconds, 2),
            "avg_inference_ms": round(self.avg_inference_ms, 3),
            "min_inference_ms": round(self.min_inference_ms, 3),
            "max_inference_ms": round(self.max_inference_ms, 3),
            "avg_fps": round(self.avg_fps, 2),
            "vram_peak_mb": round(self.vram_peak_mb, 2),
            "vram_avg_mb": round(self.vram_avg_mb, 2)
        }


class BenchmarkContext:
    """Context manager for benchmarking inference runs."""
    
    def __init__(self, engine_name: str, model_name: str):
        self.engine_name = engine_name
        self.model_name = model_name
        self.inference_times: List[float] = []
        self.vram_samples: List[float] = []
        self.start_time: float = 0.0
    
    def record_inference(self, time_ms: float) -> None:
        """Record an inference timing."""
        self.inference_times.append(time_ms)
    
    def record_vram(self, vram_mb: float) -> None:
        """Record VRAM usage sample."""
        self.vram_samples.append(vram_mb)
    
    def get_result(self) -> BenchmarkResult:
        """Generate benchmark result."""
        elapsed = time.perf_counter() - self.start_time
        
        return BenchmarkResult(
            engine_name=self.engine_name,
            model_name=self.model_name,
            total_frames=len(self.inference_times),
            total_time_seconds=elapsed,
            avg_inference_ms=sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0,
            min_inference_ms=min(self.inference_times) if self.inference_times else 0,
            max_inference_ms=max(self.inference_times) if self.inference_times else 0,
            avg_fps=len(self.inference_times) / elapsed if elapsed > 0 else 0,
            vram_peak_mb=max(self.vram_samples) if self.vram_samples else 0,
            vram_avg_mb=sum(self.vram_samples) / len(self.vram_samples) if self.vram_samples else 0
        )
    
    def __enter__(self) -> "BenchmarkContext":
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args) -> None:
        pass


def benchmark_run(engine_name: str, model_name: str) -> BenchmarkContext:
    """
    Create a benchmark context for an inference run.
    
    Example:
        with benchmark_run("YOLOv11", "yolo11n.pt") as bench:
            for frame in frames:
                result = engine.predict(frame)
                bench.record_inference(result.inference_time_ms)
                bench.record_vram(monitor.get_vram_mb())
            
            report = bench.get_result()
    """
    return BenchmarkContext(engine_name, model_name)
