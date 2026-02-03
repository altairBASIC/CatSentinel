"""
GPU Metrics Collection for Benchmarking.

Uses pynvml (NVIDIA Management Library) to collect VRAM
and GPU utilization metrics for performance comparison.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """GPU device information."""
    
    name: str
    driver_version: str
    cuda_version: str
    memory_total_mb: float
    compute_capability: str


@dataclass
class VRAMSnapshot:
    """VRAM usage snapshot."""
    
    used_mb: float
    free_mb: float
    total_mb: float
    utilization_percent: float


class VRAMMonitor:
    """
    Monitor VRAM usage using NVIDIA Management Library.
    
    Example:
        monitor = VRAMMonitor()
        monitor.initialize()
        
        vram = monitor.get_vram_usage()
        print(f"VRAM: {vram.used_mb:.0f}MB / {vram.total_mb:.0f}MB")
        
        monitor.shutdown()
    """
    
    def __init__(self, device_index: int = 0):
        """
        Initialize VRAM monitor.
        
        Args:
            device_index: GPU device index (default: 0).
        """
        self.device_index = device_index
        self._initialized = False
        self._handle = None
        self._pynvml = None
    
    def initialize(self) -> bool:
        """
        Initialize NVML library.
        
        Returns:
            True if initialization successful, False otherwise.
        """
        if self._initialized:
            return True
        
        try:
            import pynvml
            self._pynvml = pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._initialized = True
            logger.info("NVML initialized successfully")
            return True
        except ImportError:
            logger.warning(
                "pynvml not installed. VRAM monitoring disabled. "
                "Install with: pip install pynvml"
            )
            return False
        except Exception as e:
            logger.warning(f"Failed to initialize NVML: {e}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown NVML library."""
        if self._initialized and self._pynvml:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
            self._initialized = False
    
    def get_vram_usage(self) -> Optional[VRAMSnapshot]:
        """
        Get current VRAM usage.
        
        Returns:
            VRAMSnapshot or None if monitoring unavailable.
        """
        if not self._initialized:
            return None
        
        try:
            mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            
            return VRAMSnapshot(
                used_mb=mem_info.used / (1024 ** 2),
                free_mb=mem_info.free / (1024 ** 2),
                total_mb=mem_info.total / (1024 ** 2),
                utilization_percent=util.memory
            )
        except Exception as e:
            logger.warning(f"Failed to get VRAM usage: {e}")
            return None
    
    def get_vram_mb(self) -> float:
        """Get current VRAM usage in MB (convenience method)."""
        snapshot = self.get_vram_usage()
        return snapshot.used_mb if snapshot else 0.0
    
    def get_gpu_info(self) -> Optional[GPUInfo]:
        """
        Get GPU device information.
        
        Returns:
            GPUInfo or None if unavailable.
        """
        if not self._initialized:
            return None
        
        try:
            name = self._pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            
            driver = self._pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver, bytes):
                driver = driver.decode('utf-8')
            
            cuda_version = self._pynvml.nvmlSystemGetCudaDriverVersion_v2()
            cuda_major = cuda_version // 1000
            cuda_minor = (cuda_version % 1000) // 10
            
            mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            
            major, minor = self._pynvml.nvmlDeviceGetCudaComputeCapability(
                self._handle
            )
            
            return GPUInfo(
                name=name,
                driver_version=driver,
                cuda_version=f"{cuda_major}.{cuda_minor}",
                memory_total_mb=mem_info.total / (1024 ** 2),
                compute_capability=f"{major}.{minor}"
            )
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
            return None
    
    def __enter__(self) -> "VRAMMonitor":
        self.initialize()
        return self
    
    def __exit__(self, *args) -> None:
        self.shutdown()


def get_gpu_info(device_index: int = 0) -> Optional[GPUInfo]:
    """
    Convenience function to get GPU info.
    
    Args:
        device_index: GPU device index.
    
    Returns:
        GPUInfo or None if unavailable.
    """
    with VRAMMonitor(device_index) as monitor:
        return monitor.get_gpu_info()
