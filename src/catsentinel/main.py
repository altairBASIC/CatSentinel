"""
CatSentinel - Main Entry Point.

Real-time cat detection system with modular inference engines
and benchmarking capabilities for legacy GPU hardware.

Usage:
    python -m catsentinel.main --config configs/config.yaml
    python -m catsentinel.main --help
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

import cv2

from .benchmarking import VRAMMonitor, benchmark_run
from .capture import VideoSource
from .detection import DetectionProcessor
from .engines import InferenceEngine, YOLOv11Engine
from .notifications import TelegramNotifier, WebhookNotifier, Notifier
from .utils import load_config, Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class CatSentinel:
    """
    Main application class with dependency injection.
    
    Example:
        config = load_config("configs/config.yaml")
        
        async with CatSentinel(config) as sentinel:
            await sentinel.run()
    """
    
    def __init__(self, config: Config):
        """
        Initialize CatSentinel with configuration.
        
        Args:
            config: Application configuration object.
        """
        self.config = config
        self.engine: Optional[InferenceEngine] = None
        self.video_source: Optional[VideoSource] = None
        self.processor: Optional[DetectionProcessor] = None
        self.notifiers: list[Notifier] = []
        self.vram_monitor: Optional[VRAMMonitor] = None
        self._running = False
    
    def _create_engine(self) -> InferenceEngine:
        """
        Factory method to create inference engine based on config.
        
        This is where you add new engine types for future YOLO versions.
        """
        engine_type = self.config.detection.engine.lower()
        
        # Engine registry - add new engines here
        engines = {
            "yolov11": YOLOv11Engine,
            "yolo11": YOLOv11Engine,
            # Future: "yolov26": YOLOv26Engine,
        }
        
        if engine_type not in engines:
            available = ", ".join(engines.keys())
            raise ValueError(
                f"Unknown engine: {engine_type}. Available: {available}"
            )
        
        engine_class = engines[engine_type]
        
        return engine_class(
            model_path=self.config.detection.model_path,
            confidence_threshold=self.config.detection.confidence_threshold,
            target_classes=self.config.detection.target_classes,
            device=self.config.detection.device,
            half_precision=self.config.detection.half_precision
        )
    
    def _create_notifiers(self) -> list[Notifier]:
        """Create notification dispatchers based on config."""
        notifiers = []
        notif_config = self.config.notifications
        
        if not notif_config.enabled:
            return notifiers
        
        # Telegram notifier
        if notif_config.telegram.enabled:
            if notif_config.telegram.bot_token and notif_config.telegram.chat_id:
                notifiers.append(TelegramNotifier(
                    bot_token=notif_config.telegram.bot_token,
                    chat_id=notif_config.telegram.chat_id,
                    enabled=True,
                    cooldown_seconds=notif_config.cooldown_seconds,
                    send_photo=notif_config.telegram.send_photo
                ))
                logger.info("Telegram notifier configured")
            else:
                logger.warning(
                    "Telegram enabled but credentials missing. "
                    "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars."
                )
        
        # Webhook notifier
        if notif_config.webhook.enabled:
            if notif_config.webhook.url:
                notifiers.append(WebhookNotifier(
                    url=notif_config.webhook.url,
                    enabled=True,
                    cooldown_seconds=notif_config.cooldown_seconds,
                    headers=notif_config.webhook.headers
                ))
                logger.info(f"Webhook notifier configured: {notif_config.webhook.url}")
            else:
                logger.warning("Webhook enabled but URL missing.")
        
        return notifiers
    
    async def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing CatSentinel...")
        
        # Create and load inference engine
        self.engine = self._create_engine()
        logger.info(f"Loading engine: {self.engine.model_name}")
        self.engine.load_model()
        
        engine_info = self.engine.get_engine_info()
        logger.info(
            f"Engine ready: {engine_info.engine_name} on {engine_info.device} "
            f"({engine_info.precision})"
        )
        
        # Create video source
        self.video_source = VideoSource(
            source=self.config.camera.source,
            width=self.config.camera.width,
            height=self.config.camera.height,
            fps=self.config.camera.fps
        )
        
        # Create notifiers
        self.notifiers = self._create_notifiers()
        
        # Create detection processor
        self.processor = DetectionProcessor(
            engine=self.engine,
            notifiers=self.notifiers,
            roi=self.config.detection.roi,
            draw_boxes=self.config.display.draw_boxes,
            draw_fps=self.config.display.draw_fps
        )
        
        # Initialize VRAM monitor
        if self.config.benchmarking.enabled:
            self.vram_monitor = VRAMMonitor()
            self.vram_monitor.initialize()
            
            gpu_info = self.vram_monitor.get_gpu_info()
            if gpu_info:
                logger.info(
                    f"GPU: {gpu_info.name} | "
                    f"VRAM: {gpu_info.memory_total_mb:.0f}MB | "
                    f"CUDA: {gpu_info.cuda_version}"
                )
        
        logger.info("Initialization complete")
    
    async def cleanup(self) -> None:
        """Cleanup all resources."""
        logger.info("Cleaning up...")
        
        if self.video_source:
            self.video_source.close()
        
        if self.engine:
            self.engine.unload_model()
        
        if self.vram_monitor:
            self.vram_monitor.shutdown()
        
        # Close notifier sessions
        for notifier in self.notifiers:
            if hasattr(notifier, 'close'):
                await notifier.close()
        
        cv2.destroyAllWindows()
        logger.info("Cleanup complete")
    
    async def run(self) -> None:
        """Main processing loop."""
        if not self.video_source.open():
            logger.error("Failed to open video source")
            return
        
        self._running = True
        logger.info("Starting detection loop. Press 'q' to quit.")
        
        engine_info = self.engine.get_engine_info()
        
        with benchmark_run(engine_info.engine_name, engine_info.model_name) as bench:
            while self._running:
                # Read frame
                frame_result = self.video_source.read()
                if not frame_result.is_valid:
                    logger.warning("Failed to read frame")
                    break
                
                # Process frame
                result, annotated = await self.processor.process_frame_async(
                    frame_result.frame
                )
                
                # Record benchmark data
                if self.config.benchmarking.enabled:
                    bench.record_inference(result.inference_time_ms)
                    if self.vram_monitor:
                        bench.record_vram(self.vram_monitor.get_vram_mb())
                
                # Log periodic benchmarks
                if (self.config.benchmarking.enabled and 
                    self.video_source.frame_count % self.config.benchmarking.log_interval == 0):
                    vram = self.vram_monitor.get_vram_mb() if self.vram_monitor else 0
                    logger.info(
                        f"Frame {self.video_source.frame_count} | "
                        f"Inference: {result.inference_time_ms:.2f}ms | "
                        f"FPS: {result.fps:.1f} | "
                        f"VRAM: {vram:.0f}MB"
                    )
                
                # Display preview
                if self.config.display.show_preview:
                    cv2.imshow(self.config.display.window_name, annotated)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Quit requested by user")
                        break
            
            # Print final benchmark results
            if self.config.benchmarking.enabled:
                benchmark_result = bench.get_result()
                logger.info("=" * 60)
                logger.info("BENCHMARK RESULTS")
                logger.info("=" * 60)
                for key, value in benchmark_result.to_dict().items():
                    logger.info(f"  {key}: {value}")
                logger.info("=" * 60)
    
    def stop(self) -> None:
        """Signal the main loop to stop."""
        self._running = False
    
    async def __aenter__(self) -> "CatSentinel":
        await self.initialize()
        return self
    
    async def __aexit__(self, *args) -> None:
        await self.cleanup()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CatSentinel - Real-time cat detection with benchmarking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "-e", "--env",
        type=str,
        default=".env",
        help="Path to .env file"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Override inference device"
    )
    
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable video preview window"
    )
    
    parser.add_argument(
        "--benchmark-only",
        type=int,
        metavar="FRAMES",
        help="Run benchmark for N frames and exit"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    """Async main entry point."""
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config_path = args.config if Path(args.config).exists() else None
    env_path = args.env if Path(args.env).exists() else None
    
    config = load_config(config_path, env_path)
    
    # Apply command line overrides
    if args.device:
        config.detection.device = args.device
    if args.no_preview:
        config.display.show_preview = False
    
    # Create and run application
    sentinel = CatSentinel(config)
    
    # Handle signals for graceful shutdown
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        sentinel.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        async with sentinel:
            await sentinel.run()
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1


def main() -> int:
    """Main entry point."""
    args = parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
