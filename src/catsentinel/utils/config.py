"""
Configuration Management.

Loads settings from YAML config file and environment variables,
providing a unified configuration interface.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Camera/video source configuration."""
    source: int | str = 0
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class ROIConfig:
    """Region of Interest configuration."""
    enabled: bool = False
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0


@dataclass
class DetectionConfig:
    """Detection engine configuration."""
    engine: str = "yolov11"
    model_path: str = "models/yolo11n.pt"
    confidence_threshold: float = 0.5
    target_classes: List[str] = field(default_factory=lambda: ["cat"])
    device: str = "cuda"
    half_precision: bool = True
    roi: ROIConfig = field(default_factory=ROIConfig)


@dataclass
class TelegramConfig:
    """Telegram notification configuration."""
    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""
    send_photo: bool = True


@dataclass
class WebhookConfig:
    """Webhook notification configuration."""
    enabled: bool = False
    url: str = ""
    headers: dict = field(default_factory=dict)


@dataclass
class NotificationConfig:
    """Notification system configuration."""
    enabled: bool = True
    cooldown_seconds: float = 30.0
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    webhook: WebhookConfig = field(default_factory=WebhookConfig)


@dataclass
class BenchmarkConfig:
    """Benchmarking configuration."""
    enabled: bool = True
    log_interval: int = 100
    save_results: bool = True
    results_dir: str = "benchmarks"


@dataclass
class DisplayConfig:
    """Display/visualization configuration."""
    show_preview: bool = True
    draw_boxes: bool = True
    draw_fps: bool = True
    window_name: str = "CatSentinel"


@dataclass
class Config:
    """Main application configuration."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    benchmarking: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)


def load_config(
    config_path: Optional[str] = None,
    env_file: Optional[str] = None
) -> Config:
    """
    Load configuration from YAML file and environment variables.
    
    Priority (highest to lowest):
        1. Environment variables
        2. YAML config file
        3. Default values
    
    Args:
        config_path: Path to config.yaml file.
        env_file: Path to .env file.
    
    Returns:
        Populated Config object.
    """
    config = Config()
    
    # Load .env file if provided
    if env_file:
        _load_env_file(env_file)
    
    # Load YAML config if provided
    if config_path:
        config = _load_yaml_config(config_path, config)
    
    # Override with environment variables
    config = _apply_env_overrides(config)
    
    return config


def _load_env_file(env_file: str) -> None:
    """Load environment variables from .env file."""
    env_path = Path(env_file)
    if not env_path.exists():
        logger.warning(f".env file not found: {env_file}")
        return
    
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())
        logger.info(f"Loaded environment from: {env_file}")
    except Exception as e:
        logger.error(f"Failed to load .env file: {e}")


def _load_yaml_config(config_path: str, config: Config) -> Config:
    """Load configuration from YAML file."""
    yaml_path = Path(config_path)
    if not yaml_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return config
    
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML not installed. Run: pip install pyyaml")
        return config
    
    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        
        if not data:
            return config
        
        # Parse camera config
        if 'camera' in data:
            cam = data['camera']
            config.camera = CameraConfig(
                source=cam.get('source', 0),
                width=cam.get('width', 1280),
                height=cam.get('height', 720),
                fps=cam.get('fps', 30)
            )
        
        # Parse detection config
        if 'detection' in data:
            det = data['detection']
            roi_data = det.get('roi', {})
            config.detection = DetectionConfig(
                engine=det.get('engine', 'yolov11'),
                model_path=det.get('model_path', 'models/yolo11n.pt'),
                confidence_threshold=det.get('confidence_threshold', 0.5),
                target_classes=det.get('target_classes', ['cat']),
                device=det.get('device', 'cuda'),
                half_precision=det.get('half_precision', True),
                roi=ROIConfig(
                    enabled=roi_data.get('enabled', False),
                    x=roi_data.get('x', 0),
                    y=roi_data.get('y', 0),
                    width=roi_data.get('width', 0),
                    height=roi_data.get('height', 0)
                )
            )
        
        # Parse notifications config
        if 'notifications' in data:
            notif = data['notifications']
            tg = notif.get('telegram', {})
            wh = notif.get('webhook', {})
            config.notifications = NotificationConfig(
                enabled=notif.get('enabled', True),
                cooldown_seconds=notif.get('cooldown_seconds', 30.0),
                telegram=TelegramConfig(
                    enabled=tg.get('enabled', False),
                    bot_token=tg.get('bot_token', ''),
                    chat_id=tg.get('chat_id', ''),
                    send_photo=tg.get('send_photo', True)
                ),
                webhook=WebhookConfig(
                    enabled=wh.get('enabled', False),
                    url=wh.get('url', ''),
                    headers=wh.get('headers', {})
                )
            )
        
        # Parse benchmarking config
        if 'benchmarking' in data:
            bench = data['benchmarking']
            config.benchmarking = BenchmarkConfig(
                enabled=bench.get('enabled', True),
                log_interval=bench.get('log_interval', 100),
                save_results=bench.get('save_results', True),
                results_dir=bench.get('results_dir', 'benchmarks')
            )
        
        # Parse display config
        if 'display' in data:
            disp = data['display']
            config.display = DisplayConfig(
                show_preview=disp.get('show_preview', True),
                draw_boxes=disp.get('draw_boxes', True),
                draw_fps=disp.get('draw_fps', True),
                window_name=disp.get('window_name', 'CatSentinel')
            )
        
        logger.info(f"Loaded config from: {config_path}")
        
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
    
    return config


def _apply_env_overrides(config: Config) -> Config:
    """Apply environment variable overrides."""
    
    # Telegram credentials from env (sensitive data)
    if os.getenv('TELEGRAM_BOT_TOKEN'):
        config.notifications.telegram.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    if os.getenv('TELEGRAM_CHAT_ID'):
        config.notifications.telegram.chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    # Webhook secrets from env
    if os.getenv('WEBHOOK_URL'):
        config.notifications.webhook.url = os.getenv('WEBHOOK_URL')
    if os.getenv('WEBHOOK_SECRET'):
        config.notifications.webhook.headers['Authorization'] = (
            f"Bearer {os.getenv('WEBHOOK_SECRET')}"
        )
    
    # Device override
    if os.getenv('CATSENTINEL_DEVICE'):
        config.detection.device = os.getenv('CATSENTINEL_DEVICE')
    
    return config
