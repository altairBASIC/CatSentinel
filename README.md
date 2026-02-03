# CatSentinel

Real-time cat detection system designed for benchmarking YOLO inference performance on legacy GPU hardware.

## Overview

CatSentinel is a modular computer vision framework that detects cats in video streams while providing detailed performance metrics. Built with extensibility in mind, it uses the Strategy Pattern to allow seamless comparison between different YOLO versions (v11, v26, and future releases) on legacy hardware like the NVIDIA GTX 1080.

## Features

- **Modular Inference Engines**: Swap between YOLO versions without modifying application logic
- **Real-time Detection**: Process video streams from cameras, files, or RTSP sources
- **Async Notifications**: Non-blocking alerts via Telegram and webhooks
- **Comprehensive Benchmarking**: Measure inference latency, FPS, and VRAM usage
- **ROI Support**: Focus detection on specific regions of interest
- **Configurable**: YAML configuration with environment variable overrides

## Architecture

```
catsentinel/
├── engines/           # Inference engine implementations (Strategy Pattern)
│   ├── base.py        # Abstract InferenceEngine class
│   └── yolov11_engine.py
├── benchmarking/      # Performance measurement utilities
│   ├── decorators.py  # Timing decorators
│   └── metrics.py     # VRAM monitoring via pynvml
├── notifications/     # Async alert system
│   ├── telegram.py    # Telegram bot integration
│   └── webhook.py     # Generic webhook sender
├── capture/           # Video source abstraction
├── detection/         # Detection pipeline orchestrator
├── utils/             # Configuration and logging
└── main.py            # Entry point with dependency injection
```

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- Ubuntu (WSL) or Linux

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/catsentinel.git
cd catsentinel

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/WSL

# Install dependencies
pip install -e .

# Download YOLOv11 model
wget -P models/ https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
```

## Configuration

Copy the example configuration files:

```bash
cp .env.example .env
# Edit .env with your Telegram credentials (optional)
```

Edit `configs/config.yaml` to customize:

- Camera source and resolution
- Detection confidence threshold
- Notification settings
- Benchmarking parameters

## Usage

### Basic Usage

```bash
# Run with default configuration
python -m catsentinel.main

# Specify custom config
python -m catsentinel.main --config configs/config.yaml

# Run on CPU
python -m catsentinel.main --device cpu

# Headless mode (no preview)
python -m catsentinel.main --no-preview
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `-c, --config` | Path to YAML configuration file |
| `-e, --env` | Path to .env file |
| `--device` | Override inference device (cuda/cpu) |
| `--no-preview` | Disable video preview window |
| `-v, --verbose` | Enable debug logging |

### Keyboard Controls

- `q` - Quit application

## Adding New Engines

To add support for a new YOLO version (e.g., YOLOv26):

1. Create `src/catsentinel/engines/yolov26_engine.py`
2. Inherit from `InferenceEngine` base class
3. Implement required methods: `load_model()`, `predict()`, `get_engine_info()`
4. Register in the engine factory (`main.py`)

```python
from .base import InferenceEngine, InferenceResult

class YOLOv26Engine(InferenceEngine):
    @property
    def model_name(self) -> str:
        return "YOLOv26"
    
    def load_model(self) -> None:
        # Load model implementation
        pass
    
    def predict(self, frame) -> InferenceResult:
        # Inference implementation
        pass
    
    def get_engine_info(self) -> EngineInfo:
        # Return engine metadata
        pass
```

## Benchmarking

CatSentinel automatically collects performance metrics:

- **Inference Time**: Pure model inference in milliseconds
- **FPS**: Frames processed per second
- **VRAM Usage**: GPU memory consumption via pynvml

Metrics are logged periodically (configurable interval) and summarized at exit:

```
============================================================
BENCHMARK RESULTS
============================================================
  engine: YOLOv11Engine
  model: yolo11n.pt
  frames: 1000
  avg_inference_ms: 8.45
  avg_fps: 118.34
  vram_peak_mb: 1024.0
============================================================
```

## Project Structure

```
Yolo-CatSentinel/
├── src/
│   └── catsentinel/       # Main package
├── tests/                 # Test suite
├── configs/
│   └── config.yaml        # Application configuration
├── models/                # YOLO model weights
├── .env.example           # Environment template
├── requirements.txt       # Dependencies
├── pyproject.toml         # Project metadata
└── README.md
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request
