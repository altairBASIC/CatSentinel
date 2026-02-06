# CatSentinel

Real-time cat detection system designed for benchmarking object detection models on legacy GPU hardware.

## Overview

CatSentinel is a modular computer vision framework that detects cats in video streams while providing detailed performance metrics. Built with extensibility in mind, it uses the Strategy Pattern to allow seamless comparison between different detection models on hardware like the NVIDIA GTX 1080.

### Supported Models

| Engine | Framework | Models | Architecture |
|--------|-----------|--------|-------------|
| YOLOv11 | ultralytics | yolo11n, yolo11s | CNN (baseline) |
| YOLOv26 | ultralytics | yolo26n, yolo26s | CNN, NMS-free |
| RF-DETR | rfdetr | nano, small | Transformer (DINOv2) |

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
├── engines/              # Inference engine implementations (Strategy Pattern)
│   ├── base.py           # Abstract InferenceEngine class
│   ├── yolov11_engine.py # YOLOv11 (ultralytics)
│   ├── yolov26_engine.py # YOLOv26 (ultralytics, NMS-free)
│   └── rfdetr_engine.py  # RF-DETR (transformer-based)
├── benchmarking/         # Performance measurement utilities
│   ├── decorators.py     # Timing decorators
│   └── metrics.py        # VRAM monitoring via pynvml
├── notifications/        # Async alert system
├── capture/              # Video source abstraction
├── detection/            # Detection pipeline orchestrator
├── utils/                # Configuration and logging
└── main.py               # Entry point with dependency injection
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

## Using Different Engines

```python
from catsentinel.engines import YOLOv11Engine, YOLOv26Engine, RFDETREngine

# YOLOv11 (baseline)
engine = YOLOv11Engine(model_path="yolo11n.pt", confidence_threshold=0.5)

# YOLOv26 (NMS-free, edge-optimized)
engine = YOLOv26Engine(model_path="yolo26n.pt", confidence_threshold=0.5)

# RF-DETR (transformer-based)
engine = RFDETREngine(model_size="nano", confidence_threshold=0.5)

# Use any engine with the same interface
with engine:
    result = engine.predict(frame)
    print(f"Detections: {len(result.detections)}, Time: {result.inference_time_ms:.2f}ms")
```

## Benchmarking

> **Status**: _In development_ - benchmark results coming soon!

CatSentinel automatically collects performance metrics:

- **Inference Time**: Pure model inference in milliseconds
- **FPS**: Frames processed per second  
- **VRAM Usage**: GPU memory consumption via pynvml

### Engine Comparison (GTX 1080)

| Engine | Model | Avg Inference | FPS | VRAM Peak |
|--------|-------|---------------|-----|----------|
| YOLOv11 | yolo11n | ~8ms | ~118 | ~1GB |
| YOLOv26 | yolo26n | TBD | TBD | TBD |
| RF-DETR | nano | TBD | TBD | TBD |

> **Note**: RF-DETR uses transformers (DINOv2 backbone) which are more memory-intensive. On 8GB GPUs, use Nano/Small variants.

Metrics are logged periodically and summarized at exit:

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
