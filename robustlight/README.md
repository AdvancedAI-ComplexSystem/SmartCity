# RobustLight: A Robust Traffic Signal Control System

RobustLight is an advanced traffic signal control system that uses deep reinforcement learning with enhanced robustness against sensor noise and failures.

## Requirements

```bash
# Core Dependencies
torch>=1.8.0
numpy>=1.19.2
pandas>=1.2.0
tensorflow>=2.4.0
cityflow  # For traffic simulation
```

## Usage

### Quick Start

Run the main training script:
```bash
python run_advanced_colight_dsi.py
```

### Configuration

Key configuration files:
- `utils/config.py`: Main configuration parameters
- `models/colight_agent_dsi.py`: CoLight DSI agent implementation
- `utils/model_test.py`: Testing utilities

## Results Analysis

Use `summary.py` to analyze experimental results:
```bash
python summary.py
```