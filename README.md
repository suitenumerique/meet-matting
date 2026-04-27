# Background Segmentation — Meet Matting

> **Warning: This project is in a very early experimental stage.**
> Breaking changes can be pushed at any time. Expect instability and incomplete features. Use at your own risk.

A research project focused on real-time background segmentation for video calls (visioconference). The work is organized into three main phases:

1. **Benchmark** — systematic evaluation of existing segmentation models on video data
2. **Pre & Post-processing** — development of methods to improve inference, mask quality around edges and stabilize temporal consistency
3. **Web implementation** — integration of the best approach into a browser-based interface

---

## Project Structure

```
meet-matting/
├── benchmark/              # Phase 1: model evaluation
│   ├── models/             # One file per model wrapper
│   ├── dataset/            # Input videos + ground-truth masks
│   ├── output/             # CSV/JSON results and saved masks
│   ├── weights/            # Pre-downloaded model weights
│   ├── dashboard.py        # Streamlit benchmark UI
│   ├── runner.py           # Core inference + metric loop
│   ├── metrics.py          # IoU, Boundary F-measure, FWE
│   └── config.py           # Paths and global parameters
└── pyproject.toml
```

---

## Benchmark

Phase 1 evaluates ten real-time matting models on a 25-clip dataset, measuring both quality (IoU, Boundary F-measure, Flow Warping Error) and compute cost (P95 latency, FLOPs per frame).

See [`benchmark/README.md`](benchmark/README.md) for the full list of models, metric definitions and interpretation, and dataset construction.

```bash
streamlit run benchmark/dashboard.py           # interactive UI
python -m benchmark.run_benchmark --help       # CLI
```

---

## Installation

Requires Python 3.11+. Dependencies are managed with [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

To run without uv:

```bash
pip install -e .
```