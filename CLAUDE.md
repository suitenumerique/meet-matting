# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Benchmarking framework for video background segmentation/matting models. Evaluates 9 models on quality metrics (IoU, Boundary F-measure, Flow Warping Error) and performance metrics (p95 latency, FLOPs).

## Commands

```bash
# Install dependencies
pip install -r benchmark/requirements.txt

# Run full benchmark (CLI)
python -m benchmark.run_benchmark

# Run specific models
python -m benchmark.run_benchmark --models rvm pphumanseg_v2 mediapipe_portrait

# List available models
python -m benchmark.run_benchmark --list-models

# With output options
python -m benchmark.run_benchmark --models rvm --num-videos 5 --shuffle --save-masks --save-video --save-segmented

# Interactive dashboard
streamlit run benchmark/dashboard.py
```

## Architecture

### Pipeline (per model × video)

1. **Discovery** (`runner.py:discover_datasets`) — pairs `dataset/videos/*.mp4` with `dataset/ground_truth/` (PNG folder or chromakey video)
2. **Inference** (`runner.py:run_inference`) — `VideoPrefetcher` (background thread) streams frames → `model.predict_batch(8)` → collects masks + per-frame latencies (skips first `WARMUP_FRAMES=5`)
3. **Evaluation** (`runner.py:run_evaluation`) — calls `metrics.py:compute_all_metrics()` with parallelism:
   - IoU: single cumulative pass
   - Boundary F-measure: `ThreadPoolExecutor(8)` per frame (OpenCV releases GIL)
   - Flow Warping Error: `ProcessPoolExecutor` (Farneback optical flow, CPU-bound)
4. **Report** — writes `output/benchmark_results.csv` and `.json`

### Model abstraction (`benchmark/models/base.py`)

All models implement `BaseModelWrapper`:
- `load()` — init model, auto-download weights to `benchmark/weights/`
- `predict(frame_bgr) -> np.ndarray` — single frame, returns float32 mask [0,1]
- `predict_batch(frames) -> List` — default loops over `predict()`, override for true batching
- `get_flops(input_shape) -> float`
- `reset_state()` — called per video for recurrent models (RVM uses GRU hidden state)
- `cleanup()` — release GPU/ONNX session

New models: add to `benchmark/models/__init__.py:MODEL_REGISTRY`.

### Registered models

| Key | Model | Framework |
|-----|-------|-----------|
| `mediapipe_portrait` | MediaPipe Portrait Segmenter | TFLite |
| `mediapipe_selfie_multiclass` | MediaPipe Selfie Multiclass | TFLite |
| `mediapipe_landscape` | MediaPipe Landscape Segmenter | TFLite |
| `rvm` | Robust Video Matting (recurrent GRU) | ONNX + PyTorch |
| `mobilenetv3_lraspp` | MobileNetV3 + LRASPP | ONNX |
| `trimap_matting` | Trimap-based (GrabCut fallback) | OpenCV |
| `modnet` | MODNet | PyTorch/ONNX |
| `pphumanseg_v2` | PP-HumanSeg V2 | ONNX |
| `efficient_vit` | EfficientViT | PyTorch |

### Key config (`benchmark/config.py`)

```python
DEFAULT_INPUT_SIZE = (256, 256)
WARMUP_FRAMES = 5
LATENCY_PERCENTILE = 95
MASK_THRESHOLD = 0.5
BOUNDARY_TOLERANCE_PX = 3
```

### Ground truth format

`_load_ground_truth_masks()` supports two formats:
- **PNG folder** — binary masks (0=background, 255=foreground)
- **Video** — chromakey extraction via HSV-based green/black detection + morphological cleanup

### Output artifacts

- `benchmark/output/benchmark_results.csv` — main results table (one row per model×video)
- `benchmark/output/benchmark_results.json` — richer version with per-frame latencies
- `benchmark/output/masks/` — optional per-frame PNG predictions (`--save-masks`)
- `benchmark/temp_results/` — intermediate masks, cleaned after run
