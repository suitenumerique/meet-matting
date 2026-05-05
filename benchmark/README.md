# Benchmark — Video Matting Evaluation

Systematic evaluation of real-time video matting models for video conferencing. Measures both **quality** (mask accuracy, contour precision, temporal stability) and **compute cost** (latency, FLOPs).

```bash
streamlit run benchmark/dashboard.py           # interactive UI
python -m benchmark.run_benchmark --help       # CLI
```

---

## Models

Nine model wrappers, all sharing the [`BaseModelWrapper`](models/base.py) interface (`load`, `predict`, `reset_state`, `cleanup`).

| Key | Model | Backend | Input size | Notes |
|---|---|---|---|---|
| `mediapipe_portrait` | MediaPipe Selfie Segmenter | TFLite | 256×256 | Portrait orientation |
| `mediapipe_landscape` | MediaPipe Selfie Landscape | TFLite | 256×256 | Landscape orientation |
| `mediapipe_selfie_multiclass` | MediaPipe Multiclass | TFLite | 256×256 | 5 classes (person, hair, skin…) |
| `rvm` | Robust Video Matting | ONNX | Dynamic | **Recurrent** (GRU state) — best temporal coherence |
| `mobilenetv3_lraspp` | MobileNetV3 LR-ASPP | PyTorch | Dynamic | Lightweight pyramid decoder |
| `modnet` | MODNet | ONNX | 512×512 | High-quality portrait alpha matting |
| `pphumanseg_v2` | PP-HumanSeg V2 | ONNX | Dynamic | Lightweight PaddlePaddle model |
| `efficient_vit` | EfficientViT-Seg | ONNX | 224×224 | Vision Transformer (ADE20K, person class #12) |
| `trimap_matting` | Trimap GrabCut | OpenCV | Dynamic | Morphological baseline |

To add a model: subclass `BaseModelWrapper`, implement `predict(frame_bgr) → mask float32 [0, 1]`, and register it in [`models/__init__.py`](models/__init__.py).

---

## Metrics

Five metrics are reported per (model, video) pair: three quality metrics and two compute metrics.

### 1. IoU — Intersection over Union

**What it measures.** Global pixel-level agreement between predicted and ground-truth masks across the entire video.

**How it's computed.** Both masks are binarized at threshold 0.5. Intersection and union are accumulated over **all pixels of all frames**, then a single ratio is computed:

```
IoU = Σ (pred ∧ gt)  /  Σ (pred ∨ gt)
```

This pixel-weighted aggregation (rather than a per-frame mean) is more robust to frames where the subject is small or absent.

**Interpretation.** Range `[0, 1]`. Higher is better.
- `> 0.95` — excellent
- `0.90 – 0.95` — good, suitable for most use cases
- `< 0.85` — visible silhouette errors

### 2. Boundary F-measure

**What it measures.** Precision of mask **contours** (the edges of the silhouette — hair, fingers, accessories). IoU is dominated by interior pixels and barely penalizes edge errors; this metric isolates them.

**How it's computed.** Per-frame DAVIS protocol (Perazzi et al. 2016):
1. Extract a 1-pixel boundary by morphological erosion: `boundary = mask − erode(mask)`.
2. Compute a tolerance disk of radius `0.008 × diagonal` (≈ 5 px at 1080p).
3. Dilate both boundaries by that disk, then compute precision and recall against the dilated counterpart.
4. Return `F1 = 2·P·R / (P + R)`.

Computed at 540p max for speed (4× fewer pixels than 1080p without changing the relative tolerance), parallelized across 8 threads. The reported value is the mean over all frames.

**Interpretation.** Range `[0, 1]`. Higher is better.
- `> 0.85` — clean edges
- `0.70 – 0.85` — acceptable, some edge artifacts
- `< 0.60` — visible halo or jagged contours

### 3. Flow Warping Error (FWE)

**What it measures.** Temporal stability — how much the mask "flickers" between consecutive frames, beyond what the actual motion explains. A model can have great IoU per frame and still be unusable if the mask shimmers.

**How it's computed.** Lai et al. 2018, using DIS optical flow:
1. For each consecutive pair `(t-1, t)`: compute forward and backward optical flow on the RGB frames.
2. Build a validity mask: forward/backward consistency check + photometric error threshold (filters out occlusions and disocclusions).
3. Warp the previous predicted mask by the backward flow → expected mask at time `t`.
4. Accumulate L1 error on valid pixels: `error = |pred_t − warp(pred_{t-1})|`.

Computed at 320p max with `frame_step=2` (every other frame pair), cropped to a bounding box around the mask union for further speedup.

**Interpretation.** Range `[0, 1]`. **Lower is better** — this is an error.
- `< 0.005` — very stable
- `0.005 – 0.02` — minor flicker
- `> 0.05` — visible flicker, distracting in real conditions

Recurrent models (RVM) typically score much better here than per-frame models (MediaPipe, MobileNetV3).

### 4. P95 Latency

**What it measures.** End-to-end inference time per frame, in milliseconds. P95 (rather than mean) captures real-world responsiveness — a model with low average but occasional spikes will drop frames.

**How it's computed.** Wall-clock time (`time.perf_counter()`) wraps each `model.predict(frame)` call. The first 5 frames are discarded (warm-up — JIT compilation, GPU kernel caching). The 95th percentile is taken over the remaining frames. Frame decoding runs asynchronously in a separate thread (`VideoPrefetcher`) and is **not** counted.

**Interpretation.** Lower is better. For 30 fps real-time:
- `< 33 ms` — real-time on a single stream
- `33 – 66 ms` — usable but drops frames under load
- `> 100 ms` — not real-time

Reported alongside `latency_mean_ms` and `latency_std_ms` for diagnostics.

### 5. FLOPs per Frame

**What it measures.** Theoretical compute cost — floating-point operations per inferred frame. Hardware-independent complexity indicator, useful when comparing latency across machines.

**How it's computed.** Hardcoded estimates from published papers and official model cards, at each model's effective input resolution. Returned in GFLOPs.

**Interpretation.** Lower is better, but FLOPs do not directly equal latency: memory bandwidth, kernel launch overhead, and operator support on the target hardware (CoreML, CUDA, CPU) all matter. Use FLOPs to explain **why** a model is slow, not to predict **how** slow.

---

## Evaluation Dataset

The benchmark is run on **25 video clips** designed to cover the realistic spectrum of video-call conditions:

- **15 clips** sampled from a public Kaggle person-segmentation challenge — controlled studio footage with varied subjects.
- **10 clips** recorded by us to cover edge cases that public datasets typically miss:
  - outdoor scenes with natural lighting variations,
  - multiple people simultaneously in frame,
  - hand-held / moving camera,
  - varied backgrounds (cluttered indoor, urban, low contrast),
  - subjects entering and leaving the frame.

### Ground-truth masks

Per-frame ground-truth masks are generated with **[Mat-Anything v2](https://github.com/hustvl/Matte-Anything)**, a strong open-source matting model based on SAM. Mat-Anything v2 produces high-quality alpha mattes that resolve fine structures (hair, edges of clothing) far beyond what manual annotation could achieve at this scale. Masks are reviewed manually and corrected when needed.

This makes Mat-Anything v2 effectively the **upper bound** for the benchmark — no benchmarked model can outperform the reference itself. The goal is to compare lightweight real-time models against each other, using the matting model's output as a near-ideal target.

### Dataset layout

```
benchmark/dataset/
├── videos/          # 25 source clips, zero-padded numbering
│   ├── 0000.mp4
│   ├── 0001.mp4
│   └── ...
└── ground_truth/    # one mask video per source clip, same numbering
    ├── 0000.mp4
    ├── 0001.mp4
    └── ...
```

Each ground-truth file is a single-channel `.mp4` where each frame is the binary mask for the corresponding source frame. The runner pairs files by name (`0000 ↔ 0000`, etc.).

The runner also supports per-clip folders of PNG masks (one file per frame, zero-padded) and green-screen videos (chroma key auto-detected and inverted).
