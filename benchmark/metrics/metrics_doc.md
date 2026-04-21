# Evaluation metrics documentation

Video segmentation pipeline for person isolation (background blur for video conferencing).

The metrics fall into two families:

- **Compute-cost metrics** — how expensive the model is to run (FLOPs/frame, CPU p95 latency). Independent of segmentation quality.
- **Performance metrics** — how accurate and stable the segmentation is (IoU, Boundary F-measure, Flow Warping Error). Independent of compute cost.

The two families are orthogonal: a good production model has to score well on both.

---

## 1. Compute-cost metrics

### 1.1 FLOPs per frame — `compute_model_gflops_per_frame` - Smaller = Better

**What it measures.** The number of floating-point operations required for one forward pass of the model, for an input frame of a given size (default: 720p RGB). This is an **intrinsic** property of the computation graph: it depends neither on the hardware (CPU, GPU, NPU), nor on the runtime, nor on the kernel implementations.

**What it's for.** Ranking models by theoretical complexity and comparing against numbers reported in the literature. Useful to quickly discard models that are too large before even benchmarking them, and to check whether a model fits the compute budget of a target device.

**How it's computed.** The model is first converted to ONNX to guarantee a measurement that is comparable across frameworks (PyTorch, TensorFlow, etc.). The `onnx_tool` library performs shape inference and then profiles the graph node by node, counting MACs (Multiply-Accumulate operations). We return `2 × sum(MACs)`, following the standard vision convention used by MobileNet, EfficientNet, and others. Careful: some libraries (e.g. `thop`) call "FLOPs" what is actually MACs, hence the factor-of-2 discrepancies between sources.

**Limitations.** FLOPs do not predict real latency: two models with the same FLOP count can have very different latencies depending on parallelizability, memory access patterns, and the kernels available on the target platform. This metric should therefore always be paired with a latency measurement.

**Reference.** There is no single foundational paper for this metric — it's a reporting convention that has emerged over the years in the vision literature.

---

### 1.2 CPU p95 latency — `benchmark_model_latency_cpu` - Smaller = Better

**What it measures.** The actual inference time of a frame on CPU, reported as percentiles (p50, p95, p99) over several hundred invocations. The **p95 is the primary metric** for model selection: it answers the question "how long does a frame take, in the worst reasonable case?"

**What it's for.** For a real-time application like video conferencing (30 fps ≈ 33 ms budget per frame), it's not the average that determines perceived smoothness, but the worst case. A model averaging 20 ms but spiking to 80 ms every tenth frame will produce visible stutter. The p95 filters out extreme outliers while still capturing realistic worst cases — it's therefore the indicator to optimize first for a smooth user experience.

**How it's computed.** The model is exported to ONNX and executed via ONNX Runtime (CPUExecutionProvider, all graph optimizations enabled). The measurement follows standard benchmarking methodology:

- **Warmup** of 20 iterations to absorb kernel selection, memory allocation, CPU cache warm-up, and thread pool spin-up.
- **Measurement** of 200 iterations with `time.perf_counter` (monotonic, high-resolution timer).
- The same input tensors are reused across calls to isolate compute cost from input-generation cost (input values don't affect the graph, only shapes do).
- **Batch size = 1** to simulate online streaming, consistent with the video conferencing use case. Increasing the batch would measure throughput, not latency.

**Limitations.** Latency depends heavily on the CPU used: it must be benchmarked on a machine representative of the target deployment. The number of threads (`intra_op_num_threads`) should likewise reflect real conditions; in single-thread mode (=1), numbers are reproducible but pessimistic.

**Reference.** Standard ONNX Runtime benchmarking methodology. No academic paper: this is common practice for measuring inference latency in production.

---

## 2. Performance metrics

### 2.1 Global IoU — `compute_iou` - Higher = Better

**What it measures.** The Intersection-over-Union (also known as Jaccard index) between the predicted mask and the ground truth, aggregated over **all pixels of the video**:

$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|}$$

Score between 0 (no overlap) and 1 (perfect segmentation).

**What it's for.** Measures the **overall region quality**: is the person correctly detected on the whole? Baseline metric, the most intuitive and widely used in image segmentation. For our use case, a high IoU guarantees that we don't blur the person and that we don't leave large background regions sharp.

**Implementation.** Aggregation at the level of **pixels cumulated over the whole video** (not an average of per-frame IoUs). This is more robust: a frame with a near-empty mask can produce a degenerate IoU that would dominate a frame-wise mean. When both masks are entirely empty, we return 1.0 by convention (perfect agreement: no object present).

**Limitations.** IoU is not very sensitive to contour quality: a mask with good global overlap but blurry edges gets a good score, even though the blur rendering will look bad. That's why we always pair IoU with the Boundary F-measure.

**Reference.** Classical metric, Jaccard (1912). So universally used that it has no foundational paper for its computer-vision adoption.

---

### 2.2 Boundary F-measure — `compute_boundary_f_measure` - Higher = Better

**What it measures.** The quality of **contour** alignment between predicted mask and ground truth, computed as the per-frame average of an F-measure (harmonic mean of precision and recall) over boundary pixels, with a spatial tolerance of a few pixels.

For each frame:
1. Extract contours by morphological subtraction: `boundary = mask − erode(mask)`.
2. Dilate the contours with a disk of radius ≈ 0.8% of the image diagonal (DAVIS rule, ≈ 2 px at 480p).
3. **Precision** = fraction of predicted contour pixels falling within the dilated GT contour.
4. **Recall** = fraction of GT contour pixels falling within the dilated predicted contour.
5. **F = 2·P·R/(P+R)**, then averaged across frames.

**What it's for.** This is **the** critical metric for our use case. In background blur, the human eye is immediately drawn to contour defects — chopped hair, missing ears, chunks of background that remain sharp around the silhouette. IoU can be very good while F-measure is mediocre, and it's the F-measure that will predict the perceived quality. Complementary to IoU: IoU = region, F-measure = contour.

**Implementation.** Internal holes in the mask are counted as boundaries (and therefore penalized), which is DAVIS-standard behavior and what we want: a hole in the mask translates to a visible artifact in the blur. The matching between predicted and GT contours is a morphological approximation of the bipartite matching from Martin et al. 2004 — fast and accurate enough in practice.

**Limitations.** The choice of tolerance radius influences the numbers; we stick with the DAVIS default (0.008 × diagonal) to allow comparison with the literature. On very high-resolution images this tolerance can become too strict — worth keeping in mind if you change the evaluation resolution.

**Reference.** Perazzi, F., Pont-Tuset, J., McWilliams, B., Van Gool, L., Gross, M., & Sorkine-Hornung, A. (2016). *A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation*. CVPR 2016. The matching via morphological dilation comes from Martin, D., Fowlkes, C., & Malik, J. (2004), *Learning to Detect Natural Image Boundaries Using Local Brightness, Color, and Texture Cues*, PAMI.

---

### 2.3 Flow Warping Error — `compute_flow_warping_error_farneback` / `compute_flow_warping_error_raft` - Smaller = Better

**What it measures.** The **temporal stability** of the segmentation. Concretely: if we take the mask of frame *t−1* and move it according to optical flow to align it with frame *t*, how closely does it match the predicted mask at *t*? A temporally stable model produces a small error; a model that flickers (the mask changes even though the person doesn't move) produces a large error.

Mean L1 error over valid pixels:
$$E = \frac{1}{\sum_t \sum_p V_t(p)} \sum_t \sum_p V_t(p) \cdot |M_t(p) - \text{warp}(M_{t-1}, F_{t \to t-1})(p)|$$

where *V* is a validity mask (see below) and *F* is the backward flow.

**What it's for.** This is **the** flickering metric. A model can have excellent per-frame IoU and F-measure, yet still be unusable because the mask vibrates from one frame to the next: the background blur becomes "alive" and very distracting. Flow Warping Error detects precisely this kind of defect, which spatial metrics miss entirely.

**Validity mask.** Some pixels cannot be evaluated honestly (disocclusions, optical-flow errors). They are excluded via two criteria:

1. **Forward/backward flow consistency** (Sundaram et al. 2010): we go from *t* to *t−1* via the backward flow, then back via the forward flow; if we don't end up at the starting pixel, the flow is unreliable there (typically a disocclusion).
2. **Photometric consistency** (Lai et al. 2018): warping frame *t−1* should reproduce the observed colors at *t* with small L1 error. Otherwise the flow is unreliable even if it's self-consistent.

**Two variants.**

| Variant | Flow solver | Speed | Accuracy | Usage |
|---|---|---|---|---|
| Farneback | OpenCV, CPU, classical | Fast | Approximate | Fast ranking during iteration |
| RAFT | Deep learning, GPU recommended | Slow | State of the art | Final numbers, publications |

Farneback tends to over-smooth the flow around motion boundaries, which can slightly penalize models with sharp contours. The forward/backward consistency check mitigates this bias, but for final numbers we prefer RAFT.

**Limitations.** The metric depends on the quality of the estimated optical flow. An incorrect flow can either penalize a good model (false error positives) or mask the instability of a bad model. The validity mask mitigates this risk but does not eliminate it. The metric also does not distinguish a "stable but wrong" model from a "stable and correct" one — it must always be read in conjunction with IoU and F-measure.

**References.**

- Lai, W.-S., Huang, J.-B., Wang, O., Shechtman, E., Yumer, E., & Yang, M.-H. (2018). *Learning Blind Video Temporal Consistency*. ECCV 2018. (Definition of warping error and photometric consistency.)
- Sundaram, N., Brox, T., & Keutzer, K. (2010). *Dense Point Trajectories by GPU-accelerated Large Displacement Optical Flow*. ECCV 2010. (Forward/backward flow consistency criterion.)
- Farneback, G. (2003). *Two-Frame Motion Estimation Based on Polynomial Expansion*. SCIA 2003. (OpenCV optical flow algorithm, fast variant.)
- Teed, Z., & Deng, J. (2020). *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow*. ECCV 2020 (Best Paper). (High-accuracy optical flow algorithm, reference variant.)

---

## 3. Reading the metrics together

The five metrics are complementary and should be read jointly. Typical priority order for the video conferencing use case:

1. **p95 latency**: hard filter. If p95 exceeds the budget (≈ 33 ms at 30 fps), the model is out, regardless of its quality.
2. **Boundary F-measure**: decisive for the perceived blur quality. A gap of a few points is visible to the naked eye.
3. **Flow Warping Error**: detects flickering, often the most irritating defect for the user.
4. **IoU**: sanity check on overall region quality; low IoU combined with a good F-measure usually indicates a systemic issue (inverted class, etc.).
5. **GFLOPs/frame**: context metric, useful for portability and for comparing against the literature, but not decisive on its own.