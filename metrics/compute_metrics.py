"""
Compute-cost metrics for the video segmentation pipeline.

Framework-agnostic model analysis through ONNX. All models (PyTorch,
TensorFlow, already-exported ONNX, ...) are first converted to ONNX, then
profiled or benchmarked on the ONNX graph. This guarantees that the
numbers are directly comparable across frameworks, using a single
well-defined convention and a single runtime.

Two metrics are provided:

1. `compute_model_gflops_per_frame` — intrinsic model complexity, in
   GFLOPs per forward pass. Hardware-agnostic. Useful for ranking models
   by theoretical cost and for comparing against the literature.

2. `benchmark_model_latency_cpu` — practical inference latency on CPU,
   reported as percentiles (p50 / p95 / p99) over many calls. The p95 is
   the primary selection metric for real-time applications: it answers
   "how long does one frame take, in the worst reasonable case?". A model
   with low p95 is one that will feel smooth to the end user.

Dependencies
------------
- onnx
- onnx_tool     (FLOPs counting only)
- onnxruntime   (latency benchmarking only)
- torch         (optional, only for `export_pytorch_to_onnx`)
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnx_tool


# =============================================================================
# ONNX export helper (PyTorch only)
# =============================================================================

def export_pytorch_to_onnx(
    model,
    example_inputs,
    onnx_path: str,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset_version: int = 17,
) -> str:
    """
    Export a PyTorch model to ONNX.

    For stateless (frame-by-frame) models, `example_inputs` is typically a
    single tensor of shape (1, 3, H, W). For stateful / recurrent models,
    pass a tuple including both the frame tensor and the state tensor(s),
    so that the ONNX graph captures the full per-frame computation
    (including the state update), as required for a steady-state
    measurement of both FLOPs and latency.

    Note: for non-PyTorch models, perform the equivalent export with your
    framework's tooling (e.g. `tf2onnx` for TensorFlow) and pass the
    resulting `.onnx` file directly to the metric functions below.

    Args:
        model:           PyTorch `nn.Module`, in `eval()` mode.
        example_inputs:  Tensor or tuple of tensors traced through the model.
        onnx_path:       Destination path for the `.onnx` file.
        input_names:     Human-readable names for the model inputs.
        output_names:    Human-readable names for the model outputs.
        dynamic_axes:    Optional mapping, e.g. {"input": {0: "batch"}},
                         to mark dimensions as dynamic.
        opset_version:   ONNX opset version. 17 is a safe modern default.

    Returns:
        The `onnx_path` argument, for chaining.
    """
    import torch  # local import: keep torch optional at module level

    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            example_inputs,
            onnx_path,
            input_names=list(input_names) if input_names else None,
            output_names=list(output_names) if output_names else None,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )
    # Sanity check: a malformed ONNX file would break downstream profiling.
    onnx.checker.check_model(onnx_path)
    return onnx_path


# =============================================================================
# Metric 1: FLOPs per frame (onnx_tool)
# =============================================================================

# Shape specification for a single ONNX input:
#   - a tuple/list of ints  -> fully concrete shape
#   - a numpy array         -> concrete shape inferred from the array
ShapeSpec = Union[Sequence[int], np.ndarray]


def count_flops_onnx(
    onnx_path: str,
    input_shapes: Optional[Dict[str, ShapeSpec]] = None,
) -> int:
    """
    Count the total FLOPs of one forward pass of an ONNX model.

    Internally runs shape inference (required before profiling) and then
    profiles the graph via `onnx_tool`, which reports MACs per node.
    Returns 2 * sum(MACs) following the standard vision convention
    (MobileNet, EfficientNet, etc.). Be aware that some libraries
    (e.g. `thop`) call their output "FLOPs" while actually returning MACs;
    numbers therefore differ by a factor of 2 between sources.

    Args:
        onnx_path:    Path to the `.onnx` file.
        input_shapes: Optional mapping {input_name: shape} used to resolve
                      any dynamic axes in the model. Required if the model
                      was exported with dynamic axes; otherwise the model's
                      static shapes are used.

    Returns:
        Total FLOPs for one forward pass.
    """
    model = onnx_tool.Model(onnx_path)
    # Shape inference is a prerequisite for profiling: the FLOPs per node
    # depend on the tensor shapes that flow through it. Any input with a
    # dynamic axis must be resolved here.
    model.graph.shape_infer(input_shapes)
    # Compute per-node MACs. After this call each node in `nodemap` has a
    # `macs` attribute populated.
    model.graph.profile()
    total_macs = sum(node.macs for node in model.graph.nodemap.values())
    return int(2 * total_macs)


def compute_model_gflops_per_frame(
    onnx_path: str,
    frame_shape: Tuple[int, int, int, int] = (1, 3, 720, 1280),
    frame_input_name: str = "input",
    extra_input_shapes: Optional[Dict[str, ShapeSpec]] = None,
) -> float:
    """
    GFLOPs required per frame by a segmentation model, in steady state.

    For stateless models (the common case), simply pass the expected frame
    tensor shape — typically (1, 3, H, W). For recurrent/stateful models,
    the ONNX export should expose the state tensors as additional inputs;
    pass their shapes via `extra_input_shapes` so that the profiled graph
    reflects the per-frame cost *including* the state update — this is
    the steady-state cost, which is what we want to report.

    The FLOPs count is an intrinsic property of the model graph and does
    not depend on the execution hardware (CPU vs GPU vs accelerator).
    Hardware-dependent cost must be measured separately via latency.

    Args:
        onnx_path:          Path to the ONNX model.
        frame_shape:        Expected shape of the frame input tensor,
                            default (1, 3, 720, 1280) ~ 720p RGB.
        frame_input_name:   Name of the frame input in the ONNX graph.
                            Must match what was used at export time.
        extra_input_shapes: Shapes of any additional inputs (recurrent
                            state tensors, etc.), keyed by ONNX input name.

    Returns:
        GFLOPs (1e9 FLOPs) per frame.
    """
    input_shapes: Dict[str, ShapeSpec] = {frame_input_name: frame_shape}
    if extra_input_shapes:
        input_shapes.update(extra_input_shapes)

    flops = count_flops_onnx(onnx_path, input_shapes=input_shapes)
    return flops / 1e9


# =============================================================================
# Metric 2: CPU inference latency (onnxruntime)
# =============================================================================

# Mapping from ONNX TensorProto element types to numpy dtypes. Only the
# types actually encountered in vision models are listed — extend as
# needed. The element-type IDs are defined in onnx.TensorProto.
_ONNX_DTYPE_TO_NUMPY: Dict[int, np.dtype] = {
    1:  np.dtype(np.float32),  # FLOAT
    2:  np.dtype(np.uint8),    # UINT8
    3:  np.dtype(np.int8),     # INT8
    6:  np.dtype(np.int32),    # INT32
    7:  np.dtype(np.int64),    # INT64
    9:  np.dtype(bool),        # BOOL
    10: np.dtype(np.float16),  # FLOAT16
    11: np.dtype(np.float64),  # DOUBLE
}


def _get_input_dtypes(onnx_path: str) -> Dict[str, np.dtype]:
    """
    Read the expected dtype for each input of an ONNX model.

    Returns a mapping {input_name: numpy.dtype}. Unknown element types
    fall back to float32, which covers the vast majority of real models.
    """
    model = onnx.load(onnx_path)
    dtypes: Dict[str, np.dtype] = {}
    for inp in model.graph.input:
        elem_type = inp.type.tensor_type.elem_type
        dtypes[inp.name] = _ONNX_DTYPE_TO_NUMPY.get(elem_type, np.dtype(np.float32))
    return dtypes


def benchmark_model_latency_cpu(
    onnx_path: str,
    frame_shape: Tuple[int, int, int, int] = (1, 3, 720, 1280),
    frame_input_name: str = "input",
    extra_input_shapes: Optional[Dict[str, Sequence[int]]] = None,
    n_warmup: int = 20,
    n_measure: int = 200,
    percentiles: Sequence[float] = (50, 95, 99),
    intra_op_num_threads: int = 0,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Benchmark a model's single-frame inference latency on CPU.

    Measures the wall-clock time of `session.run(...)` for `n_measure`
    consecutive invocations, after a warm-up phase of `n_warmup` calls.
    Reports latency statistics in milliseconds, with percentiles as the
    primary metric.

    Why percentiles, not mean? In real-time applications (30 fps video
    conferencing has a ~33 ms per-frame budget) the user-perceived quality
    is determined by the worst reasonable case, not the average: a model
    averaging 20 ms but spiking to 80 ms every tenth frame produces visible
    stutter. The p95 answers "how long does a frame take, in the worst
    5% of cases?" and is the recommended metric for model selection.

    Methodological notes:
      - Uses `time.perf_counter` for monotonic, high-resolution timing.
      - Batch size is whatever you put in `frame_shape`; keep it at 1 to
        match online / streaming inference. Larger batches measure
        throughput, not latency.
      - The same randomly-drawn input tensors are reused across all
        iterations. This isolates model compute time from input-generation
        overhead, and is safe because input *values* don't affect the
        computation graph (only shapes do).
      - ONNX Runtime's graph optimizations are enabled (ORT_ENABLE_ALL),
        matching a production deployment.

    Args:
        onnx_path:            Path to the ONNX model.
        frame_shape:          Shape of the frame input tensor.
        frame_input_name:     Name of the frame input in the ONNX graph.
        extra_input_shapes:   Shapes of any additional inputs (state
                              tensors for recurrent models).
        n_warmup:             Number of warm-up iterations to discard.
                              Covers JIT, memory allocation, CPU cache
                              effects, and thread pool spin-up.
        n_measure:            Number of measured iterations. 200 is
                              typically enough for a stable p95; use 500+
                              if you care about p99.
        percentiles:          Percentiles to report, in the range (0, 100).
        intra_op_num_threads: Number of CPU threads for intra-op
                              parallelism. 0 (default) lets ONNX Runtime
                              use all physical cores — representative of
                              a realistic deployment. Set to 1 for
                              deterministic single-threaded benchmarks.
        seed:                 RNG seed for the dummy input tensors.

    Returns:
        A dict with keys `mean`, `std`, `min`, `max`, `n_samples`, and one
        `p{N}` key per requested percentile. All latencies are in
        milliseconds.
    """
    # Deferred import: onnxruntime is only required for this function.
    import onnxruntime as ort

    # --- 1. Prepare dummy inputs ---------------------------------------------
    input_shapes: Dict[str, Sequence[int]] = {frame_input_name: frame_shape}
    if extra_input_shapes:
        input_shapes.update(extra_input_shapes)

    dtypes = _get_input_dtypes(onnx_path)
    rng = np.random.default_rng(seed)
    inputs: Dict[str, np.ndarray] = {}
    for name, shape in input_shapes.items():
        dtype = dtypes.get(name, np.dtype(np.float32))
        if np.issubdtype(dtype, np.floating):
            inputs[name] = rng.standard_normal(shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            # Keep integer inputs in a small positive range — good enough
            # for shape-driven kernels (embeddings, gather ops, ...).
            inputs[name] = rng.integers(0, 128, size=shape, dtype=dtype)
        else:
            inputs[name] = np.zeros(shape, dtype=dtype)

    # --- 2. Create the CPU inference session ---------------------------------
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = intra_op_num_threads
    session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    # --- 3. Warm-up (discarded) ----------------------------------------------
    # The first few calls are systematically slower: kernel selection,
    # memory allocation, and cold CPU caches. Including them would heavily
    # bias the percentiles, so they are measured-and-discarded.
    for _ in range(n_warmup):
        session.run(None, inputs)

    # --- 4. Timed measurements -----------------------------------------------
    timings_ms = np.empty(n_measure, dtype=np.float64)
    for i in range(n_measure):
        t0 = time.perf_counter()
        session.run(None, inputs)
        t1 = time.perf_counter()
        timings_ms[i] = (t1 - t0) * 1000.0

    # --- 5. Aggregate statistics ---------------------------------------------
    stats: Dict[str, float] = {
        "mean": float(timings_ms.mean()),
        "std": float(timings_ms.std()),
        "min": float(timings_ms.min()),
        "max": float(timings_ms.max()),
        "n_samples": int(n_measure),
    }
    for p in percentiles:
        stats[f"p{int(p)}"] = float(np.percentile(timings_ms, p))
    return stats


# =============================================================================
# Example usage
# =============================================================================
#
# Stateless PyTorch model (frame-by-frame U-Net, DeepLab, ...):
# ----------------------------------------------------------------
#   import torch
#   dummy = torch.randn(1, 3, 720, 1280)
#   export_pytorch_to_onnx(
#       model, dummy, "model.onnx",
#       input_names=["input"], output_names=["mask"],
#   )
#   gflops  = compute_model_gflops_per_frame("model.onnx")
#   latency = benchmark_model_latency_cpu("model.onnx")
#   print(f"{gflops:.2f} GFLOPs/frame, p95 = {latency['p95']:.1f} ms")
#
# Recurrent PyTorch model (ConvLSTM-style, propagation from previous frame):
# ---------------------------------------------------------------------------
# The model's forward must accept (frame, state) and return (mask, new_state).
#
#   dummy_frame = torch.randn(1, 3, 720, 1280)
#   dummy_state = torch.randn(1, 64, 45, 80)   # any plausible warm state
#   export_pytorch_to_onnx(
#       model, (dummy_frame, dummy_state), "rnn.onnx",
#       input_names=["input", "state_in"],
#       output_names=["mask", "state_out"],
#   )
#   extras  = {"state_in": (1, 64, 45, 80)}
#   gflops  = compute_model_gflops_per_frame("rnn.onnx",
#                                            extra_input_shapes=extras)
#   latency = benchmark_model_latency_cpu("rnn.onnx",
#                                         extra_input_shapes=extras)
#
# Pre-exported ONNX model (e.g. converted from TensorFlow with tf2onnx):
# ----------------------------------------------------------------------
#   gflops  = compute_model_gflops_per_frame("third_party_model.onnx")
#   latency = benchmark_model_latency_cpu("third_party_model.onnx")