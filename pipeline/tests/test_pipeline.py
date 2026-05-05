"""Integration tests for MattingPipeline.process_frame — shapes, dtypes, and end-to-end behaviour."""

import numpy as np
from core.pipeline import MattingPipeline
from core.registry import models, postprocessors, preprocessors

# Ensure all components are registered before tests run.
preprocessors.discover("preprocessing")
models.discover("models")
postprocessors.discover("postprocessing")


def _make_frame(h=64, w=64):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _build(model_params=None, post_names=None, post_params_list=None):
    pre = [preprocessors.get("identity")()]
    model_params = model_params or {"mask_value": 1.0}
    mdl = models.get("dummy_passthrough")(**model_params)
    mdl.load(None)
    post_names = post_names or ["identity"]
    post_params_list = post_params_list or [{}]
    post = [postprocessors.get(n)(**p) for n, p in zip(post_names, post_params_list, strict=True)]
    return MattingPipeline(pre, mdl, post)


def test_basic_shapes_and_dtypes():
    """process_frame result dict must contain all expected keys with correct shapes and dtypes."""
    frame = _make_frame()
    pipeline = _build()
    result = pipeline.process_frame(frame)

    assert result["original"].shape == (64, 64, 3)
    assert result["original"].dtype == np.uint8
    assert result["preprocessed"].shape == (64, 64, 3)
    assert result["preprocessed"].dtype == np.uint8
    assert result["raw_mask"].shape == (64, 64)
    assert result["raw_mask"].dtype == np.float32
    assert result["final"].shape == (64, 64, 3)
    assert result["final"].dtype == np.uint8


def test_identity_pre_leaves_frame_equal():
    """With identity preprocessor, original and preprocessed frames must be pixel-identical."""
    frame = _make_frame()
    pipeline = _build()
    result = pipeline.process_frame(frame)
    np.testing.assert_array_equal(result["original"], result["preprocessed"])


def test_mask_value_one_final_equals_original():
    """A mask of all-ones means the composite must equal the original frame."""
    frame = _make_frame()
    pipeline = _build(model_params={"mask_value": 1.0})
    result = pipeline.process_frame(frame)
    np.testing.assert_array_equal(result["final"], frame)


def test_mask_value_zero_final_is_zeros():
    """A mask of all-zeros means the composite must be a black frame (pure background)."""
    frame = _make_frame()
    pipeline = _build(model_params={"mask_value": 0.0})
    result = pipeline.process_frame(frame)
    np.testing.assert_array_equal(result["final"], np.zeros_like(frame))


def test_threshold_above_cutoff_gives_all_ones():
    """Threshold postprocessor must output all-ones when mask_value > cutoff."""
    frame = _make_frame()
    pipeline = _build(
        model_params={"mask_value": 0.6},
        post_names=["threshold"],
        post_params_list=[{"cutoff": 0.5}],
    )
    result = pipeline.process_frame(frame)
    np.testing.assert_array_equal(result["final_mask"], np.ones((64, 64), dtype=np.float32))


def test_threshold_below_cutoff_gives_all_zeros():
    """Threshold postprocessor must output all-zeros when mask_value < cutoff."""
    frame = _make_frame()
    pipeline = _build(
        model_params={"mask_value": 0.4},
        post_names=["threshold"],
        post_params_list=[{"cutoff": 0.5}],
    )
    result = pipeline.process_frame(frame)
    np.testing.assert_array_equal(result["final_mask"], np.zeros((64, 64), dtype=np.float32))
