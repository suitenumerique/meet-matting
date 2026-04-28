from datetime import datetime

import streamlit as st
from config import OUTPUT_DIR
from core.pipeline import MattingPipeline
from core.registry import models, postprocessors, preprocessors, upsamplers
from core.video_io import frame_count, read_frame
from core.video_processing import process_video
from ui.sidebar import render_sidebar
from ui.video_panel import display_four_panels

# Auto-discover all plugins.
preprocessors.discover("preprocessing")
models.discover("models")
postprocessors.discover("postprocessing")
upsamplers.discover("upsampling")

st.set_page_config(layout="wide", page_title="Matting Pipeline Lab")
st.title("Background matting pipeline")
st.caption(
    "Configure the pipeline in the sidebar (model, pre/postprocessors), "
    "then inspect results frame-by-frame or export the full video."
)

selection = render_sidebar()

if selection.video_path is None:
    st.info("Drop a video into `data/videos/` to begin.")
    st.stop()

# Build pipeline from selection.
pre_components = [preprocessors.get(name)(**params) for name, params in selection.preprocessors]
model = models.get(selection.model_name)(**selection.model_params)
model.upsampler = upsamplers.get(selection.upsampler[0])(**selection.upsampler[1])
model.load(selection.weights_path)  # may be None
post_components = [postprocessors.get(name)(**params) for name, params in selection.postprocessors]
pipeline = MattingPipeline(pre_components, model, post_components)

# ── Frame preview ──────────────────────────────────────────────────────────────
st.subheader("Frame preview")
st.caption(
    "Drag the slider to inspect any frame. "
    "The four panels show each stage of the pipeline: "
    "original → after preprocessing → raw model mask → final composite. "
    "Note: the final composite is always blended onto the **original** frame, "
    "so preprocessing changes are visible in the mask panel but not the composite."
)
total = frame_count(selection.video_path)
idx = st.slider("Frame", 0, max(total - 1, 0), 0)
frame = read_frame(selection.video_path, idx)
result = pipeline.process_frame(frame)
display_four_panels(result)

st.divider()

# ── Export full video ──────────────────────────────────────────────────────────
st.subheader("Export full video")
st.caption(
    "Runs the full pipeline on every frame and saves two videos to `data/output/`: "
    "**mask.mp4** (alpha matte) and **composite.mp4** (subject on black background). "
    "Each run is saved in its own timestamped folder."
)

col_btn, col_info = st.columns([1, 3])
with col_btn:
    run_clicked = st.button("Process & save", type="primary", use_container_width=True)
with col_info:
    st.caption(
        f"Will process **{total} frames** from `{selection.video_path.name}` "
        f"with model `{selection.model_name}`."
    )

if run_clicked:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"{selection.video_path.stem}__{selection.model_name}__{timestamp}"

    progress_bar = st.progress(0.0)
    status = st.empty()

    def _on_progress(done: int, total_frames: int) -> None:
        progress_bar.progress(done / max(total_frames, 1))
        status.caption(f"Frame {done} / {total_frames}")

    with st.spinner("Processing video…"):
        paths = process_video(pipeline, selection.video_path, run_dir, _on_progress)

    progress_bar.progress(1.0)
    status.empty()
    st.success(f"Saved to `{run_dir.relative_to(OUTPUT_DIR.parent)}`")

    col_m, col_c = st.columns(2)
    with col_m:
        st.caption("Mask — alpha matte (white = foreground)")
        st.video(paths["mask"].read_bytes())
    with col_c:
        st.caption("Composite — subject on black background")
        st.video(paths["composite"].read_bytes())

st.divider()

# ── Browse saved outputs ───────────────────────────────────────────────────────
with st.expander("Browse saved outputs", expanded=False):
    st.caption("Replay any previously exported run. Runs are listed newest-first.")
    if not OUTPUT_DIR.exists() or not any(OUTPUT_DIR.iterdir()):
        st.info("No saved runs yet. Process a video above to create one.")
    else:
        runs = sorted(
            (p for p in OUTPUT_DIR.iterdir() if p.is_dir()),
            reverse=True,
        )
        run_names = [r.name for r in runs]
        selected_run = st.selectbox("Run", run_names, key="browse_run")
        run_dir = OUTPUT_DIR / selected_run

        col_m, col_c = st.columns(2)
        with col_m:
            mask_file = run_dir / "mask.mp4"
            if mask_file.exists():
                st.caption("Mask — alpha matte (white = foreground)")
                st.video(mask_file.read_bytes())
            else:
                st.info("mask.mp4 not found.")
        with col_c:
            comp_file = run_dir / "composite.mp4"
            if comp_file.exists():
                st.caption("Composite — subject on black background")
                st.video(comp_file.read_bytes())
            else:
                st.info("composite.mp4 not found.")
