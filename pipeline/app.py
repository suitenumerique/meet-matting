import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from config import OUTPUT_DIR
from core.pipeline import MattingPipeline
from core.registry import models, postprocessors, preprocessors, skip_strategies, upsamplers
from core.video_io import frame_count, read_frame
from core.video_processing import process_video
from ui.sidebar import render_sidebar
from ui.synced_player import display_synced_player
from ui.video_panel import display_four_panels

# Auto-discover all plugins.
preprocessors.discover("preprocessing")
models.discover("models")
postprocessors.discover("postprocessing")
upsamplers.discover("upsampling")
skip_strategies.discover("skip_strategies")

st.set_page_config(layout="wide", page_title="Matting Pipeline Lab")
st.title("Background matting pipeline")

# ── Config Import ──────────────────────────────────────────────────────────────
with st.sidebar.expander("Import Config", expanded=False):
    uploaded_config = st.file_uploader("Upload config.json", type="json")

    # Only apply if it's a NEW file we haven't processed yet
    should_apply = False
    if uploaded_config is not None:
        file_id = f"{uploaded_config.name}_{uploaded_config.size}"
        if st.session_state.get("last_config_id") != file_id:
            should_apply = True
            st.session_state["last_config_id"] = file_id

    if should_apply:
        try:
            config_to_apply = json.load(uploaded_config)

            # Basic keys
            if "model_name" in config_to_apply:
                st.session_state["model_select"] = config_to_apply["model_name"]
            if "weights_path" in config_to_apply:
                st.session_state["weights_path"] = config_to_apply["weights_path"]

            # Video matching
            if "video_source" in config_to_apply:
                from config import VIDEO_DIR
                from core.video_io import list_videos

                source_name = Path(config_to_apply["video_source"]).name
                for v in list_videos(VIDEO_DIR):
                    if v.name == source_name:
                        st.session_state["video_select"] = v
                        break

            # Model params
            if "model_params" in config_to_apply:
                for k, v in config_to_apply["model_params"].items():
                    st.session_state[f"model_{k}"] = v

            # Preprocessors
            if "preprocessors" in config_to_apply:
                names = [p[0] for p in config_to_apply["preprocessors"]]
                st.session_state["pre_select"] = names
                for i, (_name, params) in enumerate(config_to_apply["preprocessors"]):
                    for pk, pv in params.items():
                        st.session_state[f"pre_{i}_{pk}"] = pv

            # Postprocessors
            if "postprocessors" in config_to_apply:
                names = [p[0] for p in config_to_apply["postprocessors"]]
                st.session_state["post_select"] = names
                for i, (_name, params) in enumerate(config_to_apply["postprocessors"]):
                    for pk, pv in params.items():
                        st.session_state[f"post_{i}_{pk}"] = pv

            st.success("Configuration chargée ! Vous pouvez maintenant la modifier librement.")
            # Trigger a rerun to make sure all widgets see the new session_state
            st.rerun()
        except Exception as e:
            st.error(f"Erreur lors de l'import : {e}")

st.caption(
    "Configure the pipeline in the sidebar (model, pre/postprocessors), "
    "then inspect results frame-by-frame or export the full video."
)

selection = render_sidebar()

if selection.video_path is None:
    st.info("Drop a video into `data/videos/` to begin.")
    st.stop()

# Cache model + upsampler in session_state (expensive to reload on every interaction).
# Pre/post components are rebuilt every run so parameter changes take effect immediately.
_model_key = hashlib.md5(
    json.dumps(
        {
            "model": selection.model_name,
            "model_params": selection.model_params,
            "weights": selection.weights_path,
            "upsampler": selection.upsampler,
        },
        sort_keys=True,
        default=str,
    ).encode()
).hexdigest()
if st.session_state.get("_model_key") != _model_key:
    _m = models.get(selection.model_name)(**selection.model_params)
    _m.upsampler = upsamplers.get(selection.upsampler[0])(**selection.upsampler[1])
    _m.load(selection.weights_path)
    st.session_state["_model"] = _m
    st.session_state["_model_key"] = _model_key

model = st.session_state["_model"]
pre_components = [preprocessors.get(name)(**params) for name, params in selection.preprocessors]
post_components = [postprocessors.get(name)(**params) for name, params in selection.postprocessors]
pipeline = MattingPipeline(pre_components, model, post_components, bg_color=selection.bg_color)

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
col_slider, col_fps = st.columns([4, 1])
with col_slider:
    idx = st.slider("Frame", 0, max(total - 1, 0), 0)

# Warm up stateful filters (EMA, 1-Euro, Kalman, median) by replaying the
# preceding WARMUP frames silently before displaying frame idx.
# Without this, filters always show cold-start behaviour (= raw mask unchanged).
_WARMUP = 10
for _w in range(max(0, idx - _WARMUP), idx):
    pipeline.process_frame(read_frame(selection.video_path, _w))

frame = read_frame(selection.video_path, idx)
t0 = time.time()
result = pipeline.process_frame(frame)
t1 = time.time()
fps = 1.0 / max(t1 - t0, 0.001)

with col_fps:
    st.metric("Inference FPS", f"{fps:.1f}")

display_four_panels(result)

st.divider()

# ── Export full video ──────────────────────────────────────────────────────────
st.subheader("Export full video")
skip_frames = selection.skip_frames
st.caption(
    "Runs the full pipeline on every frame and saves results to `data/output/`. "
    "Configure skip frames and strategy in the sidebar."
)

col_btn, col_info = st.columns([1, 3])
with col_btn:
    run_clicked = st.button("Process & save", type="primary", use_container_width=True)
with col_info:
    frames_to_process = (total + skip_frames - 1) // skip_frames
    strategy_name = selection.skip_strategy[0]
    st.caption(
        f"Will process **{frames_to_process} frames** (1 every {skip_frames}, "
        f"strategy: `{strategy_name}`) with model `{selection.model_name}`."
    )

if run_clicked:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"{selection.video_path.stem}__{selection.model_name}__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config_data = {
        "timestamp": timestamp,
        "video_source": str(selection.video_path),
        "model_name": selection.model_name,
        "model_params": selection.model_params,
        "preprocessors": selection.preprocessors,
        "postprocessors": selection.postprocessors,
        "skip_frames": skip_frames,
        "skip_strategy": selection.skip_strategy,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_data, f, indent=4)

    pipeline.reset()

    strategy_name, strategy_params = selection.skip_strategy
    strategy_instance = skip_strategies.get(strategy_name)(**strategy_params)

    progress_bar = st.progress(0.0)
    status = st.empty()

    def _on_progress(done: int, total_frames: int, current_fps: float) -> None:
        progress_bar.progress(done / max(total_frames, 1))
        status.caption(f"Frame {done} / {total_frames} | {current_fps:.1f} FPS")

    with st.spinner("Processing video..."):
        paths = process_video(
            pipeline,
            selection.video_path,
            run_dir,
            _on_progress,
            skip_frames=skip_frames,
            skip_strategy=strategy_instance,
        )

    progress_bar.progress(1.0)
    status.empty()
    st.success(f"Saved to `{run_dir.relative_to(OUTPUT_DIR.parent)}`")
    display_synced_player(paths)

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
        if not run_names:
            st.info("No saved runs found.")
        else:
            selected_run = st.selectbox("Run", run_names, key="browse_run")
            if selected_run:
                run_dir = OUTPUT_DIR / selected_run

                # Show config for this run
                config_file = run_dir / "config.json"
                if config_file.exists():
                    with open(config_file) as f:
                        conf = json.load(f)
                    st.json(conf)

                display_synced_player(
                    {
                        "original": run_dir / "original.mp4",
                        "mask": run_dir / "mask.mp4",
                        "raw": run_dir / "raw.mp4",
                        "composite": run_dir / "composite.mp4",
                    }
                )
