import os
os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

import time
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
import numpy as np
from config import OUTPUT_DIR
from core.pipeline import MattingPipeline
from core.registry import models, postprocessors, preprocessors, skip_strategies, upsamplers
from core.video_io import frame_count, read_frame
from core.video_processing import process_video
from ui.sidebar import render_sidebar
from ui.synced_player import display_synced_player
from ui.video_panel import display_four_panels

import urllib.request
import tempfile
import cv2

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

# ── TABS ───────────────────────────────────────────────────────────────────────
tab_lab, tab_live = st.tabs(["📽️ Laboratoire Vidéo", "📸 Test Caméra"])

with tab_lab:
    # ── Frame preview ──────────────────────────────────────────────────────────
    st.subheader("Frame preview")
    st.caption(
        "Drag the slider to inspect any frame. "
        "The four panels show each stage of the pipeline: "
        "original → after preprocessing → raw model mask → final composite. "
    )
    total = frame_count(selection.video_path)
    col_slider, col_fps = st.columns([4, 1])
    with col_slider:
        idx = st.slider("Frame", 0, max(total - 1, 0), 0)

    # Warm up stateful filters
    _WARMUP = 10
    for _w in range(max(0, idx - _WARMUP), idx):
        pipeline.process_frame(read_frame(selection.video_path, _w))

    frame = read_frame(selection.video_path, idx)
    t0 = time.time()
    result = pipeline.process_frame(frame)
    fps = 1.0 / max(time.time() - t0, 0.001)

    with col_fps:
        st.metric("Inference FPS", f"{fps:.1f}")

    display_four_panels(result)
    st.divider()

    # ── Export full video ──────────────────────────────────────────────────────
    st.subheader("Export full video")
    skip_frames = selection.skip_frames
    st.caption("Runs the full pipeline on every frame and saves results to `data/output/`.")

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button("Process & save", type="primary", use_container_width=True)
    with col_info:
        frames_to_process = (total + skip_frames - 1) // skip_frames
        strategy_name = selection.skip_strategy[0]
        st.caption(f"Processing **{frames_to_process} frames** (Skip: {skip_frames}, Strategy: `{strategy_name}`).")

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

        def _on_progress(done, total_f, current_fps):
            progress_bar.progress(done / max(total_f, 1))
            status.caption(f"Frame {done} / {total_f} | {current_fps:.1f} FPS")

        with st.spinner("Processing video..."):
            paths = process_video(
                pipeline, selection.video_path, run_dir, _on_progress,
                skip_frames=skip_frames, skip_strategy=strategy_instance
            )

        st.success(f"Saved to `{run_dir.name}`")
        display_synced_player(paths)

    st.divider()

    # ── Browse saved outputs ───────────────────────────────────────────────────
    with st.expander("Browse saved outputs", expanded=False):
        if not OUTPUT_DIR.exists():
            st.info("No saved runs yet.")
        else:
            run_dirs = sorted([d for d in OUTPUT_DIR.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
            run_names = [d.name for d in run_dirs]
            if not run_names:
                st.info("No saved runs yet.")
            else:
                selected_run = st.selectbox("Select a run", run_names, key="browse_run_select")
                rd = OUTPUT_DIR / selected_run
                conf_p = rd / "config.json"
                if conf_p.exists():
                    with open(conf_p) as f: st.json(json.load(f))
                
                v_paths = {
                    "original": rd / "original.mp4",
                    "mask": rd / "mask.mp4",
                    "raw": rd / "raw.mp4",
                    "composite": rd / "composite.mp4"
                }
                if all(p.exists() for p in v_paths.values()):
                    display_synced_player(v_paths)
                else:
                    st.warning("Video files missing in this run directory.")

with tab_live:
    st.subheader("Démo en temps réel")
    col_ctrl, col_stats = st.columns([3, 1])
    with col_ctrl:
        c1, c2 = st.columns(2)
        with c1:
            bg_mode = st.selectbox("Arrière-plan", ["Original", "Noir", "Flou (Portrait)", "Bureau Moderne", "Nature"], index=0)
        with c2:
            live_skip = st.number_input("Skip Frames (Live)", min_value=1, value=1)
    with col_stats:
        fps_placeholder = st.empty()
    
    run_cam = st.toggle("Démarrer la caméra", value=False)
    
    if run_cam:
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        st_frame = st.empty()
        st_debug = st.empty()
        
        bg_cache = {}
        bg_resized_cache = {}

        def get_bg(mode, w, h):
            if mode not in bg_cache:
                if mode == "Bureau Moderne":
                    url = "https://images.unsplash.com/photo-1497366216548-37526070297c?auto=format&fit=crop&w=1280&q=80"
                elif mode == "Nature":
                    url = "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?auto=format&fit=crop&w=1280&q=80"
                else: return None
                try:
                    resp = urllib.request.urlopen(url)
                    img = cv2.imdecode(np.asarray(bytearray(resp.read()), dtype="uint8"), 1)
                    bg_cache[mode] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except: return None
            
            key = (mode, w, h)
            if key not in bg_resized_cache:
                bg_resized_cache[key] = cv2.resize(bg_cache[mode], (w, h))
            return bg_resized_cache[key]

        idx = 0
        last_mask = None
        fps_history = []
        inf_history = []
        
        while run_cam:
            loop_start = time.time()
            ret, bgr = cap.read()
            if not ret: break
            rgb_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h_full, w_full = rgb_full.shape[:2]
            
            disp_w = 640
            scale = disp_w / w_full
            disp_h = int(h_full * scale)
            rgb = cv2.resize(rgb_full, (disp_w, disp_h))
            
            inf_start = time.time()
            inf_ran = False
            if bg_mode != "Original":
                if idx % live_skip == 0 or last_mask is None:
                    res = pipeline.process_frame(rgb_full)
                    last_mask_full = res.get("final_mask", res.get("mask"))
                    if last_mask_full is None:
                        last_mask_full = np.ones((h_full, w_full), dtype=np.float32)
                    last_mask = cv2.resize(last_mask_full, (disp_w, disp_h))
                    inf_ran = True
                
                m3d = last_mask[:, :, np.newaxis]
            
            if inf_ran:
                inf_history.append(time.time() - inf_start)
                if len(inf_history) > 30: inf_history.pop(0)
            
            if bg_mode == "Original":
                final = rgb
            elif bg_mode == "Noir": 
                final = (rgb * m3d).astype(np.uint8)
            elif bg_mode == "Flou (Portrait)":
                small = cv2.resize(rgb, (disp_w//4, disp_h//4))
                small_blur = cv2.GaussianBlur(small, (15, 15), 0)
                blurred = cv2.resize(small_blur, (disp_w, disp_h))
                final = (rgb * m3d + blurred * (1 - m3d)).astype(np.uint8)
            else:
                curr_bg = get_bg(bg_mode, disp_w, disp_h)
                if curr_bg is not None:
                    final = (rgb * m3d + curr_bg * (1 - m3d)).astype(np.uint8)
                else:
                    final = (rgb * m3d).astype(np.uint8)

            st_frame.image(final, channels="RGB", use_container_width=True, output_format="JPEG")
            fps_history.append(time.time() - loop_start)
            if len(fps_history) > 30: fps_history.pop(0)
            
            idx += 1
            if idx % 5 == 0:
                avg_loop = sum(fps_history) / len(fps_history)
                avg_inf = sum(inf_history) / len(inf_history) if inf_history else 0
                current_fps = 1.0 / avg_loop if avg_loop > 0 else 0
                fps_placeholder.metric("Live FPS", f"{current_fps:.1f}")
                st_debug.caption(f"IA: {avg_inf*1000:.1f}ms | Boucle: {avg_loop*1000:.1f}ms")
                
        cap.release()
