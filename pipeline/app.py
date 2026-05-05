import os
os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "0"

import time
import collections
import hashlib
import json
import threading
import queue
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

col_title, col_download = st.columns([3, 1])
with col_title:
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
            if config_to_apply.get("video_source"):
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

# ── Sidebar Selection ──────────────────────────────────────────────────────────
selection = render_sidebar()

# ── Config Export Button ───────────────────────────────────────────────────────
config_to_save = {
    "model_name": selection.model_name,
    "weights_path": selection.weights_path,
    "video_source": str(selection.video_path) if selection.video_path else None,
    "model_params": selection.model_params,
    "preprocessors": selection.preprocessors,
    "postprocessors": selection.postprocessors,
    "upsampler": selection.upsampler,
    "bg_color": selection.bg_color,
    "skip_frames": selection.skip_frames,
    "skip_strategy": selection.skip_strategy,
    "exported_at": datetime.now().isoformat(),
}
config_json = json.dumps(config_to_save, indent=4)

with col_download:
    st.download_button(
        label="💾 Exporter Config (JSON)",
        data=config_json,
        file_name=f"config_{selection.model_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        width="stretch",
        key="global_export_config"
    )

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
        run_clicked = st.button("Process & save", type="primary", width="stretch")
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
        c1, c2, c3 = st.columns(3)
        with c1:
            bg_mode = st.selectbox(
                "Arrière-plan",
                ["Couleur sidebar", "Noir", "Flou (Portrait)", "Bureau Moderne", "Nature"],
                index=0,
            )
        with c2:
            live_skip = st.number_input("Skip Frames (Live)", min_value=1, value=1)
        with c3:
            show_panels = st.checkbox("Vue 4 panneaux", value=False)
    with col_stats:
        fps_placeholder = st.empty()
        inf_placeholder = st.empty()

    run_cam = st.toggle("Démarrer la caméra", value=False)
    cam_status = st.empty()

    if run_cam:
        pipeline.reset()

        # Try opening camera — with AVFOUNDATION backend on macOS, then fallback
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            cam_status.error(
                "❌ Impossible d'ouvrir la caméra (index 0). "
                "Sur macOS : Réglages système → Confidentialité et sécurité → Caméra "
                "→ autorisez l'accès pour Terminal ou l'app Python."
            )
        else:
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            # Warm up: discard first few frames (camera auto-exposure)
            for _ in range(3):
                cap.read()

            ret_test, bgr_test = cap.read()
            if not ret_test:
                cam_status.error(
                    "❌ La caméra s'est ouverte mais ne retourne pas de frame. "
                    "Vérifiez qu'elle n'est pas utilisée par une autre application."
                )
                cap.release()
            else:
                cam_status.success("✅ Caméra ouverte — flux en cours...")

                # ── UI Placeholders ───────────────────────────────────────────
                if show_panels:
                    col_a, col_b = st.columns(2)
                    col_c, col_d = st.columns(2)
                    ph_orig = col_a.empty()
                    ph_final = col_b.empty()
                    ph_raw = col_c.empty()
                    ph_fin_mask = col_d.empty()
                else:
                    ph_final = st.empty()

                ph_profiling = st.empty()
                st_debug = st.empty()

                # ── Background Worker Setup ──────────────────────────────────
                # Use a shared dict for state and queues to survive reruns
                if "cam_worker" not in st.session_state:
                    st.session_state.cam_worker = None
                
                # Queues: maxsize=1 ensures we always have the LATEST frame (low latency)
                q_raw = queue.Queue(maxsize=1)
                q_result = queue.Queue(maxsize=1)
                stop_event = threading.Event()

                def capture_thread(cap, q_raw, stop_event):
                    while not stop_event.is_set():
                        ret, bgr = cap.read()
                        if not ret: break
                        t_capture = time.time()
                        # Lossy: overwrite if full
                        if q_raw.full():
                            try: q_raw.get_nowait()
                            except queue.Empty: pass
                        q_raw.put((bgr, t_capture))
                    cap.release()

                def inference_thread(pipeline, q_raw, q_result, stop_event, live_skip, bg_mode, get_bg_func):
                    idx = 0
                    last_result = None
                    inf_history = collections.deque(maxlen=30)
                    DISP_W = 640
                    
                    while not stop_event.is_set():
                        try:
                            bgr, t_capture = q_raw.get(timeout=0.1)
                        except queue.Empty:
                            continue
                        
                        rgb_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        h_full, w_full = rgb_full.shape[:2]
                        disp_h = int(h_full * DISP_W / w_full)
                        
                        if idx % live_skip == 0 or last_result is None:
                            try:
                                t0 = time.perf_counter()
                                last_result = pipeline.process_frame(rgb_full)
                                inf_history.append(time.perf_counter() - t0)
                            except Exception:
                                continue
                        
                        # Move Rendering/Composition here to offload UI thread
                        rgb_disp = cv2.resize(rgb_full, (DISP_W, disp_h))
                        final_mask = cv2.resize(last_result["final_mask"], (DISP_W, disp_h))
                        m3d = final_mask[:, :, np.newaxis]
                        
                        if bg_mode == "Couleur sidebar":
                            final = cv2.resize(last_result["final"], (DISP_W, disp_h))
                        elif bg_mode == "Noir":
                            final = (rgb_disp * m3d).astype(np.uint8)
                        elif bg_mode == "Flou (Portrait)":
                            small = cv2.resize(rgb_disp, (DISP_W // 4, disp_h // 4))
                            blurred = cv2.resize(cv2.GaussianBlur(small, (15, 15), 0), (DISP_W, disp_h))
                            final = (rgb_disp * m3d + blurred * (1.0 - m3d)).astype(np.uint8)
                        else:
                            curr_bg = get_bg_func(bg_mode, DISP_W, disp_h)
                            if curr_bg is not None:
                                final = (rgb_disp * m3d + curr_bg * (1.0 - m3d)).astype(np.uint8)
                            else:
                                final = (rgb_disp * m3d).astype(np.uint8)

                        # Prepare extra debug panels if needed
                        panels = {}
                        # We only prepare panels if we really need them to save CPU
                        # (Passing them through the queue is expensive)
                        
                        res_data = {
                            "final": final,
                            "preprocessed": cv2.resize(last_result["preprocessed"], (DISP_W, disp_h)),
                            "raw_mask": (cv2.resize(last_result["raw_mask"], (DISP_W, disp_h)) * 255).astype(np.uint8),
                            "final_mask_viz": (final_mask * 255).astype(np.uint8),
                            "inf_history": list(inf_history),
                            "idx": idx,
                            "t_capture": t_capture
                        }
                        
                        if q_result.full():
                            try: q_result.get_nowait()
                            except queue.Empty: pass
                        q_result.put(res_data)
                        idx += 1

                # ── Background rendering logic ───────────────────────────────
                bg_cache = {}
                bg_resized_cache = {}

                def get_bg(mode: str, w: int, h: int):
                    if mode not in bg_cache:
                        urls = {
                            "Bureau Moderne": "https://images.unsplash.com/photo-1497366216548-37526070297c?auto=format&fit=crop&w=1280&q=80",
                            "Nature": "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?auto=format&fit=crop&w=1280&q=80",
                        }
                        url = urls.get(mode)
                        if url is None: return None
                        try:
                            import urllib.request
                            resp = urllib.request.urlopen(url)
                            img = cv2.imdecode(np.asarray(bytearray(resp.read()), dtype="uint8"), 1)
                            bg_cache[mode] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        except Exception: return None
                    
                    key = (mode, w, h)
                    if key not in bg_resized_cache:
                        bg_resized_cache[key] = cv2.resize(bg_cache[mode], (w, h))
                    return bg_resized_cache[key]

                DISP_W = 640
                delivery_timestamps = collections.deque(maxlen=30)
                e2e_latencies = collections.deque(maxlen=30)

                # Metrics placeholders in sidebar
                st.sidebar.markdown("---")
                st.sidebar.subheader("Performance")
                real_fps_ph = st.sidebar.empty()
                model_fps_ph = st.sidebar.empty()
                e2e_lat_ph = st.sidebar.empty()

                # Start workers
                t_cap = threading.Thread(target=capture_thread, args=(cap, q_raw, stop_event), daemon=True)
                t_inf = threading.Thread(target=inference_thread, args=(pipeline, q_raw, q_result, stop_event, live_skip, bg_mode, get_bg), daemon=True)
                
                t_cap.start()
                t_inf.start()

                try:
                    last_ui_update = 0
                    while True:
                        try:
                            # Use a short timeout
                            data = q_result.get(timeout=0.02)
                        except queue.Empty:
                            if stop_event.is_set(): break
                            continue
                        
                        t_now = time.time()
                        
                        # Update images (placeholder already exists)
                        if show_panels:
                            ph_orig.image(data["preprocessed"], caption="Pre-processing (BBoxes)", width="stretch", output_format="JPEG")
                            ph_final.image(data["final"], channels="RGB", caption="Composite", width="stretch", output_format="JPEG")
                            ph_raw.image(data["raw_mask"], caption="Masque brut", width="stretch", output_format="JPEG")
                            ph_fin_mask.image(data["final_mask_viz"], caption="Masque final (post-proc)", width="stretch", output_format="JPEG")
                        else:
                            ph_final.image(data["final"], channels="RGB", width="stretch", output_format="JPEG")

                        # Metrics logic
                        delivery_timestamps.append(t_now)
                        e2e_latencies.append(t_now - data["t_capture"])

                        # Update metrics only every 100ms to avoid UI stutter
                        if t_now - last_ui_update > 0.1:
                            avg_inf = sum(data["inf_history"]) / len(data["inf_history"]) if data["inf_history"] else 0
                            model_fps = 1.0 / avg_inf if avg_inf > 0 else 0
                            avg_e2e = sum(e2e_latencies) / len(e2e_latencies) if e2e_latencies else 0
                            if len(delivery_timestamps) > 1:
                                real_fps = (len(delivery_timestamps) - 1) / (delivery_timestamps[-1] - delivery_timestamps[0])
                            else: real_fps = 0.0

                            real_fps_ph.metric("FPS Réel (Affichage)", f"{real_fps:.1f}")
                            model_fps_ph.metric("FPS Théorique (IA)", f"{model_fps:.1f}")
                            e2e_lat_ph.metric("Latence Totale (E2E)", f"{avg_e2e*1000:.0f} ms")
                            last_ui_update = t_now

                        # Crucial: give a tiny bit of time for Streamlit's event loop
                        time.sleep(0.001)

                        inf_placeholder.metric("FPS Réel (Display)", f"{real_fps:.1f}")
                        fps_placeholder.metric("Latence Inférence", f"{avg_inf * 1000:.0f} ms")
                            
                        avg_loop = 1.0 / real_fps if real_fps > 0 else 0
                        st_debug.caption(
                                f"Frame {idx} | "
                                f"Pipeline IA : {avg_inf*1000:.1f}ms ({model_fps:.1f} FPS th.) | "
                                f"Overhead Streamlit : {max(0, (avg_loop - avg_inf)*1000):.1f}ms"
                            )

                        if "timings" in result:
                                t = result["timings"]
                                table_md = "| Composant | Latence (ms) |\n| :--- | :--- |\n"
                                for k, v in t.items():
                                    if k.startswith("pre_"): table_md += f"| 🟢 Pre: {k[4:]} | {v*1000:.2f} |\n"
                                table_md += f"| 🧠 **Inférence IA** | **{t.get('model_inference', 0)*1000:.2f}** |\n"
                                table_md += f"| ⬆️ **Upsampling** | **{t.get('upsampling', 0)*1000:.2f}** |\n"
                                for k, v in t.items():
                                    if k.startswith("post_"): table_md += f"| 🔵 Post: {k[5:]} | {v*1000:.2f} |\n"
                                table_md += f"| 🎬 Composition | {t.get('compositing', 0)*1000:.2f} |\n"
                                table_md += f"| --- | --- |\n"
                                table_md += f"| ⏱️ **TOTAL PIPELINE** | **{t.get('total_pipeline', 0)*1000:.2f}** |"
                                ph_profiling.markdown(table_md)

                finally:
                    # Clean shutdown
                    stop_event.set()
                    t_cap.join(timeout=1.0)
                    t_inf.join(timeout=1.0)

    else:
        cam_status.info("Caméra arrêtée.")

