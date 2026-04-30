import os
os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "0"

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
        use_container_width=True,
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

                bg_cache: dict = {}
                bg_resized_cache: dict = {}

                def get_bg(mode: str, w: int, h: int):
                    if mode not in bg_cache:
                        urls = {
                            "Bureau Moderne": "https://images.unsplash.com/photo-1497366216548-37526070297c?auto=format&fit=crop&w=1280&q=80",
                            "Nature": "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?auto=format&fit=crop&w=1280&q=80",
                        }
                        url = urls.get(mode)
                        if url is None:
                            return None
                        try:
                            resp = urllib.request.urlopen(url)
                            img = cv2.imdecode(np.asarray(bytearray(resp.read()), dtype="uint8"), 1)
                            bg_cache[mode] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        except Exception:
                            return None
                    key = (mode, w, h)
                    if key not in bg_resized_cache:
                        bg_resized_cache[key] = cv2.resize(bg_cache[mode], (w, h))
                    return bg_resized_cache[key]

                DISP_W = 640
                idx = 0
                last_result: dict | None = None
                fps_history: list[float] = []
                inf_history: list[float] = []

                # Inject the first frame we already read above
                frames_to_process = [bgr_test]

                while True:
                    loop_start = time.time()

                    if frames_to_process:
                        bgr = frames_to_process.pop(0)
                    else:
                        ret, bgr = cap.read()
                        if not ret:
                            cam_status.warning(f"⚠️ Lecture caméra échouée à la frame {idx}.")
                            break

                    rgb_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    h_full, w_full = rgb_full.shape[:2]
                    disp_h = int(h_full * DISP_W / w_full)
                    rgb_disp = cv2.resize(rgb_full, (DISP_W, disp_h))

                    # Always run the full pipeline (pre + model + post)
                    if idx % live_skip == 0 or last_result is None:
                        try:
                            inf_start = time.time()
                            last_result = pipeline.process_frame(rgb_full)
                            inf_history.append(time.time() - inf_start)
                            if len(inf_history) > 30:
                                inf_history.pop(0)
                        except Exception as exc:
                            cam_status.error(f"⚠️ Erreur pipeline : {exc}")
                            # Show raw frame so the user sees something
                            ph_final.image(rgb_disp, channels="RGB", use_container_width=True, output_format="JPEG")
                            idx += 1
                            fps_history.append(time.time() - loop_start)
                            continue

                    result = last_result
                    final_mask = cv2.resize(result["final_mask"], (DISP_W, disp_h))
                    m3d = final_mask[:, :, np.newaxis]

                    if bg_mode == "Couleur sidebar":
                        final = cv2.resize(result["final"], (DISP_W, disp_h))
                    elif bg_mode == "Noir":
                        final = (rgb_disp * m3d).astype(np.uint8)
                    elif bg_mode == "Flou (Portrait)":
                        small = cv2.resize(rgb_disp, (DISP_W // 4, disp_h // 4))
                        blurred = cv2.resize(cv2.GaussianBlur(small, (15, 15), 0), (DISP_W, disp_h))
                        final = (rgb_disp * m3d + blurred * (1.0 - m3d)).astype(np.uint8)
                    else:
                        curr_bg = get_bg(bg_mode, DISP_W, disp_h)
                        if curr_bg is not None:
                            final = (rgb_disp * m3d + curr_bg * (1.0 - m3d)).astype(np.uint8)
                        else:
                            final = (rgb_disp * m3d).astype(np.uint8)

                    if show_panels:
                        # On affiche la version "preprocessed" (avec les bboxes du Person Zoom)
                        pre_disp = cv2.resize(result["preprocessed"], (DISP_W, disp_h))
                        ph_orig.image(pre_disp, caption="Pre-processing (BBoxes)", use_container_width=True, output_format="JPEG")
                        
                        ph_final.image(final, channels="RGB", caption="Composite", use_container_width=True, output_format="JPEG")
                        
                        raw_uint8 = (cv2.resize(result["raw_mask"], (DISP_W, disp_h)) * 255).astype(np.uint8)
                        ph_raw.image(raw_uint8, caption="Masque brut", use_container_width=True, output_format="JPEG")
                        
                        fin_uint8 = (final_mask * 255).astype(np.uint8)
                        ph_fin_mask.image(fin_uint8, caption="Masque final (post-proc)", use_container_width=True, output_format="JPEG")
                    else:
                        ph_final.image(final, channels="RGB", use_container_width=True, output_format="JPEG")

                    fps_history.append(time.time() - loop_start)
                    if len(fps_history) > 30:
                        fps_history.pop(0)

                    idx += 1
                    if idx % 5 == 0:
                        avg_loop = sum(fps_history) / len(fps_history)
                        avg_inf = sum(inf_history) / len(inf_history) if inf_history else 0
                        model_fps = 1.0 / avg_inf if avg_inf > 0 else 0
                        
                        # Affichage prioritaire pour le benchmark de production
                        inf_placeholder.metric("FPS Modèle (Brut)", f"{model_fps:.1f}")
                        fps_placeholder.metric("Latence Inférence", f"{avg_inf * 1000:.0f} ms")
                        
                        st_debug.caption(
                            f"Frame {idx} | "
                            f"Pipeline IA : {avg_inf*1000:.1f}ms ({model_fps:.1f} FPS) | "
                            f"Overhead Streamlit : {(avg_loop - avg_inf)*1000:.1f}ms"
                        )

                        # Affichage du profiling détaillé sous forme de tableau
                        if "timings" in result:
                            t = result["timings"]
                            table_md = "| Composant | Latence (ms) |\n| :--- | :--- |\n"
                            
                            # Section Pre
                            for k, v in t.items():
                                if k.startswith("pre_"):
                                    table_md += f"| 🟢 Pre: {k[4:]} | {v*1000:.2f} |\n"
                            
                            # Section Modèle
                            table_md += f"| 🧠 **Inférence IA** | **{t.get('model_inference', 0)*1000:.2f}** |\n"
                            
                            # Section Upsampling
                            table_md += f"| ⬆️ **Upsampling** | **{t.get('upsampling', 0)*1000:.2f}** |\n"
                            
                            # Section Post
                            for k, v in t.items():
                                if k.startswith("post_"):
                                    table_md += f"| 🔵 Post: {k[5:]} | {v*1000:.2f} |\n"
                            
                            # Section Rendu
                            table_md += f"| 🎬 Composition | {t.get('compositing', 0)*1000:.2f} |\n"
                            table_md += f"| --- | --- |\n"
                            table_md += f"| ⏱️ **TOTAL PIPELINE** | **{t.get('total_pipeline', 0)*1000:.2f}** |"
                            
                            ph_profiling.markdown(table_md)

                cap.release()
                cam_status.info("Caméra arrêtée.")

