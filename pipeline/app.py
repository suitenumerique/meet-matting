import os
os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

import time
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
import numpy as np
from config import OUTPUT_DIR
from core.pipeline import MattingPipeline
from core.registry import models, postprocessors, preprocessors
from core.video_io import frame_count, read_frame
from core.video_processing import process_video
from ui.sidebar import render_sidebar
from ui.video_panel import display_four_panels

import urllib.request
import tempfile
import cv2

# Auto-discover all plugins.
preprocessors.discover("preprocessing")
models.discover("models")
postprocessors.discover("postprocessing")

st.set_page_config(layout="wide", page_title="Matting Pipeline Lab")
st.title("Background matting pipeline")

# ── Config Import ──────────────────────────────────────────────────────────────
with st.sidebar.expander("Import Config", expanded=False):
    uploaded_config = st.file_uploader("Upload config.json", type="json")
    
    should_apply = False
    if uploaded_config is not None:
        file_id = f"{uploaded_config.name}_{uploaded_config.size}"
        if st.session_state.get("last_config_id") != file_id:
            should_apply = True
            st.session_state["last_config_id"] = file_id

    if should_apply:
        try:
            config_to_apply = json.load(uploaded_config)
            if "model_name" in config_to_apply: st.session_state["model_select"] = config_to_apply["model_name"]
            if "weights_path" in config_to_apply: st.session_state["weights_path"] = config_to_apply["weights_path"]
            if "video_source" in config_to_apply:
                from core.video_io import list_videos
                from config import VIDEO_DIR
                source_name = Path(config_to_apply["video_source"]).name
                for v in list_videos(VIDEO_DIR):
                    if v.name == source_name:
                        st.session_state["video_select"] = v
                        break
            if "model_params" in config_to_apply:
                for k, v in config_to_apply["model_params"].items(): st.session_state[f"model_{k}"] = v
            if "preprocessors" in config_to_apply:
                st.session_state["pre_select"] = [p[0] for p in config_to_apply["preprocessors"]]
                for i, (name, params) in enumerate(config_to_apply["preprocessors"]):
                    for pk, pv in params.items(): st.session_state[f"pre_{i}_{pk}"] = pv
            if "postprocessors" in config_to_apply:
                st.session_state["post_select"] = [p[0] for p in config_to_apply["postprocessors"]]
                for i, (name, params) in enumerate(config_to_apply["postprocessors"]):
                    for pk, pv in params.items(): st.session_state[f"post_{i}_{pk}"] = pv
            st.success("Configuration chargée !")
            st.rerun()
        except Exception as e:
            st.error(f"Erreur lors de l'import : {e}")

selection = render_sidebar()

# Build pipeline from selection.
pre_components = [preprocessors.get(name)(**params) for name, params in selection.preprocessors]
model = models.get(selection.model_name)(**selection.model_params)
model.load(selection.weights_path)
post_components = [postprocessors.get(name)(**params) for name, params in selection.postprocessors]
pipeline = MattingPipeline(pre_components, model, post_components)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab_lab, tab_live = st.tabs(["📽️ Laboratoire Vidéo", "📸 Test Caméra"])

with tab_lab:
    if selection.video_path is None:
        st.info("Ajoutez une vidéo dans `data/videos/` pour commencer.")
    else:
        st.subheader(f"Aperçu : {selection.video_path.name}")
        col_slider, col_fps = st.columns([4, 1])
        total = frame_count(selection.video_path)
        with col_slider:
            idx = st.slider("Frame", 0, max(total - 1, 0), 0)
        
        frame = read_frame(selection.video_path, idx)
        t0 = time.time()
        result = pipeline.process_frame(frame)
        fps = 1.0 / max(time.time() - t0, 0.001)
        
        with col_fps:
            st.metric("Inference FPS", f"{fps:.1f}")
        
        display_four_panels(result)
        st.divider()

        # Export Logic
        st.subheader("Exporter la vidéo")
        col_skip, col_btn = st.columns([1, 1])
        with col_skip:
            skip_frames = st.number_input("Skip Frames", min_value=1, value=1)
        with col_btn:
            st.write("") # Spacer
            run_clicked = st.button("Lancer l'export", type="primary", use_container_width=True)

        if run_clicked:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = OUTPUT_DIR / f"{selection.video_path.stem}__{selection.model_name}__{timestamp}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Save config
            config_data = {
                "timestamp": timestamp, "video_source": str(selection.video_path),
                "model_name": selection.model_name, "model_params": selection.model_params,
                "preprocessors": selection.preprocessors, "postprocessors": selection.postprocessors,
                "skip_frames": skip_frames,
            }
            with open(run_dir / "config.json", "w") as f: json.dump(config_data, f, indent=4)

            progress_bar = st.progress(0.0)
            status = st.empty()
            def _on_progress(d, t, f):
                progress_bar.progress(d / max(t, 1))
                status.caption(f"Frame {d}/{t} | {f:.1f} FPS")

            with st.spinner("Export en cours..."):
                paths = process_video(pipeline, selection.video_path, run_dir, _on_progress, skip_frames=skip_frames)
            
            st.success(f"Sauvegardé dans `{run_dir.name}`")
            col_m, col_c = st.columns(2)
            with col_m: st.video(paths["mask"].read_bytes())
            with col_c: st.video(paths["composite"].read_bytes())

with tab_live:
    st.subheader("Démo en temps réel")
    col_ctrl, col_stats = st.columns([3, 1])
    with col_ctrl:
        c1, c2 = st.columns(2)
        with c1:
            bg_mode = st.selectbox("Arrière-plan", ["Noir", "Flou (Portrait)", "Bureau Moderne", "Nature"])
        with c2:
            live_skip = st.number_input("Skip Frames (Live)", min_value=1, value=1, help="1 = inférence à chaque frame, 2 = une sur deux, etc.")
    with col_stats:
        fps_placeholder = st.empty()
    
    run_cam = st.toggle("Démarrer la caméra", value=False)
    
    if run_cam:
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        # Try to force 60 FPS and 720p for capture
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        st_frame = st.empty()
        st_debug = st.empty()
        
        # Background Cache to avoid re-downloading
        bg_cache = {}
        def get_bg(mode):
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
            return bg_cache.get(mode)

        idx = 0
        last_mask = None
        t_start = time.time()
        
        while run_cam:
            loop_start = time.time()
            ret, bgr = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            
            # --- INFERENCE ---
            inf_start = time.time()
            if idx % live_skip == 0 or last_mask is None:
                res = pipeline.process_frame(rgb)
                last_mask = res.get("final_mask", res.get("mask"))
                if last_mask is None:
                    last_mask = np.ones(rgb.shape[:2], dtype=np.float32)
            inf_time = (time.time() - inf_start) * 1000 # ms
            
            # Ensure mask size
            if last_mask.shape[:2] != rgb.shape[:2]:
                last_mask = cv2.resize(last_mask, (rgb.shape[1], rgb.shape[0]))
            
            m3d = last_mask[:, :, np.newaxis]
            curr_bg_img = get_bg(bg_mode)

            # Composition
            if bg_mode == "Noir": 
                final = (rgb * m3d).astype(np.uint8)
            elif bg_mode == "Flou (Portrait)":
                # Small kernel for speed in live
                blurred = cv2.GaussianBlur(rgb, (31, 31), 0)
                final = (rgb * m3d + blurred * (1 - m3d)).astype(np.uint8)
            elif curr_bg_img is not None:
                bg_r = cv2.resize(curr_bg_img, (rgb.shape[1], rgb.shape[0]))
                final = (rgb * m3d + bg_r * (1 - m3d)).astype(np.uint8)
            else: 
                final = (rgb * m3d).astype(np.uint8)

            # --- DISPLAY OPTIMIZATION (480p preview) ---
            disp_w = 640
            scale = disp_w / rgb.shape[1]
            disp_h = int(rgb.shape[0] * scale)
            final_disp = cv2.resize(final, (disp_w, disp_h))

            st_frame.image(final_disp, channels="RGB", use_container_width=True)
            
            idx += 1
            if idx % 10 == 0:
                total_time = (time.time() - loop_start) * 1000 # ms
                current_fps = 10.0 / max(time.time() - t_start, 0.001)
                fps_placeholder.metric("Live FPS", f"{current_fps:.1f}")
                st_debug.caption(f"Inférence: {inf_time:.1f}ms | Loop: {total_time:.1f}ms")
                t_start = time.time()
                
        cap.release()

# ── History ──────────────────────────────────────────────────────────────────
st.divider()
with st.expander("Historique des exports", expanded=False):
    if OUTPUT_DIR.exists():
        runs = sorted([p for p in OUTPUT_DIR.iterdir() if p.is_dir()], reverse=True)
        if runs:
            sel_run = st.selectbox("Sélectionner un export", [r.name for r in runs])
            r_dir = OUTPUT_DIR / sel_run
            
            # Afficher config
            conf_path = r_dir / "config.json"
            if conf_path.exists():
                with open(conf_path, "r") as f:
                    st.json(json.load(f))
            
            c1, c2 = st.columns(2)
            mask_path = r_dir / "mask.mp4"
            if mask_path.exists():
                with c1:
                    st.caption("Masque")
                    st.video(mask_path.read_bytes())
            
            comp_path = r_dir / "composite.mp4"
            if comp_path.exists():
                with c2:
                    st.caption("Composite")
                    st.video(comp_path.read_bytes())
        else:
            st.info("Aucun export trouvé.")
    else:
        st.info("Le dossier d'output n'existe pas encore.")
