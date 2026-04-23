"""
Tableau de bord Streamlit pour le Benchmark Video Matting.

Offre une interface moderne pour :
  - Sélectionner les modèles.
  - Choisir le nombre de vidéos (aléatoire ou non).
  - Activer la sauvegarde des masques.
  - Visualiser les résultats en temps réel.
"""

import os
# Désactivation TOTALE des logs MediaPipe/GLog au démarrage
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import pandas as pd
import sys
import os
import warnings
from pathlib import Path
import time
import logging
import os
import queue
import sys
import threading
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Silence Streamlit context warnings handled in benchmark/__init__.py


# Ajouter le parent au path pour importer les modules benchmark
sys.path.append(str(Path(__file__).parent.parent))

# Silence MediaPipe / TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

from benchmark.models import MODEL_REGISTRY
from benchmark.runner import (
    _chroma_key_to_mask,
    _get_video_info,
    _is_chroma_key,
    _load_ground_truth_masks,
    compute_metrics_on_output,
    discover_datasets,
    get_frame_at,
    load_masks_from_mask_video,
    run_benchmark,
)
from benchmark.config import DATASETS, GROUND_TRUTH_DIR, OUTPUT_DIR, TEMP_RESULTS_DIR, VIDEOS_DIR

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Video Matting Benchmark",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style Custom
st.markdown("""
<style>
  .stButton>button { width:100%; border-radius:5px; height:3em;
                     background-color:#007bff; color:white; font-weight:bold; }
  .metric-card { background:#fff; padding:15px; border-radius:10px;
                 box-shadow:0 2px 4px rgba(0,0,0,.05); }
</style>
""", unsafe_allow_html=True)

# ── Shared constants ──────────────────────────────────────────────────────────
DISPLAY_COLS = [
    "model", "video", "status",
    "iou_mean", "boundary_f_mean", "flow_warping_error",
    "latency_p95_ms", "flops_per_frame",
]

METRIC_GUIDE = """
### 📖 Guide des métriques

| Métrique | Plage | Sens | Seuils indicatifs |
|---|---|---|---|
| **IoU** (Intersection over Union) | [0, 1] | ↑ mieux | < 0.6 mauvais · 0.7–0.85 correct · > 0.85 bon |
| **Boundary F-measure** | [0, 1] | ↑ mieux | < 0.5 contours flous · > 0.75 contours nets |
| **Flow Warping Error** | [0, 1] | ↓ mieux | < 0.02 très stable · 0.02–0.07 légère instab. · > 0.07 clignotement visible |
| **Latence p95** | ms | ↓ mieux | < 33 ms → 30 fps temps réel · < 16 ms → 60 fps |
| **FLOPs/frame** | — | ↓ mieux | Comparatif relatif entre modèles |

**IoU** : proportion de pixels correctement classés (intersection / union entre masque prédit et vérité terrain). Un IoU de 1.0 signifie un accord parfait pixel à pixel.

**Boundary F-measure** : précision des contours. Utilise la méthode DAVIS (érosion morphologique + tolérance spatiale adaptée à la résolution). Un score élevé signifie que les bords de la personne sont bien détourés.

**Flow Warping Error** : stabilité temporelle. Mesure à quel point le masque de la frame *t* est cohérent avec celui de la frame *t-1* warpé par le flux optique. Un score faible signifie une vidéo sans clignotement.

**Latence p95** : 95e percentile des temps de prédiction frame-par-frame (batch=1). Plus robuste que la moyenne car insensible aux outliers de démarrage.
"""


def _styled_df(df: pd.DataFrame):
    """Apply highlight styling to a results DataFrame."""
    cols = [c for c in df.columns if c in DISPLAY_COLS]
    sub = df[cols].copy()
    s = sub.style
    if "iou_mean" in sub.columns:
        s = s.highlight_max(subset=["iou_mean"], color="#d4edda")
    if "boundary_f_mean" in sub.columns:
        s = s.highlight_max(subset=["boundary_f_mean"], color="#d4edda")
    if "latency_p95_ms" in sub.columns:
        s = s.highlight_min(subset=["latency_p95_ms"], color="#d4edda")
    if "flow_warping_error" in sub.columns:
        s = s.highlight_min(subset=["flow_warping_error"], color="#d4edda")
    return s


def _scatter_chart(df: pd.DataFrame):
    """Render a latency vs IoU scatter plot using Altair."""
    try:
        import altair as alt
    except ImportError:
        st.info("Altair non disponible — installer avec `pip install altair`.")
        return

    needed = {"latency_p95_ms", "iou_mean", "model"}
    if not needed.issubset(df.columns):
        return

    plot_df = df[df["status"] == "OK"].dropna(subset=list(needed)).copy()
    if plot_df.empty:
        return

    # Normalise flops for bubble size (fallback if missing)
    if "flops_per_frame" in plot_df.columns:
        plot_df["_size"] = (plot_df["flops_per_frame"].fillna(0) / 1e6).clip(lower=5)
    else:
        plot_df["_size"] = 50

    tooltip_cols = [c for c in DISPLAY_COLS if c in plot_df.columns]

    chart = (
        alt.Chart(plot_df)
        .mark_circle(opacity=0.8)
        .encode(
            x=alt.X("latency_p95_ms:Q", title="Latence p95 (ms)"),
            y=alt.Y("iou_mean:Q", title="IoU moyen", scale=alt.Scale(zero=False)),
            size=alt.Size("_size:Q", legend=None, scale=alt.Scale(range=[100, 800])),
            color=alt.Color("model:N", title="Modèle"),
            tooltip=[alt.Tooltip(c) for c in tooltip_cols],
        )
        .properties(title="Qualité vs Latence (bulles ∝ FLOPs)", height=400)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def _build_zip_bytes(output_dir: Path) -> bytes:
    """Build a ZIP archive (in memory) of all mask MP4s + CSV/JSON reports."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        masks_root = output_dir / "masks"
        if masks_root.is_dir():
            for f in masks_root.rglob("*.mp4"):
                zf.write(f, f.relative_to(output_dir))
        for fname in ("benchmark_results.csv", "benchmark_results.json"):
            p = output_dir / fname
            if p.exists():
                zf.write(p, fname)
    buf.seek(0)
    return buf.read()


# ── Auto-refresh fragment for tab2 background computation ────────────────────
@st.fragment(run_every=2)
def _t2_live_panel(count_pairs: int):
    """Drains the result queue every 2 s and refreshes the display."""
    q = st.session_state.get("t2_queue")
    if q is None:
        return

    # Drain everything available right now
    while True:
        try:
            msg_type, data = q.get_nowait()
            if msg_type == "result":
                st.session_state.setdefault("t2_results", []).append(data)
            elif msg_type == "status":
                st.session_state["t2_lat_status"] = data
            elif msg_type == "done":
                st.session_state["t2_running"] = False
        except Exception:
            break

    results = st.session_state.get("t2_results", [])
    running = st.session_state.get("t2_running", False)

    if not results and not running:
        return

    done_count = len(results)
    if count_pairs > 0:
        st.progress(done_count / count_pairs,
                    text=f"{done_count}/{count_pairs} paires traitées")

    lat_msg = st.session_state.get("t2_lat_status")
    if lat_msg:
        st.info(lat_msg)

    if results:
        df_live = pd.DataFrame(results)
        st.dataframe(_styled_df(df_live), width='stretch')

    if not running and results:
        df_done = pd.DataFrame(results)

        # Averages
        numeric_ok = df_done[df_done["status"] == "OK"] if "status" in df_done.columns else df_done
        if not numeric_ok.empty:
            avg = numeric_ok.groupby("model").mean(numeric_only=True).reset_index()
            st.subheader("📈 Moyennes par modèle")
            st.table(avg)
            st.subheader("⚡ Qualité vs Latence")
            _scatter_chart(numeric_ok)

        csv_bytes = df_done.to_csv(index=False).encode("utf-8")
        c1, c2 = st.columns(2)
        c1.download_button("📥 Télécharger CSV", data=csv_bytes,
                           file_name="benchmark_results.csv", mime="text/csv",
                           key="t2_dl_csv")
        with st.spinner("Création ZIP…"):
            zip_bytes = _build_zip_bytes(OUTPUT_DIR)
        c2.download_button("📦 Télécharger tout (ZIP)", data=zip_bytes,
                           file_name="benchmark_outputs.zip",
                           mime="application/zip", key="t2_dl_zip")


# ── Sidebar shared config ─────────────────────────────────────────────────────
st.sidebar.title("⚙️ Configuration")

# Dataset selection
selected_ds = st.sidebar.selectbox(
    "📂 Dataset source",
    options=list(DATASETS.keys()),
    index=0,
    help="Choisissez le dossier source des vidéos et du Ground Truth."
)
curr_dataset_root = DATASETS[selected_ds]
curr_videos_dir = curr_dataset_root / "videos"
curr_gt_dir = curr_dataset_root / "ground_truth"

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🚀 Benchmark complet",
    "📊 Métriques sur outputs",
    "🖼️ Comparateur visuel",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 1 — Full benchmark
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.title("🚀 Benchmark complet")

    # Sidebar for tab1
    with st.sidebar:
        st.subheader("📹 Vidéos")
        available_datasets = discover_datasets(curr_videos_dir, curr_gt_dir)
        total_videos = len(available_datasets)
        
        sel_mode = st.radio(
            "Mode de sélection",
            options=["Toutes", "Aléatoire", "Plage (Index)", "Manuelle"],
            horizontal=True,
            key="t1_sel_mode"
        )
        
        video_indices = None
        num_videos = 0
        use_shuffle = False

        if sel_mode == "Toutes":
            num_videos = 0
        elif sel_mode == "Aléatoire":
            num_videos = st.number_input("Nombre de vidéos", 1, total_videos, min(10, total_videos), key="t1_num_v")
            use_shuffle = True
        elif sel_mode == "Plage (Index)":
            idx_range = st.slider("Choisissez la plage", 0, total_videos - 1, (0, min(9, total_videos - 1)), key="t1_range")
            video_indices = list(range(idx_range[0], idx_range[1] + 1))
        elif sel_mode == "Manuelle":
            text_input = st.text_input("Indices (ex: 0, 5-10, 22)", key="t1_manual_text")
            video_indices = []
            if text_input:
                try:
                    parts = [p.strip() for p in text_input.split(",")]
                    for p in parts:
                        if "-" in p:
                            start, end = map(int, p.split("-"))
                            video_indices.extend(range(start, end + 1))
                        else:
                            video_indices.append(int(p))
                    # Filtrer les indices valides
                    video_indices = [i for i in video_indices if 0 <= i < total_videos]
                    # Retirer les doublons et trier
                    video_indices = sorted(list(set(video_indices)))
                except ValueError:
                    st.error("Format invalide. Utilisez des nombres, virgules ou tirets (ex: 0, 5-10).")

        st.divider()
        st.subheader("💾 Options export")
        save_masks    = st.checkbox("Sauvegarder PNG", value=False, key="t1_masks")
        save_video    = st.checkbox("Sauvegarder masques (.mp4)", value=False, key="t1_video")
        save_segmented = st.checkbox("Sauvegarder sujet (.mp4)", value=True, key="t1_seg")

    # ── Model selection ──
    col_models, col_run = st.columns([1, 2])

    with col_models:
        st.subheader("🤖 Sélection des modèles")
        c1, c2 = st.columns(2)
        if c1.button("✅ Tout", key="t1_all"):
            for k in MODEL_REGISTRY:
                st.session_state[f"t1_cb_{k}"] = True
            st.rerun()
        if c2.button("❌ Aucun", key="t1_none"):
            for k in MODEL_REGISTRY:
                st.session_state[f"t1_cb_{k}"] = False
            st.rerun()

        st.markdown("---")
        selected_models = []
        keys = list(MODEL_REGISTRY.keys())
        for i in range(0, len(keys), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(keys):
                    key = keys[i + j]
                    m_instance = MODEL_REGISTRY[key]()
                    with cols[j]:
                        with st.container(border=True):
                            checked = st.checkbox(
                                f"**{m_instance.name}**",
                                value=st.session_state.get(f"t1_cb_{key}", True),
                                key=f"t1_cb_{key}",
                            )
                            if checked:
                                selected_models.append(key)

    with col_run:
        st.subheader("ℹ️ Aperçu")
        n_seq = num_videos if num_videos > 0 else total_videos
        st.info(f"""
- **Vidéos détectées** : {total_videos}
- **Séquences à traiter** : {n_seq}
- **Combinaisons totales** : {len(selected_models) * n_seq}
- **Dossier sortie** : `{OUTPUT_DIR}`
        """)

        launch_btn = st.button("🚀 LANCER LE BENCHMARK", type="primary", key="t1_launch")

    with st.expander("📖 Guide des métriques", expanded=False):
        st.markdown(METRIC_GUIDE)

    # ── Execution ──
    if launch_btn:
        if not selected_models:
            st.error("Sélectionnez au moins un modèle.")
        else:
            results_so_far: list = []
            results_placeholder = st.empty()

            def _on_result_t1(entry: dict):
                results_so_far.append(entry)
                df_live = pd.DataFrame(results_so_far)
                with results_placeholder.container():
                    st.subheader(f"📊 Résultats en cours… ({len(results_so_far)} terminé(s))")
                    st.dataframe(_styled_df(df_live), width='stretch')

            models = [MODEL_REGISTRY[k]() for k in selected_models]

            with st.status("🛠️ Benchmark en cours…", expanded=True) as status_box:
                st.write("Initialisation des modèles…")
                t_start = time.time()
                results = run_benchmark(
                    models=models,
                    videos_dir=curr_videos_dir,
                    gt_dir=curr_gt_dir,
                    num_videos=num_videos,
                    random_selection=use_shuffle,
                    video_indices=video_indices,
                    save_masks=save_masks,
                    save_video=save_video,
                    save_segmented=save_segmented,
                    on_result=_on_result_t1,
                )
                elapsed = time.time() - t_start
                status_box.update(label=f"✅ Terminé en {elapsed:.1f}s", state="complete")

            st.success("Benchmark terminé !")

            if results:
                df = pd.DataFrame(results)
                st.subheader("📊 Résultats finaux")
                st.dataframe(_styled_df(df), width='stretch')

                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Télécharger CSV", data=csv_data,
                                   file_name="benchmark_results.csv", mime="text/csv")

                st.subheader("📈 Moyennes par modèle")
                avg = df.groupby("model").mean(numeric_only=True).reset_index()
                if not avg.empty:
                    st.table(avg)
                    if "iou_mean" in avg.columns:
                        st.bar_chart(avg, x="model", y="iou_mean")

                st.subheader("⚡ Qualité vs Latence")
                _scatter_chart(df)
            else:
                st.warning("Aucun résultat. Vérifiez le dataset.")

    else:
        res_path = OUTPUT_DIR / "benchmark_results.csv"
        if res_path.exists():
            st.subheader("Derniers résultats")
            st.dataframe(pd.read_csv(res_path).head(20), width='stretch')
        else:
            st.info("Configurez les paramètres puis cliquez sur Lancer.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 2 — Post-hoc metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.title("📊 Métriques sur outputs existants")
    st.markdown(
        "Calcule les métriques sur les masques **déjà générés** dans `output/masks/` "
        "— sans relancer l'inférence."
    )

    masks_root = OUTPUT_DIR / "masks"
    available_model_dirs = (
        sorted(d.name for d in masks_root.iterdir() if d.is_dir())
        if masks_root.is_dir() else []
    )

    if not available_model_dirs:
        st.warning(f"Aucun masque trouvé dans `{masks_root}`. "
                   "Lancez d'abord un benchmark avec l'option 'Sauvegarder sujet (.mp4)'.")
    else:
        # ── Sidebar options for tab2 ──
        with st.sidebar:
            st.divider()
            st.subheader("📊 Métriques — options")
            threshold = st.slider(
                "Seuil de binarisation",
                min_value=0.10, max_value=0.90, value=0.50, step=0.05,
                help="Seuil appliqué à tous les masques pour les binariser (foreground ≥ seuil).",
                key="t2_threshold",
            )

        selected_dirs = st.multiselect(
            "Modèles à évaluer",
            options=available_model_dirs,
            default=available_model_dirs,
            key="t2_models",
        )

        count_pairs = sum(
            len(list((masks_root / d).glob("*_mask.mp4")))
            for d in selected_dirs
        )
        st.info(f"**{count_pairs}** paires (modèle × vidéo) détectées | "
                f"Seuil binarisation : **{threshold}**")

        with st.expander("📖 Guide des métriques", expanded=False):
            st.markdown(METRIC_GUIDE)

        compute_btn = st.button("📊 Calculer les métriques", type="primary", key="t2_compute")

        if compute_btn and selected_dirs:
            # Reset state and start background thread
            bg_queue: queue.Queue = queue.Queue()
            st.session_state["t2_queue"] = bg_queue
            st.session_state["t2_results"] = []
            st.session_state["t2_running"] = True
            st.session_state["t2_lat_status"] = None

            _dirs = list(selected_dirs)
            _threshold = threshold

            def _bg_worker():
                def _on_result(entry: dict):
                    bg_queue.put(("result", entry))

                def _on_lat_status(msg: str):
                    bg_queue.put(("status", msg))

                try:
                    compute_metrics_on_output(
                        output_dir=OUTPUT_DIR,
                        gt_dir=curr_gt_dir,
                        videos_dir=curr_videos_dir,
                        model_filter=_dirs,
                        threshold=_threshold,
                        measure_missing_latency=True,
                        on_result=_on_result,
                        on_latency_status=_on_lat_status,
                    )
                finally:
                    bg_queue.put(("done", None))

            threading.Thread(target=_bg_worker, daemon=True).start()
            st.rerun()

        elif compute_btn and not selected_dirs:
            st.error("Sélectionnez au moins un modèle.")

        # Auto-refreshing live panel (runs every 2 s via @st.fragment)
        if st.session_state.get("t2_running") or st.session_state.get("t2_results"):
            _t2_live_panel(count_pairs)
        elif (OUTPUT_DIR / "benchmark_results.csv").exists():
            st.subheader("Derniers résultats enregistrés")
            st.dataframe(
                pd.read_csv(OUTPUT_DIR / "benchmark_results.csv"),
                width='stretch',
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 3 — Visual comparator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.title("🖼️ Comparateur visuel")
    st.markdown(
        "Visualisez côte à côte la **frame source**, le **masque prédit** "
        "et la **vérité terrain** sans relancer l'inférence."
    )

    # Build index of available (model, video_stem) pairs
    # Accepts both *_mask.mp4 (save_video=True) and *_segmented.mp4 (save_segmented=True)
    masks_root3 = OUTPUT_DIR / "masks"
    pairs_index: dict = {}  # model_dir_name → list[video_stem]
    if masks_root3.is_dir():
        for md in sorted(masks_root3.iterdir()):
            if not md.is_dir():
                continue
            mask_stems = sorted(
                f.stem.replace(f"_{md.name}_mask", "")
                for f in md.glob("*_mask.mp4")
            )
            if mask_stems:
                pairs_index[md.name] = mask_stems
                continue
            # Fallback: derive stems from segmented videos
            seg_stems = sorted(
                f.stem.replace(f"_{md.name}_segmented", "")
                for f in md.glob("*_segmented.mp4")
            )
            if seg_stems:
                pairs_index[md.name] = seg_stems

    if not pairs_index:
        st.warning("Aucun masque trouvé. Lancez d'abord un benchmark.")
    else:
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            chosen_model = st.selectbox(
                "Modèle",
                options=list(pairs_index.keys()),
                key="t3_model",
            )
        with col_sel2:
            chosen_video = st.selectbox(
                "Vidéo",
                options=pairs_index.get(chosen_model, []),
                key="t3_video",
            )

        if chosen_model and chosen_video:
            src_path  = curr_videos_dir / f"{chosen_video}.mp4"
            gt_path   = curr_gt_dir / f"{chosen_video}.mp4"
            mask_path = masks_root3 / chosen_model / f"{chosen_video}_{chosen_model}_mask.mp4"
            seg_path  = masks_root3 / chosen_model / f"{chosen_video}_{chosen_model}_segmented.mp4"

            # Reference video for frame count
            ref_path = mask_path if mask_path.exists() else (seg_path if seg_path.exists() else src_path)
            n_frames_vis = 1
            if ref_path.exists():
                n_frames_vis, _, _ = _get_video_info(ref_path)

            frame_idx = st.slider(
                "Numéro de frame",
                min_value=0,
                max_value=max(0, n_frames_vis - 1),
                value=0,
                key="t3_frame",
            )

            # Load source and GT
            img_src    = get_frame_at(src_path, frame_idx) if src_path.exists() else None
            img_gt_raw = get_frame_at(gt_path,  frame_idx) if gt_path.exists()  else None

            # Load predicted mask: prefer *_mask.mp4, fallback to *_segmented.mp4
            img_mask = None
            mask_source = ""
            if mask_path.exists():
                img_mask = get_frame_at(mask_path, frame_idx)
                mask_source = "mask"
            elif seg_path.exists():
                seg_frame = get_frame_at(seg_path, frame_idx)
                if seg_frame is not None:
                    # Derive mask: non-black pixels → foreground
                    seg_rgb = cv2.cvtColor(seg_frame, cv2.COLOR_BGR2RGB)
                    derived = (np.any(seg_rgb > 10, axis=2).astype(np.uint8) * 255)
                    img_mask = cv2.cvtColor(derived, cv2.COLOR_GRAY2BGR)
                    mask_source = "segmented"

            # Convert GT to mask
            img_gt_mask = None
            if img_gt_raw is not None:
                if _is_chroma_key(img_gt_raw):
                    gt_bin = _chroma_key_to_mask(img_gt_raw)
                    img_gt_mask = (gt_bin * 255).astype(np.uint8)
                else:
                    img_gt_mask = cv2.cvtColor(img_gt_raw, cv2.COLOR_BGR2GRAY)

            # ── Row 1: 3 columns ─────────────────────────────────────────────
            c_src, c_pred, c_gt = st.columns(3)

            with c_src:
                st.markdown("**Frame source**")
                if img_src is not None:
                    st.image(cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB),
                             width='stretch')
                else:
                    st.warning("Vidéo source introuvable.")

            with c_pred:
                label = "**Masque prédit**" + (" *(dérivé du segmenté)*" if mask_source == "segmented" else "")
                st.markdown(label)
                if img_mask is not None:
                    gray_pred = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
                    st.image(gray_pred, width='stretch', clamp=True)
                else:
                    st.warning("Masque prédit introuvable.")

            with c_gt:
                st.markdown("**GT binarisé**")
                if img_gt_mask is not None:
                    st.image(img_gt_mask, width='stretch', clamp=True)
                else:
                    st.warning("GT introuvable.")

            # ── Row 2: TP/FP/FN overlay ──────────────────────────────────────
            if img_src is not None and img_mask is not None and img_gt_mask is not None:
                st.markdown("---")

                thresh_vis = float(st.session_state.get("t2_threshold", 0.5))

                # Resize pred mask to source frame size if needed
                h_src, w_src = img_src.shape[:2]
                gray_pred_ov = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
                if gray_pred_ov.shape != (h_src, w_src):
                    gray_pred_ov = cv2.resize(gray_pred_ov, (w_src, h_src),
                                              interpolation=cv2.INTER_NEAREST)

                gt_ov = img_gt_mask
                if gt_ov.shape != (h_src, w_src):
                    gt_ov = cv2.resize(gt_ov, (w_src, h_src),
                                       interpolation=cv2.INTER_NEAREST)

                bin_pred = gray_pred_ov > int(thresh_vis * 255)
                bin_gt   = gt_ov > int(thresh_vis * 255)

                tp = bin_pred & bin_gt    # green  (0, 200, 0)
                fp = bin_pred & ~bin_gt   # red    (220, 50, 50)
                fn = ~bin_pred & bin_gt   # blue   (50, 50, 220)

                alpha = 0.45
                overlay_f = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB).astype(np.float32)
                overlay_f[tp] = overlay_f[tp] * (1 - alpha) + np.array([0, 200, 0],   np.float32) * alpha
                overlay_f[fp] = overlay_f[fp] * (1 - alpha) + np.array([220, 50, 50], np.float32) * alpha
                overlay_f[fn] = overlay_f[fn] * (1 - alpha) + np.array([50, 50, 220], np.float32) * alpha
                overlay_img = overlay_f.clip(0, 255).astype(np.uint8)

                # Per-frame metrics
                from benchmark.metrics import compute_iou, compute_boundary_f_measure
                pred_f32 = gray_pred_ov.astype(np.float32) / 255.0
                gt_f32   = gt_ov.astype(np.float32) / 255.0
                frame_iou = compute_iou([pred_f32], [gt_f32], threshold=thresh_vis)
                frame_bf  = compute_boundary_f_measure(pred_f32, gt_f32, threshold=thresh_vis)

                col_ov, col_leg = st.columns([3, 1])
                with col_ov:
                    st.markdown("**Overlay TP / FP / FN**")
                    st.image(overlay_img, width='stretch')
                with col_leg:
                    st.markdown("**Légende**")
                    st.markdown("""
- 🟢 **Vert** — Vrai positif (TP)
  *(pred = personne, GT = personne)*
- 🔴 **Rouge** — Faux positif (FP)
  *(pred = personne, GT = fond)*
- 🔵 **Bleu** — Faux négatif (FN)
  *(pred = fond, GT = personne)*
- *(fond inchangé = vrai négatif)*
""")
                    st.markdown("**Métriques de cette frame**")
                    st.metric("IoU", f"{frame_iou:.3f}")
                    st.metric("Boundary F", f"{frame_bf:.3f}")
                    n_tp = int(tp.sum())
                    n_fp = int(fp.sum())
                    n_fn = int(fn.sum())
                    st.caption(f"TP={n_tp:,}  FP={n_fp:,}  FN={n_fn:,} px")
