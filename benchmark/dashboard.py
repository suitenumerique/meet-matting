"""
Tableau de bord Streamlit pour le Benchmark Video Matting.

Offre une interface moderne pour :
  - Sélectionner les modèles.
  - Choisir le nombre de vidéos (aléatoire ou non).
  - Activer la sauvegarde des masques.
  - Visualiser les résultats en temps réel.
"""

import io
import json
import os
<<<<<<< HEAD
=======
from typing import List
# Désactivation TOTALE des logs MediaPipe/GLog au démarrage
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
>>>>>>> 674b3ef (post-processing visuel)

# Désactivation TOTALE des logs MediaPipe/GLog au démarrage
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import queue
import sys
import threading
import time
import warnings
import zipfile
from pathlib import Path
import logging

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

from benchmark.config import DATASETS, OUTPUT_DIR
from benchmark.models import MODEL_REGISTRY
from benchmark.runner import (
    _chroma_key_to_mask,
    _get_video_info,
    _is_chroma_key,
    compute_metrics_on_output,
    discover_datasets,
    get_frame_at,
    run_benchmark,
)
from benchmark.config import DATASETS, GROUND_TRUTH_DIR, OUTPUT_DIR, TEMP_RESULTS_DIR, VIDEOS_DIR
from benchmark.postprocess import (
    METHODS_BY_MODEL,
    SUPPORTED_MODELS as PP_SUPPORTED_MODELS,
    PostProcessor,
    load_config as pp_load_config,
    save_config as pp_save_config,
    _default_config as pp_default_config,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Video Matting Benchmark",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style Custom
st.markdown(
    """
<style>
  .stButton>button { width:100%; border-radius:5px; height:3em;
                     background-color:#007bff; color:white; font-weight:bold; }
  .metric-card { background:#fff; padding:15px; border-radius:10px;
                 box-shadow:0 2px 4px rgba(0,0,0,.05); }
</style>
""",
    unsafe_allow_html=True,
)

# ── Shared constants ──────────────────────────────────────────────────────────
DISPLAY_COLS = [
    "model",
    "video",
    "status",
    "iou_mean",
    "boundary_f_mean",
    "flow_warping_error",
    "latency_p95_ms",
    "flops_per_frame",
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


def _threshold_sensitivity_chart(results: list):
    """Display IoU and Boundary F curves vs binarization threshold per model."""
    try:
        import altair as alt
    except ImportError:
        st.info("Altair non disponible.")
        return

    rows = []
    for r in results:
        ta = r.get("threshold_analysis")
        if not ta:
            continue
        for t_str, metrics in ta.items():
            rows.append(
                {
                    "model": r.get("model", ""),
                    "threshold": float(t_str),
                    "IoU": metrics.get("iou_mean", 0),
                    "Boundary F": metrics.get("boundary_f_mean", 0),
                }
            )

    if not rows:
        st.info("Aucune analyse de seuil disponible.")
        return

    df_th = pd.DataFrame(rows).groupby(["model", "threshold"]).mean(numeric_only=True).reset_index()

    # Optimal threshold table
    idx_best_iou = df_th.groupby("model")["IoU"].idxmax()
    best = df_th.loc[idx_best_iou, ["model", "threshold", "IoU", "Boundary F"]].copy()
    best.columns = ["Modèle", "Seuil optimal (IoU)", "IoU max", "Boundary F au seuil optimal"]
    st.subheader("🎯 Seuil optimal par modèle")
    st.table(best.reset_index(drop=True))

    base = alt.Chart(df_th)
    chart_iou = (
        base.mark_line(point=True)
        .encode(
            x=alt.X(
                "threshold:Q", title="Seuil de binarisation", scale=alt.Scale(domain=[0.1, 0.9])
            ),
            y=alt.Y("IoU:Q", title="IoU moyen", scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", title="Modèle"),
            tooltip=["model", "threshold", "IoU", "Boundary F"],
        )
        .properties(title="IoU vs Seuil", height=280)
    )
    chart_bf = (
        base.mark_line(point=True)
        .encode(
            x=alt.X(
                "threshold:Q", title="Seuil de binarisation", scale=alt.Scale(domain=[0.1, 0.9])
            ),
            y=alt.Y("Boundary F:Q", title="Boundary F moyen", scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", title="Modèle"),
            tooltip=["model", "threshold", "IoU", "Boundary F"],
        )
        .properties(title="Boundary F vs Seuil", height=280)
    )
    st.altair_chart(chart_iou | chart_bf, use_container_width=True)


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


# ── Auto-refresh fragment for tab1 background benchmark ──────────────────────
@st.fragment(run_every=2)
def _t1_live_panel(total_combos: int):
    """Drains the benchmark result queue every 2 s and refreshes the display."""
    q = st.session_state.get("t1_queue")
    if q is None:
        return

    while True:
        try:
            msg_type, data = q.get_nowait()
            if msg_type == "result":
                st.session_state.setdefault("t1_results", []).append(data)
            elif msg_type == "done":
                st.session_state["t1_running"] = False
                st.session_state["t1_elapsed"] = data
        except Exception:
            break

    results = st.session_state.get("t1_results", [])
    running = st.session_state.get("t1_running", False)

    if not results and not running:
        return

    done_count = len(results)
    if total_combos > 0:
        st.progress(
            done_count / total_combos, text=f"{done_count}/{total_combos} combinaisons traitées"
        )

    if results:
        df_live = pd.DataFrame(results)
        st.subheader(f"📊 Résultats en cours… ({done_count} terminé(s))")
        st.dataframe(_styled_df(df_live), width="stretch")

    if not running and results:
        elapsed = st.session_state.get("t1_elapsed", 0)
        st.success(f"✅ Benchmark terminé en {elapsed:.1f}s !")
        df_done = pd.DataFrame(results)

        st.subheader("📈 Moyennes par modèle")
        avg = df_done.groupby("model").mean(numeric_only=True).reset_index()
        if not avg.empty:
            st.table(avg)
            if "iou_mean" in avg.columns:
                st.bar_chart(avg, x="model", y="iou_mean")

        st.subheader("⚡ Qualité vs Latence")
        _scatter_chart(df_done)

        if any(r.get("threshold_analysis") for r in results):
            st.subheader("🔬 Sensibilité au seuil de binarisation")
            st.markdown(
                "Évolution de l'IoU et du Boundary F en fonction du seuil — "
                "utile pour identifier le seuil optimal par modèle."
            )
            _threshold_sensitivity_chart(results)

        csv_data = df_done.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Télécharger CSV",
            data=csv_data,
            file_name="benchmark_results.csv",
            mime="text/csv",
            key="t1_dl_csv",
        )


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
        st.progress(done_count / count_pairs, text=f"{done_count}/{count_pairs} paires traitées")

    lat_msg = st.session_state.get("t2_lat_status")
    if lat_msg:
        st.info(lat_msg)

    if results:
        df_live = pd.DataFrame(results)
        st.dataframe(_styled_df(df_live), width="stretch")

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
        c1.download_button(
            "📥 Télécharger CSV",
            data=csv_bytes,
            file_name="benchmark_results.csv",
            mime="text/csv",
            key="t2_dl_csv",
        )
        with st.spinner("Création ZIP…"):
            zip_bytes = _build_zip_bytes(OUTPUT_DIR)
        c2.download_button(
            "📦 Télécharger tout (ZIP)",
            data=zip_bytes,
            file_name="benchmark_outputs.zip",
            mime="application/zip",
            key="t2_dl_zip",
        )


# ── Sidebar shared config ─────────────────────────────────────────────────────
st.sidebar.title("⚙️ Configuration")

# Dataset selection
selected_ds = st.sidebar.selectbox(
    "📂 Dataset source",
    options=list(DATASETS.keys()),
    index=0,
    help="Choisissez le dossier source des vidéos et du Ground Truth.",
)
curr_dataset_root = DATASETS[selected_ds]
curr_videos_dir = curr_dataset_root / "videos"
curr_gt_dir = curr_dataset_root / "ground_truth"

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚀 Benchmark complet",
    "📊 Métriques sur outputs",
    "🖼️ Comparateur visuel",
    "⚙️ Post-processing",
    "🗄️ Données",
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
            key="t1_sel_mode",
        )

        video_indices = None
        num_videos = 0
        use_shuffle = False

        if sel_mode == "Toutes":
            num_videos = 0
        elif sel_mode == "Aléatoire":
            num_videos = st.number_input(
                "Nombre de vidéos", 1, total_videos, min(10, total_videos), key="t1_num_v"
            )
            use_shuffle = True
        elif sel_mode == "Plage (Index)":
            idx_range = st.slider(
                "Choisissez la plage",
                0,
                total_videos - 1,
                (0, min(9, total_videos - 1)),
                key="t1_range",
            )
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
                    st.error(
                        "Format invalide. Utilisez des nombres, virgules ou tirets (ex: 0, 5-10)."
                    )

        st.divider()
        st.subheader("💾 Options export")
        save_masks = st.checkbox("Sauvegarder PNG", value=False, key="t1_masks")
        save_video = st.checkbox("Sauvegarder masques (.mp4)", value=False, key="t1_video")
        save_segmented = st.checkbox("Sauvegarder sujet (.mp4)", value=True, key="t1_seg")

        st.divider()
        st.subheader("🔬 Analyse avancée")
        analyze_thresholds = st.checkbox(
            "Analyser tous les seuils (0.1 → 0.9)",
            value=False,
            key="t1_thresh_analysis",
            help="Calcule IoU et Boundary F pour 9 seuils de binarisation. "
            "Augmente la durée du benchmark (~×2 pour les métriques).",
        )

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
        if video_indices:
            n_seq = len(video_indices)
        elif num_videos > 0:
            n_seq = num_videos
        else:
            n_seq = total_videos
        st.info(f"""
- **Vidéos détectées** : {total_videos}
- **Séquences à traiter** : {n_seq}
- **Combinaisons totales** : {len(selected_models) * n_seq}
- **Dossier sortie** : `{OUTPUT_DIR}`
        """)

        # Indicateur post-processing actif
        _PP_DISPLAY = {
            "mediapipe_portrait": "MediaPipe Portrait",
            "mobilenetv3_lraspp": "MobileNetV3 + LRASPP",
            "rvm": "RVM (MobileNetV3)",
        }
        active_pp = []
        for _k in selected_models:
            if _k in PP_SUPPORTED_MODELS:
                _cfg = pp_load_config(_k)
                _active = [m["name"] for m in _cfg.get("methods", []) if m.get("enabled")]
                if _active:
                    active_pp.append(f"**{_PP_DISPLAY.get(_k, _k)}** : {', '.join(_active)}")
        if active_pp:
            st.success("✅ Post-processing actif :\n" + "\n".join(f"- {x}" for x in active_pp))

        launch_btn = st.button("🚀 LANCER LE BENCHMARK", type="primary", key="t1_launch")

    with st.expander("📖 Guide des métriques", expanded=False):
        st.markdown(METRIC_GUIDE)

    # ── Execution ──
    if launch_btn:
        if not selected_models:
            st.error("Sélectionnez au moins un modèle.")
        else:
            bg_queue: queue.Queue = queue.Queue()
            st.session_state["t1_queue"] = bg_queue
            st.session_state["t1_results"] = []
            st.session_state["t1_running"] = True
            st.session_state["t1_elapsed"] = 0

            # Capture all config before the thread starts (avoid closure issues)
            _models = [MODEL_REGISTRY[k]() for k in selected_models]
            _videos_dir = curr_videos_dir
            _gt_dir = curr_gt_dir
            _num_videos = num_videos
            _shuffle = use_shuffle
            _v_indices = video_indices
            _save_masks = save_masks
            _save_video = save_video
            _save_segmented = save_segmented
            _analyze       = analyze_thresholds

            # Charger les configs post-processing pour les modèles supportés sélectionnés
            _postprocess_configs = {}
            for _k in selected_models:
                if _k in PP_SUPPORTED_MODELS:
                    _cfg = pp_load_config(_k)
                    if any(m.get("enabled") for m in _cfg.get("methods", [])):
                        _postprocess_configs[_k] = _cfg

            def _bg_benchmark():
                def _on_result(entry: dict):
                    bg_queue.put(("result", entry))

                t_start = time.time()
                try:
                    run_benchmark(
                        models=_models,
                        videos_dir=_videos_dir,
                        gt_dir=_gt_dir,
                        num_videos=_num_videos,
                        random_selection=_shuffle,
                        video_indices=_v_indices,
                        save_masks=_save_masks,
                        save_video=_save_video,
                        save_segmented=_save_segmented,
                        on_result=_on_result,
                        analyze_thresholds=_analyze,
                        postprocess_configs=_postprocess_configs or None,
                    )
                finally:
                    bg_queue.put(("done", time.time() - t_start))

            threading.Thread(target=_bg_benchmark, daemon=True).start()
            st.rerun()

    # Live panel — shows progress even when tab is not active
    if video_indices:
        n_seq = len(video_indices)
    elif num_videos > 0:
        n_seq = num_videos
    else:
        n_seq = total_videos
    total_combos = len(selected_models) * n_seq
    if st.session_state.get("t1_running") or st.session_state.get("t1_results"):
        _t1_live_panel(total_combos)
    elif not launch_btn:
        res_path = OUTPUT_DIR / "benchmark_results.csv"
        if res_path.exists():
            st.subheader("Derniers résultats")
            st.dataframe(pd.read_csv(res_path).head(20), width="stretch")
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
        sorted(d.name for d in masks_root.iterdir() if d.is_dir()) if masks_root.is_dir() else []
    )

    if not available_model_dirs:
        st.warning(
            f"Aucun masque trouvé dans `{masks_root}`. "
            "Lancez d'abord un benchmark avec l'option 'Sauvegarder sujet (.mp4)'."
        )
    else:
        # ── Sidebar options for tab2 ──
        with st.sidebar:
            st.divider()
            st.subheader("📊 Métriques — options")
            threshold = st.slider(
                "Seuil de binarisation",
                min_value=0.10,
                max_value=0.90,
                value=0.50,
                step=0.05,
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
            len(list((masks_root / d).glob("*_mask.mp4"))) +
            len(list((masks_root / d).glob("run_*/*_mask.mp4")))
            for d in selected_dirs
        )
        st.info(f"**{count_pairs}** paires (modèle × vidéo) détectées | "
                f"Seuil binarisation : **{threshold}**")

        with st.expander("📖 Guide des métriques", expanded=False):
            st.markdown(METRIC_GUIDE)

        compute_btn = st.button("📊 Calculer les métriques", type="primary", key="t2_compute")

        if compute_btn and selected_dirs:
            # Reset state and start background thread
            bg_queue = queue.Queue()
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
                width="stretch",
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

    # ── Mode photo / vidéo / comparaison ─────────────────────────────────────
    view_mode = st.radio(
        "Mode d'affichage",
        options=["📷 Mode photo", "🎬 Mode vidéo", "🔄 Mode comparaison"],
        horizontal=True,
        key="t3_view_mode",
    )

    # ── Build index of available (model_key, video_stem) pairs ────────────────
    # model_key format :
    #   "MediaPipe_Portrait"          → flat (old structure)
    #   "MediaPipe_Portrait/run_001"  → numéroté (new structure for supported models)
    masks_root3 = OUTPUT_DIR / "masks"
    # pairs_index: model_key → {"stems": [str], "base": Path, "model_dir": str}
    pairs_index: dict = {}

    def _stems_from_dir(base_dir: Path, model_dir_name: str):
        stems = sorted(
            f.stem.replace(f"_{model_dir_name}_mask", "")
            for f in base_dir.glob("*_mask.mp4")
        )
        if not stems:
            stems = sorted(
                f.stem.replace(f"_{model_dir_name}_segmented", "")
                for f in base_dir.glob("*_segmented.mp4")
            )
        return stems

    if masks_root3.is_dir():
        for md in sorted(masks_root3.iterdir()):
            if not md.is_dir():
                continue
            # Check for run_* subdirs (supported models)
            run_dirs = sorted(d for d in md.iterdir() if d.is_dir() and d.name.startswith("run_"))
            if run_dirs:
                for rd in run_dirs:
                    stems = _stems_from_dir(rd, md.name)
                    if stems:
                        key = f"{md.name}/{rd.name}"
                        pairs_index[key] = {"stems": stems, "base": rd, "model_dir": md.name}
            else:
                # Flat structure (non-supported or old benchmark)
                stems = _stems_from_dir(md, md.name)
                if stems:
                    pairs_index[md.name] = {"stems": stems, "base": md, "model_dir": md.name}

    # Index par modèle de base (pour mode comparaison)
    # base_model_index: model_dir_name → list of (label, entry_dict)
    base_model_index: dict = {}
    for _k, _v in pairs_index.items():
        _base = _v["model_dir"]
        base_model_index.setdefault(_base, []).append((_k, _v))

    if not pairs_index:
        st.warning("Aucun masque trouvé. Lancez d'abord un benchmark.")

    # ══════════════════════════════════════════════════════════════════════════
    #  MODE COMPARAISON — plusieurs runs du même modèle côte à côte
    # ══════════════════════════════════════════════════════════════════════════
    elif view_mode == "🔄 Mode comparaison":
        st.markdown("Comparez les **différentes versions post-processées** d'un même modèle côte à côte.")

        # Sélection du modèle de base et de la vidéo
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            comp_base_model = st.selectbox(
                "Modèle de base",
                options=list(base_model_index.keys()),
                format_func=lambda x: x.replace("_", " "),
                key="t3_comp_model",
            )
        # Trouver les vidéos communes à tous les runs de ce modèle
        _all_runs = base_model_index.get(comp_base_model, [])
        _common_stems: List[str] = []
        if _all_runs:
            _stem_sets = [set(entry["stems"]) for _, entry in _all_runs]
            _common_stems = sorted(set.union(*_stem_sets))
        with comp_col2:
            comp_video = st.selectbox(
                "Vidéo",
                options=_common_stems,
                key="t3_comp_video",
            )

        if comp_base_model and comp_video and _all_runs:
            # Slider frame partagé
            _ref_entry = _all_runs[0][1]
            _ref_mask = _ref_entry["base"] / f"{comp_video}_{_ref_entry['model_dir']}_mask.mp4"
            _ref_seg  = _ref_entry["base"] / f"{comp_video}_{_ref_entry['model_dir']}_segmented.mp4"
            _ref_path = _ref_mask if _ref_mask.exists() else (_ref_seg if _ref_seg.exists() else None)
            _n_frames_comp = 1
            if _ref_path and _ref_path.exists():
                _n_frames_comp, _, _ = _get_video_info(_ref_path)

            comp_frame_idx = st.slider(
                "Numéro de frame (partagé)",
                min_value=0, max_value=max(0, _n_frames_comp - 1),
                value=0, key="t3_comp_frame",
            )

            src_path_comp = curr_videos_dir / f"{comp_video}.mp4"
            src_frame = get_frame_at(src_path_comp, comp_frame_idx) if src_path_comp.exists() else None

            # Afficher la frame source
            st.markdown("---")
            if src_frame is not None:
                st.markdown("**Frame source**")
                st.image(cv2.cvtColor(src_frame, cv2.COLOR_BGR2RGB), width=300)

            st.markdown("---")
            st.markdown(f"**Comparaison de {len(_all_runs)} version(s) — `{comp_base_model.replace('_',' ')}`**")

            # Colonnes : une par run
            _ncols = len(_all_runs)
            _comp_cols = st.columns(_ncols)
            for col_idx, (run_key, run_entry) in enumerate(_all_runs):
                _run_base = run_entry["base"]
                _run_mdir = run_entry["model_dir"]
                _mask_p = _run_base / f"{comp_video}_{_run_mdir}_mask.mp4"
                _seg_p  = _run_base / f"{comp_video}_{_run_mdir}_segmented.mp4"

                # Libellé du run
                if "/" in run_key:
                    _run_label = run_key.split("/")[1]
                else:
                    _run_label = "Sans run (brut)"

                with _comp_cols[col_idx]:
                    st.markdown(f"**{_run_label}**")

                    # Config post-process
                    _rc_path = _run_base / "run_config.json"
                    _legacy_pp = _run_base / "postprocess_config.json"
                    _pp_methods = []
                    if _rc_path.exists():
                        try:
                            _rc = json.loads(_rc_path.read_text())
                            _pp_cfg = _rc.get("postprocess_config") or {}
                            _pp_methods = [m["name"] for m in _pp_cfg.get("methods", []) if m.get("enabled")]
                        except Exception:
                            pass
                    elif _legacy_pp.exists():
                        try:
                            _pp_cfg = json.loads(_legacy_pp.read_text())
                            _pp_methods = [m["name"] for m in _pp_cfg.get("methods", []) if m.get("enabled")]
                        except Exception:
                            pass
                    st.caption("🔧 " + (", ".join(_pp_methods) if _pp_methods else "Aucun post-process"))

                    # Masque
                    _mask_frame = None
                    if _mask_p.exists():
                        _mask_frame = get_frame_at(_mask_p, comp_frame_idx)
                    elif _seg_p.exists():
                        _sf = get_frame_at(_seg_p, comp_frame_idx)
                        if _sf is not None:
                            _sr = cv2.cvtColor(_sf, cv2.COLOR_BGR2RGB)
                            _mask_frame = cv2.cvtColor(
                                (np.any(_sr > 10, axis=2).astype(np.uint8) * 255),
                                cv2.COLOR_GRAY2BGR,
                            )
                    if _mask_frame is not None:
                        _gray = cv2.cvtColor(_mask_frame, cv2.COLOR_BGR2GRAY) if _mask_frame.ndim == 3 else _mask_frame
                        st.image(_gray, clamp=True, use_container_width=True)
                    else:
                        st.warning("Masque introuvable")

                    # Vidéo masque
                    if _mask_p.exists():
                        with st.expander("🎬 Voir vidéo"):
                            st.video(open(_mask_p, "rb"))

    else:
        # Modes photo et vidéo — sélecteur classique
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            chosen_model_key = st.selectbox(
                "Modèle / Run",
                options=list(pairs_index.keys()),
                key="t3_model",
            )
        with col_sel2:
            chosen_video = st.selectbox(
                "Vidéo",
                options=pairs_index.get(chosen_model_key, {}).get("stems", []),
                key="t3_video",
            )

        if chosen_model_key and chosen_video:
            _t3_entry    = pairs_index[chosen_model_key]
            _t3_base     = _t3_entry["base"]
            _t3_mdir     = _t3_entry["model_dir"]

            src_path  = curr_videos_dir / f"{chosen_video}.mp4"
            gt_path   = curr_gt_dir / f"{chosen_video}.mp4"
            mask_path = _t3_base / f"{chosen_video}_{_t3_mdir}_mask.mp4"
            seg_path  = _t3_base / f"{chosen_video}_{_t3_mdir}_segmented.mp4"
            dbg_inter = _t3_base / f"{chosen_video}_DEBUG_intersection.mp4"
            # DEBUG union video is saved as {dest_base.name}_DEBUG_union.mp4 in the run dir
            # dest_base is run_dir / video_path.stem, so DEBUG files are in run_dir
            dbg_union = _t3_base / f"{chosen_video}_DEBUG_union.mp4"

            # Afficher les infos du run si disponibles
            _rc_path_t3 = _t3_base / "run_config.json"
            _legacy_path_t3 = _t3_base / "postprocess_config.json"
            if _rc_path_t3.exists():
                with st.expander("ℹ️ Infos du run", expanded=False):
                    try:
                        _rc_data = json.loads(_rc_path_t3.read_text())
                        _c1, _c2 = st.columns(2)
                        with _c1:
                            st.markdown(f"**Run #{_rc_data.get('run_id', '?')}** — {_rc_data.get('timestamp', '')[:19]}")
                            _settings = _rc_data.get("settings", {})
                            st.caption(" · ".join(k for k, v in _settings.items() if v))
                        with _c2:
                            _pp_c = _rc_data.get("postprocess_config") or {}
                            _active_pp = [m["name"] for m in _pp_c.get("methods", []) if m.get("enabled")]
                            st.markdown("**Post-process :** " + (", ".join(f"`{m}`" for m in _active_pp) if _active_pp else "aucun"))
                            _summ = _rc_data.get("results_summary", {})
                            if _summ:
                                st.caption(f"IoU {_summ.get('iou_mean','?')} · BF {_summ.get('boundary_f_mean','?')} · {_summ.get('successful','?')}/{_summ.get('total_videos','?')} vidéos OK")
                    except Exception:
                        st.warning("Config illisible.")
            elif _legacy_path_t3.exists():
                with st.expander("🔧 Config post-processing (ancien format)", expanded=False):
                    try:
                        st.json(json.loads(_legacy_path_t3.read_text()))
                    except Exception:
                        st.warning("Config illisible.")

            # ══════════════════════════════════════════════════════════════════
            #  MODE VIDÉO
            # ══════════════════════════════════════════════════════════════════
            if view_mode == "🎬 Mode vidéo":
                st.markdown("---")
                st.subheader("🎬 Lecture vidéo")

                c_v1, c_v2, c_v3 = st.columns(3)
                with c_v1:
                    st.markdown("**Vidéo source**")
                    if src_path.exists():
                        st.video(open(src_path, "rb"))
                    else:
                        st.warning("Vidéo source introuvable.")

                with c_v2:
                    st.markdown("**Masque prédit**")
                    if mask_path.exists():
                        st.video(open(mask_path, "rb"))
                    else:
                        st.warning("Masque introuvable.")

                with c_v3:
                    st.markdown("**GT (vidéo)**")
                    if gt_path.exists():
                        st.video(open(gt_path, "rb"))
                    else:
                        st.info("GT vidéo non disponible.")

                st.markdown("---")
                c_v4, c_v5, c_v6 = st.columns(3)
                with c_v4:
                    st.markdown("**Sujet détouré**")
                    if seg_path.exists():
                        st.video(open(seg_path, "rb"))
                    else:
                        st.info("Vidéo détourée non disponible.")

                with c_v5:
                    st.markdown("**Debug — Intersection (TP)**")
                    if dbg_inter.exists():
                        st.video(open(dbg_inter, "rb"))
                    else:
                        st.info("Non disponible — lancez un benchmark avec GT.")

                with c_v6:
                    st.markdown("**Debug — Union (erreurs)**")
                    if dbg_union.exists():
                        st.video(open(dbg_union, "rb"))
                    else:
                        st.info("Non disponible — lancez un benchmark avec GT.")

                # Métriques globales depuis le CSV
                csv_path = OUTPUT_DIR / "benchmark_results.csv"
                if csv_path.exists():
                    df_csv = pd.read_csv(csv_path)
                    # model name from dir name
                    _model_display = _t3_mdir.replace("_", " ")
                    _video_file = f"{chosen_video}.mp4"
                    row = df_csv[
                        (df_csv.get("model", pd.Series(dtype=str)) == _model_display) &
                        (df_csv.get("video", pd.Series(dtype=str)) == _video_file)
                    ] if "model" in df_csv.columns else pd.DataFrame()
                    if not row.empty:
                        st.markdown("---")
                        st.subheader("📊 Métriques globales (depuis CSV)")
                        mcols = st.columns(4)
                        r = row.iloc[-1]
                        for col_w, key, label in zip(
                            mcols,
                            ["iou_mean", "boundary_f_mean", "flow_warping_error", "latency_p95_ms"],
                            ["IoU moyen", "Boundary F moyen", "Flow Warping Error", "Latence p95 (ms)"],
                        ):
                            val = r.get(key)
                            col_w.metric(label, f"{float(val):.4f}" if pd.notna(val) else "N/A")

            # ══════════════════════════════════════════════════════════════════
            #  MODE PHOTO (comportement original)
            # ══════════════════════════════════════════════════════════════════
            else:
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

                # ── Row 1: 3 columns ──────────────────────────────────────────
                c_src, c_pred, c_gt = st.columns(3)

                with c_src:
                    st.markdown("**Frame source**")
                    if img_src is not None:
                        st.image(cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB), width='stretch')
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

                # ── Row 2: TP/FP/FN overlay ──────────────────────────────────
                if img_src is not None and img_mask is not None and img_gt_mask is not None:
                    st.markdown("---")

                    thresh_vis = float(st.session_state.get("t2_threshold", 0.5))

                    h_src, w_src = img_src.shape[:2]
                    gray_pred_ov = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
                    if gray_pred_ov.shape != (h_src, w_src):
                        gray_pred_ov = cv2.resize(gray_pred_ov, (w_src, h_src),
                                                  interpolation=cv2.INTER_NEAREST)

                    gt_ov = img_gt_mask
                    if gt_ov.shape != (h_src, w_src):
                        gt_ov = cv2.resize(gt_ov, (w_src, h_src), interpolation=cv2.INTER_NEAREST)

                    bin_pred = gray_pred_ov > int(thresh_vis * 255)
                    bin_gt   = gt_ov > int(thresh_vis * 255)

                    tp = bin_pred & bin_gt
                    fp = bin_pred & ~bin_gt
                    fn = ~bin_pred & bin_gt

                    alpha = 0.45
                    overlay_f = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB).astype(np.float32)
                    overlay_f[tp] = overlay_f[tp] * (1 - alpha) + np.array([0, 200, 0],   np.float32) * alpha
                    overlay_f[fp] = overlay_f[fp] * (1 - alpha) + np.array([220, 50, 50], np.float32) * alpha
                    overlay_f[fn] = overlay_f[fn] * (1 - alpha) + np.array([50, 50, 220], np.float32) * alpha
                    overlay_img = overlay_f.clip(0, 255).astype(np.uint8)

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 4 — Post-processing : aperçu visuel sur une frame
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_resource(show_spinner=False)
def _t4_load_model(model_key: str):
    m = MODEL_REGISTRY[model_key]()
    m.load()
    return m


def _t4_run_inference(model_key: str, video_path: Path, frame_idx: int, window_size: int):
    """Run model on `window_size` frames ending at frame_idx. Cache hit avoids re-inference when only post-process params change."""
    cache: dict = st.session_state.setdefault("_t4_inf_cache", {})
    key = (model_key, str(video_path), int(frame_idx), int(window_size))
    if key in cache:
        return cache[key]

    model = _t4_load_model(model_key)
    model.reset_state()
    start = max(0, frame_idx - window_size + 1)
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(frame_idx - start + 1):
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    if not frames:
        return [], []

    masks = model.predict_batch(frames)
    masks = [np.asarray(m, dtype=np.float32).squeeze() for m in masks]

    if len(cache) > 4:
        cache.pop(next(iter(cache)))
    cache[key] = (frames, masks)
    return frames, masks


def _t4_apply_pp(frames, masks, methods):
    pp = PostProcessor({"methods": methods})
    pp.reset()
    return [pp.apply(m, frame_bgr=f) for m, f in zip(masks, frames)]


def _t4_get_gt_at(gt_path: Path, frame_idx: int):
    if not gt_path or not gt_path.exists():
        return None
    masks = _load_ground_truth_masks(gt_path, frame_idx + 1)
    if not masks or len(masks) <= frame_idx:
        return None
    m = masks[frame_idx]
    if m.dtype == np.uint8:
        m = m.astype(np.float32) / 255.0
    return np.clip(m, 0.0, 1.0)


def _t4_overlay(frame_bgr, mask):
    h, w = frame_bgr.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    bg = np.full_like(frame_bgr, (40, 40, 40))
    m3 = np.dstack([mask] * 3)
    out = (frame_bgr.astype(np.float32) * m3 + bg.astype(np.float32) * (1 - m3)).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


with tab4:
    st.title("⚙️ Post-processing — aperçu visuel")
    st.caption(
        "Ajustez les méthodes et **prévisualisez** leur effet sur une frame "
        "en comparant le masque brut au masque post-traité. "
        "La config sauvegardée est utilisée lors du prochain benchmark."
    )

    _PP_MODEL_LABELS = {
        "mediapipe_portrait": "MediaPipe Portrait",
        "mobilenetv3_lraspp": "MobileNetV3 + LRASPP",
        "rvm": "RVM (MobileNetV3)",
    }

    col_cfg, col_prev = st.columns([1, 2], gap="large")

    with col_cfg:
        chosen_pp_model = st.selectbox(
            "Modèle",
            options=list(_PP_MODEL_LABELS.keys()),
            format_func=lambda k: _PP_MODEL_LABELS[k],
            key="t4_model",
        )

        _t4_pairs = discover_datasets(curr_videos_dir, curr_gt_dir)
        if not _t4_pairs:
            st.warning("Aucune vidéo trouvée pour ce dataset.")
            st.stop()

        _video_options = {p[0].name: p for p in _t4_pairs}
        _chosen_video_name = st.selectbox(
            "Vidéo source",
            options=list(_video_options.keys()),
            key="t4_video",
        )
        _video_path, _gt_path = _video_options[_chosen_video_name]

        try:
            _n_frames, _fps, _ = _get_video_info(_video_path)
        except IOError:
            st.error("Impossible d'ouvrir la vidéo.")
            st.stop()

        if _n_frames <= 0:
            st.error("Vidéo vide.")
            st.stop()

        _frame_idx = st.slider(
            "Frame index",
            0, max(0, _n_frames - 1),
            value=min(30, max(0, _n_frames - 1)),
            key="t4_frame",
        )

        st.divider()
        st.markdown("**Méthodes**")

        current_cfg = pp_load_config(chosen_pp_model)
        methods_def = METHODS_BY_MODEL.get(chosen_pp_model, [])
        current_methods_by_name = {m["name"]: m for m in current_cfg.get("methods", [])}

        new_methods = []
        for method_def in methods_def:
            mname = method_def["name"]
            mlabel = method_def["label"]
            mdesc = method_def["description"]
            params_def = method_def["params"]

            cur = current_methods_by_name.get(mname, {})
            cur_enabled = cur.get("enabled", False)
            cur_params = cur.get("params", {})

            with st.container(border=True):
                enabled = st.checkbox(
                    f"**{mlabel}**",
                    value=cur_enabled,
                    key=f"t4_{chosen_pp_model}_{mname}_en",
                    help=mdesc,
                )
                new_params = {}
                for pname, pdef in params_def.items():
                    cur_val = cur_params.get(pname, pdef["default"])
                    if pdef["type"] == "int":
                        new_params[pname] = st.slider(
                            pname,
                            min_value=int(pdef["min"]),
                            max_value=int(pdef["max"]),
                            value=int(cur_val),
                            step=int(pdef["step"]),
                            key=f"t4_{chosen_pp_model}_{mname}_{pname}",
                            disabled=not enabled,
                        )
                    else:
                        new_params[pname] = st.slider(
                            pname,
                            min_value=float(pdef["min"]),
                            max_value=float(pdef["max"]),
                            value=float(cur_val),
                            step=float(pdef["step"]),
                            key=f"t4_{chosen_pp_model}_{mname}_{pname}",
                            disabled=not enabled,
                            format="%.2f",
                        )
                new_methods.append({
                    "name": mname,
                    "enabled": enabled,
                    "params": new_params,
                })

        st.divider()
        c1, c2 = st.columns(2)
        if c1.button("💾 Sauvegarder", type="primary", key="t4_save", use_container_width=True):
            pp_save_config(chosen_pp_model, {"model_key": chosen_pp_model, "methods": new_methods})
            st.success("Config sauvegardée.")
            st.rerun()
        if c2.button("🔄 Réinitialiser", key="t4_reset", use_container_width=True):
            pp_save_config(chosen_pp_model, pp_default_config(chosen_pp_model))
            st.info("Config réinitialisée.")
            st.rerun()

    with col_prev:
        st.subheader(f"🔍 Aperçu — frame {_frame_idx}")

        WINDOW = 8  # frames pour les méthodes temporelles (EMA)
        try:
            with st.spinner("Inférence…"):
                frames, raw_masks = _t4_run_inference(chosen_pp_model, _video_path, _frame_idx, WINDOW)
        except Exception as e:
            st.error(f"Inférence échouée : {e}")
            st.stop()

        if not frames or not raw_masks:
            st.error("Aucune frame récupérée.")
            st.stop()

        active_methods = [m for m in new_methods if m.get("enabled")]
        pp_masks = _t4_apply_pp(frames, raw_masks, new_methods) if active_methods else raw_masks

        frame = frames[-1]
        raw_mask = raw_masks[-1]
        pp_mask = pp_masks[-1]
        gt_mask = _t4_get_gt_at(_gt_path, _frame_idx)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W = frame.shape[:2]

        def _to_display(m):
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
            return np.clip(m, 0, 1)

        raw_disp = _to_display(raw_mask)
        pp_disp = _to_display(pp_mask)
        gt_disp = _to_display(gt_mask) if gt_mask is not None else None

        r1c1, r1c2 = st.columns(2)
        r1c1.image(frame_rgb, caption="Frame source", use_container_width=True)
        if gt_disp is not None:
            r1c2.image(gt_disp, caption="Ground truth", use_container_width=True, clamp=True)
        else:
            r1c2.info("Pas de ground truth pour cette frame.")

        st.divider()
        st.markdown("#### Masques")
        r2c1, r2c2 = st.columns(2)
        r2c1.image(raw_disp, caption="Avant post-process", use_container_width=True, clamp=True)
        if active_methods:
            r2c2.image(pp_disp, caption=f"Après ({len(active_methods)} méthode·s active·s)", use_container_width=True, clamp=True)
        else:
            r2c2.info("Activez une méthode pour voir l'effet.")

        st.markdown("#### Overlay sur la frame")
        r3c1, r3c2 = st.columns(2)
        r3c1.image(_t4_overlay(frame, raw_disp), caption="Avant post-process", use_container_width=True)
        if active_methods:
            r3c2.image(_t4_overlay(frame, pp_disp), caption="Après post-process", use_container_width=True)

        if active_methods:
            with st.expander("📐 Différence (zones modifiées)", expanded=False):
                diff = np.abs(pp_disp - raw_disp)
                diff_norm = np.clip(diff * 5, 0, 1)
                st.image(
                    diff_norm,
                    caption=f"|après - avant| × 5  (max diff = {float(diff.max()):.3f}, moyenne = {float(diff.mean()):.4f})",
                    use_container_width=True,
                    clamp=True,
                )

        with st.expander("📄 Config en cours (non sauvegardée)", expanded=False):
            st.json({"model_key": chosen_pp_model, "methods": new_methods})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 5 — Données enregistrées
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ─── Descriptions de tous les modèles ────────────────────────────────────────
_MODEL_INFO = {
    "mediapipe_portrait": {
        "display": "MediaPipe Portrait",
        "icon": "🎭",
        "desc": (
            "Segmentation sémantique de portrait par Google MediaPipe (TFLite). "
            "Optimisé pour les selfies et gros plans. Traitement **frame par frame** "
            "sans mémoire temporelle : chaque image est traitée indépendamment. "
            "Très faible latence grâce au modèle ultra-léger (~7.5M FLOPs, 256×256 px fixe)."
        ),
        "tech": ["TFLite", "GPU auto-sélectionné", "256×256 px"],
    },
    "mobilenetv3_lraspp": {
        "display": "MobileNetV3 + LRASPP",
        "icon": "📱",
        "desc": (
            "Segmentation sémantique COCO via **PyTorch**. Extrait la classe *personne* "
            "(index 15 COCO/VOC) grâce à un encodeur MobileNetV3-Large + décodeur "
            "LRASPP léger. Supporte le **batching GPU natif** pour accélérer le traitement "
            "multi-frames. Normalisation ImageNet appliquée en entrée."
        ),
        "tech": ["PyTorch", "CUDA / MPS / CPU", "256×256 px", "~70M FLOPs"],
    },
    "rvm": {
        "display": "RVM (Robust Video Matting)",
        "icon": "🎬",
        "desc": (
            "Réseau récurrent **GRU** (Gated Recurrent Unit) conçu pour la cohérence "
            "temporelle en vidéo. Produit un **alpha matte** continu [0–1]. Maintient "
            "un état caché entre frames pour éviter le flickering naturellement. "
            "ONNX Runtime avec accélération **CoreML** sur Mac. Résolution dynamique."
        ),
        "tech": ["ONNX Runtime", "CoreML / CUDA / CPU", "Résolution dynamique", "~600M FLOPs"],
    },
    "mediapipe_selfie_multiclass": {
        "display": "MediaPipe Selfie Multiclass",
        "icon": "👤",
        "desc": (
            "Variante **multiclasse** de MediaPipe Selfie : différencie corps, visage, "
            "cheveux et arrière-plan en une seule passe. TFLite, optimisé pour les selfies."
        ),
        "tech": ["TFLite", "GPU auto-sélectionné", "Multiclasse"],
    },
    "mediapipe_landscape": {
        "display": "MediaPipe Landscape",
        "icon": "🌄",
        "desc": (
            "Variante **grand angle** de MediaPipe : optimisée pour les caméras de bureau "
            "et les personnes éloignées. Modèle paysage dédié, moins précis sur les "
            "gros plans mais meilleur sur les scènes larges."
        ),
        "tech": ["TFLite", "Résolution paysage"],
    },
    "modnet": {
        "display": "MODNet",
        "icon": "✂️",
        "desc": (
            "Matting en **temps réel sans trimap**. Décompose le problème en sémantique "
            "+ détail + fusion pour un détourage net des cheveux et contours fins. "
            "Disponible en PyTorch ou ONNX. Spécialisé portraits naturels."
        ),
        "tech": ["PyTorch / ONNX", "Portraits et scènes naturelles"],
    },
    "pphumanseg_v2": {
        "display": "PP-HumanSeg V2",
        "icon": "🧍",
        "desc": (
            "Modèle de segmentation humaine de **PaddlePaddle**, exporté en ONNX. "
            "Raffinage des contours intégré. Bon équilibre précision/vitesse pour "
            "la segmentation du corps entier, y compris en plan large."
        ),
        "tech": ["ONNX Runtime", "Corps entier"],
    },
    "efficient_vit": {
        "display": "EfficientViT",
        "icon": "⚡",
        "desc": (
            "Vision Transformer efficace pour la segmentation sémantique. Réduit la "
            "complexité des **attention layers** par des approximations linéaires, "
            "restant rapide tout en capturant le contexte global de l'image."
        ),
        "tech": ["PyTorch", "ViT léger"],
    },
    "trimap_matting": {
        "display": "Trimap Matting V3",
        "icon": "🎨",
        "desc": (
            "Pipeline hybride : estimation de **pose** → GrabCut avec trimap automatique "
            "→ filtre guidé pour les détails fins → EMA temporel pour la stabilité. "
            "Le plus précis sur les zones semi-transparentes (cheveux, flou) mais aussi "
            "le plus lent. Idéal pour les plans fixes ou les scènes avec fond simple."
        ),
        "tech": ["OpenCV GrabCut", "Filtre guidé", "EMA temporel", "Trimap auto"],
    },
}

with tab5:
    st.title("🗄️ Données enregistrées")
    st.markdown(
        "Consultez toutes les données de benchmark stockées sur cet ordinateur. "
        "Téléchargez les résultats CSV, les vidéos de masques et accédez aux détails "
        "de chaque modèle et version post-processée."
    )

    # ── Section 1 : Résultats CSV globaux ────────────────────────────────────
    st.subheader("📊 Résultats globaux")
    _csv_path = OUTPUT_DIR / "benchmark_results.csv"
    _json_path = OUTPUT_DIR / "benchmark_results.json"

    if _csv_path.exists():
        _df_global = pd.read_csv(_csv_path)
        st.dataframe(_df_global, use_container_width=True)
        _dl1, _dl2, _ = st.columns([1, 1, 3])
        with _dl1:
            st.download_button(
                "📥 Télécharger CSV",
                data=_csv_path.read_bytes(),
                file_name="benchmark_results.csv",
                mime="text/csv",
                key="t5_dl_csv",
            )
        with _dl2:
            if _json_path.exists():
                st.download_button(
                    "📥 Télécharger JSON",
                    data=_json_path.read_bytes(),
                    file_name="benchmark_results.json",
                    mime="application/json",
                    key="t5_dl_json",
                )
    else:
        st.info("Aucun résultat CSV trouvé — lancez d'abord un benchmark.")

    st.divider()

    # ── Section 2 : Données par modèle ───────────────────────────────────────
    st.subheader("🤖 Données par modèle")

    _masks_root5 = OUTPUT_DIR / "masks"

    if not _masks_root5.is_dir():
        st.warning("Aucun dossier de masques trouvé.")
    else:
        for _m_dir in sorted(_masks_root5.iterdir()):
            if not _m_dir.is_dir():
                continue
            _m_display = _m_dir.name.replace("_", " ")

            # Trouver la clé du registry pour ce modèle
            _m_key = None
            for k, info in _MODEL_INFO.items():
                if info["display"].replace(" ", "_") == _m_dir.name or \
                   info["display"] == _m_display:
                    _m_key = k
                    break

            _info = _MODEL_INFO.get(_m_key, {}) if _m_key else {}

            with st.expander(
                f"{_info.get('icon', '🔹')} **{_m_display}**",
                expanded=False,
            ):
                # ── Bandeau descriptif du modèle ──────────────────────────────
                if _info:
                    st.info(
                        f"**{_info['display']}**  \n"
                        + _info["desc"]
                        + "\n\n🔧 **Stack** : " + " · ".join(_info.get("tech", []))
                    )

                # ── Métriques depuis CSV ───────────────────────────────────────
                if _csv_path.exists():
                    _df_m = _df_global[_df_global["model"] == _m_display] if "model" in _df_global.columns else pd.DataFrame()
                    if not _df_m.empty:
                        st.markdown("**Métriques agrégées**")
                        _num_cols = [c for c in ["iou_mean", "boundary_f_mean", "flow_warping_error", "latency_p95_ms"] if c in _df_m.columns]
                        if _num_cols:
                            _agg = _df_m[_num_cols].mean(numeric_only=True)
                            _mcols = st.columns(len(_num_cols))
                            _labels = {
                                "iou_mean": "IoU moyen",
                                "boundary_f_mean": "Boundary F",
                                "flow_warping_error": "FWE",
                                "latency_p95_ms": "Latence p95 (ms)",
                            }
                            for ci, col_name in enumerate(_num_cols):
                                _mcols[ci].metric(_labels.get(col_name, col_name), f"{_agg[col_name]:.4f}")

                # ── Runs numérotés ─────────────────────────────────────────────
                _run_dirs5 = sorted(d for d in _m_dir.iterdir() if d.is_dir() and d.name.startswith("run_"))
                _flat_masks = list(_m_dir.glob("*_mask.mp4")) + list(_m_dir.glob("*_segmented.mp4"))

                if _run_dirs5:
                    st.markdown(f"**{len(_run_dirs5)} run(s) enregistré(s)**")
                    for _rd in _run_dirs5:
                        _rd_label = _rd.name

                        # Lire run_config.json (ou legacy postprocess_config.json)
                        _rc5_path = _rd / "run_config.json"
                        _legacy5_path = _rd / "postprocess_config.json"
                        _rc5: dict = {}
                        if _rc5_path.exists():
                            try:
                                _rc5 = json.loads(_rc5_path.read_text())
                            except Exception:
                                pass
                        elif _legacy5_path.exists():
                            try:
                                _rc5 = {"postprocess_config": json.loads(_legacy5_path.read_text())}
                            except Exception:
                                pass

                        _pp5 = _rc5.get("postprocess_config") or {}
                        _active5 = [f"`{m['name']}`" for m in _pp5.get("methods", []) if m.get("enabled")]
                        _settings5 = _rc5.get("settings", {})
                        _summ5 = _rc5.get("results_summary", {})
                        _run_mp4s = sorted(_rd.glob("*_mask.mp4"))
                        _run_segs = sorted(_rd.glob("*_segmented.mp4"))
                        _run_csv5 = _rd / "results.csv"
                        _run_cfg5 = _rc5_path if _rc5_path.exists() else (_legacy5_path if _legacy5_path.exists() else None)

                        with st.container(border=True):
                            _r1, _r2 = st.columns([3, 1])
                            with _r1:
                                # En-tête du run
                                _ts = _rc5.get("timestamp", "")[:16].replace("T", " ")
                                st.markdown(f"📁 **{_rd_label}** — {_ts or 'date inconnue'}")

                                # Post-process
                                st.markdown("🔧 Post-process : " + (" · ".join(_active5) if _active5 else "aucun"))

                                # Settings
                                if _settings5:
                                    _sflags = [k.replace("_", " ") for k, v in _settings5.items() if v]
                                    if _sflags:
                                        st.caption("⚙️ " + " · ".join(_sflags))

                                # Résumé des résultats
                                if _summ5:
                                    _sm_cols = st.columns(4)
                                    for _ci, (_sk, _sl) in enumerate([
                                        ("iou_mean", "IoU"), ("boundary_f_mean", "BF"),
                                        ("flow_warping_error", "FWE"), ("latency_p95_ms", "Lat. p95"),
                                    ]):
                                        _v = _summ5.get(_sk)
                                        _sm_cols[_ci].metric(_sl, f"{_v:.4f}" if _v is not None else "—")

                                # Nombre de vidéos
                                _total_vids = len(_run_mp4s) + len(_run_segs)
                                st.caption(f"{_total_vids} vidéo(s) · {len(list(_rd.iterdir()))} fichier(s)")

                            with _r2:
                                # Téléchargement CSV par run
                                if _run_csv5.exists():
                                    st.download_button(
                                        "📥 CSV",
                                        data=_run_csv5.read_bytes(),
                                        file_name=f"{_m_dir.name}_{_rd_label}_results.csv",
                                        mime="text/csv",
                                        key=f"t5_csv_{_m_dir.name}_{_rd.name}",
                                    )
                                # Téléchargement config
                                if _run_cfg5:
                                    st.download_button(
                                        "📥 Config",
                                        data=_run_cfg5.read_bytes(),
                                        file_name=f"{_m_dir.name}_{_rd_label}_config.json",
                                        mime="application/json",
                                        key=f"t5_cfg_{_m_dir.name}_{_rd.name}",
                                    )
                                # Téléchargement ZIP (vidéos + config + CSV)
                                if _run_mp4s or _run_segs:
                                    _zip_buf = io.BytesIO()
                                    with zipfile.ZipFile(_zip_buf, "w", zipfile.ZIP_DEFLATED) as _zf:
                                        for _f in list(_run_mp4s) + list(_run_segs):
                                            _zf.write(_f, _f.name)
                                        if _run_cfg5:
                                            _zf.write(_run_cfg5, _run_cfg5.name)
                                        if _run_csv5.exists():
                                            _zf.write(_run_csv5, "results.csv")
                                    _zip_buf.seek(0)
                                    st.download_button(
                                        f"📦 ZIP",
                                        data=_zip_buf.read(),
                                        file_name=f"{_m_dir.name}_{_rd_label}.zip",
                                        mime="application/zip",
                                        key=f"t5_zip_{_m_dir.name}_{_rd.name}",
                                    )

                            # Aperçu vidéos
                            if _run_mp4s:
                                with st.expander(f"🎬 Vidéos masques ({len(_run_mp4s)})"):
                                    for _vf in _run_mp4s[:6]:
                                        st.caption(_vf.name)
                                        st.video(open(_vf, "rb"))
                            if _run_segs:
                                with st.expander(f"🎨 Vidéos détourées ({len(_run_segs)})"):
                                    for _vf in _run_segs[:4]:
                                        st.caption(_vf.name)
                                        st.video(open(_vf, "rb"))

                elif _flat_masks:
                    # Structure plate (ancien format ou modèles non supportés)
                    st.markdown(f"**{len(_flat_masks)} fichier(s) vidéo (structure plate)**")
                    _dl_flat, _ = st.columns([1, 2])
                    with _dl_flat:
                        _zip_flat = io.BytesIO()
                        with zipfile.ZipFile(_zip_flat, "w", zipfile.ZIP_DEFLATED) as _zf:
                            for _f in _flat_masks:
                                _zf.write(_f, _f.name)
                        _zip_flat.seek(0)
                        st.download_button(
                            "📦 Télécharger toutes les vidéos",
                            data=_zip_flat.read(),
                            file_name=f"{_m_dir.name}_videos.zip",
                            mime="application/zip",
                            key=f"t5_zip_flat_{_m_dir.name}",
                        )
                    with st.expander(f"🎬 Vidéos ({len(_flat_masks)})"):
                        for _vf in _flat_masks[:6]:
                            st.caption(_vf.name)
                            st.video(open(_vf, "rb"))
                else:
                    st.info("Aucune vidéo enregistrée pour ce modèle.")
