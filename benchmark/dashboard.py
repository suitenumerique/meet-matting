"""
Tableau de bord Streamlit pour le Benchmark Video Matting.

Offre une interface moderne pour :
  - Sélectionner les modèles.
  - Choisir le nombre de vidéos (aléatoire ou non).
  - Régler le seuil de binarisation ou lancer une analyse multi-seuils.
  - Visualiser les résultats en temps réel.
  - Recharger l'historique des benchmarks précédents.
"""

import os

# Désactivation TOTALE des logs MediaPipe/GLog au démarrage
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import logging
import queue
import sys
import threading
import time
import traceback
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

from benchmark.config import GROUND_TRUTH_DIR, OUTPUT_DIR, VIDEOS_DIR
from benchmark.models import MODEL_REGISTRY
from benchmark.runner import (
    _chroma_key_to_mask,
    _get_video_info,
    _is_chroma_key,
    discover_datasets,
    get_frame_at,
    run_benchmark,
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
    "fps",
    "pipeline_fps",
]

# Colonnes attendues pour le tableau "Moyennes par modèle" (ordre strict)
MODEL_SUMMARY_COLUMNS = [
    "modèle",
    "latency_p95",
    "FPS réel du pipeline complet",
    "IoU mean",
    "boundary_f mean",
    "flow warping error",
    "best_threshold",
]

METRIC_GUIDE = """
### 📖 Guide des métriques

| Métrique | Plage | Sens | Seuils indicatifs |
|---|---|---|---|
| **IoU** (Intersection over Union) | [0, 1] | ↑ mieux | < 0.6 mauvais · 0.7–0.85 correct · > 0.85 bon |
| **Boundary F-measure** | [0, 1] | ↑ mieux | < 0.5 contours flous · > 0.75 contours nets |
| **Flow Warping Error** | [0, 1] | ↓ mieux | < 0.02 très stable · 0.02–0.07 légère instab. · > 0.07 clignotement visible |
| **Latence p95** | ms | ↓ mieux | < 33 ms → 30 fps temps réel · < 16 ms → 60 fps |
| **FPS** (inférence) | frames/s | ↑ mieux | 1000 / latency_mean_ms — vitesse d'inférence pure du modèle |
| **FPS pipeline** | frames/s | ↑ mieux | num_frames / wall_clock — inclut inférence + I/O + métriques |

**IoU** : proportion de pixels correctement classés (intersection / union entre masque prédit et vérité terrain).

**Boundary F-measure** : précision des contours (méthode DAVIS).

**Flow Warping Error** : stabilité temporelle (Lai et al. 2018) — cohérence entre frames warpées par le flux optique.

**Latence p95** : 95e percentile des temps de prédiction frame-par-frame (batch=1).

**FPS pipeline** : débit effectif englobant tout le traitement (chargement vidéo, inférence, GT, métriques, export).
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helpers — analyse et affichage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _threshold_sensitivity_chart(results: list):
    """Affiche les courbes IoU et Boundary F en fonction du seuil de binarisation."""
    import altair as alt

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
    st.altair_chart(chart_iou | chart_bf, width="stretch")


def _styled_df(df: pd.DataFrame):
    """Applique du highlighting sur le DataFrame des résultats."""
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
    if "fps" in sub.columns:
        s = s.highlight_max(subset=["fps"], color="#d4edda")
    if "pipeline_fps" in sub.columns:
        s = s.highlight_max(subset=["pipeline_fps"], color="#d4edda")
    return s


def _scatter_chart(df: pd.DataFrame):
    """Scatter IoU vs latence p95, bulles ∝ FPS d'inférence."""
    import altair as alt

    needed = {"latency_p95_ms", "iou_mean", "model"}
    if not needed.issubset(df.columns):
        return

    plot_df = df[df["status"] == "OK"].dropna(subset=list(needed)).copy()
    if plot_df.empty:
        return

    if "fps" in plot_df.columns:
        plot_df["_size"] = plot_df["fps"].fillna(0).clip(lower=5)
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
        .properties(title="Qualité vs Latence (bulles ∝ FPS d'inférence)", height=400)
        .interactive()
    )
    st.altair_chart(chart, width="stretch")


def _model_summary(df: pd.DataFrame, best_by_model: dict[str, float]) -> pd.DataFrame:
    """Construit le tableau 'Moyennes par modèle' avec les colonnes strictes."""
    if "status" in df.columns:
        df_ok = df[df["status"] == "OK"].copy()
    else:
        df_ok = df.copy()
    if df_ok.empty:
        return pd.DataFrame(columns=MODEL_SUMMARY_COLUMNS)

    rows = []
    for model_name, sub in df_ok.groupby("model"):
        t_best = best_by_model.get(model_name)
        rows.append(
            {
                "modèle": model_name,
                "latency_p95": (
                    round(float(sub["latency_p95_ms"].mean()), 2)
                    if "latency_p95_ms" in sub.columns
                    else None
                ),
                "FPS réel du pipeline complet": (
                    round(float(sub["pipeline_fps"].mean()), 2)
                    if "pipeline_fps" in sub.columns
                    else None
                ),
                "IoU mean": (
                    round(float(sub["iou_mean"].mean()), 4) if "iou_mean" in sub.columns else None
                ),
                "boundary_f mean": (
                    round(float(sub["boundary_f_mean"].mean()), 4)
                    if "boundary_f_mean" in sub.columns
                    else None
                ),
                "flow warping error": (
                    round(float(sub["flow_warping_error"].mean()), 4)
                    if "flow_warping_error" in sub.columns
                    else None
                ),
                "best_threshold": t_best if t_best is not None else "—",
            }
        )
    return pd.DataFrame(rows, columns=MODEL_SUMMARY_COLUMNS)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Historique des benchmarks (lecture)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _list_history_runs(output_dir: Path) -> list[dict]:
    """Liste les benchmarks précédents (lecture stricte — toute erreur lève)."""
    history_root = output_dir / "history"
    if not history_root.is_dir():
        return []

    runs: list[dict] = []
    for run_dir in sorted(history_root.glob("benchmark_*")):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            err = f"summary.json manquant dans {run_dir} — dossier ignoré"
            logging.warning(err)
            st.warning(err)
            continue
        try:
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
        except json.JSONDecodeError as e:
            err = f"Impossible de lire {summary_path}: {e}"
            st.error(err)
            raise
        required_keys = {"id", "timestamp", "models", "num_videos"}
        missing = required_keys - summary.keys()
        if missing:
            err = f"Clés manquantes dans {summary_path}: {missing}"
            st.error(err)
            raise KeyError(err)
        summary["_dir"] = run_dir
        runs.append(summary)
    return runs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Nettoyage après arrêt
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _cleanup_stopped_run(cleanup_info: dict) -> None:
    """Supprime tous les fichiers/dossiers générés pendant un benchmark arrêté."""
    import shutil

    paths = [Path(p) for p in cleanup_info.get("model_dirs_created", [])]
    deleted, errors = [], []
    for p in paths:
        if p.exists():
            try:
                shutil.rmtree(p)
                deleted.append(str(p))
                logging.info("🗑️  Supprimé : %s", p)
            except Exception as e:
                errors.append(f"{p}: {e}")
                logging.error("Erreur suppression %s : %s", p, e)

    if deleted:
        st.success(f"✅ {len(deleted)} dossier(s) supprimé(s) suite à l'arrêt.")
    if errors:
        for err in errors:
            st.error(f"Impossible de supprimer : {err}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Auto-refresh fragment for tab1 background benchmark
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.fragment(run_every=2)
def _t1_live_panel(total_combos: int):
    """Vide la queue de résultats toutes les 2 s et rafraîchit l'affichage."""
    q = st.session_state.get("t1_queue")
    if q is None:
        return

    while True:
        try:
            msg_type, data = q.get_nowait()
        except queue.Empty:
            break
        if msg_type == "result":
            st.session_state.setdefault("t1_results", []).append(data)
        elif msg_type == "done":
            st.session_state["t1_running"] = False
            st.session_state["t1_elapsed"] = data
        elif msg_type == "summary":
            st.session_state["t1_summary"] = data
        elif msg_type == "stopped":
            st.session_state["t1_running"] = False
            st.session_state["t1_stopped"] = True
            st.session_state["t1_cleanup_info"] = data  # model_dirs_created
        elif msg_type == "error":
            # Le worker a échoué — afficher la stack trace dans l'UI et arrêter.
            err_msg, err_tb = data
            st.session_state["t1_running"] = False
            st.session_state["t1_error"] = (err_msg, err_tb)
        else:
            # Type inconnu — anomalie à signaler explicitement.
            st.session_state["t1_running"] = False
            raise RuntimeError(f"Message de queue inattendu: {msg_type!r}")

    err = st.session_state.get("t1_error")
    if err:
        err_msg, err_tb = err
        st.error(f"❌ Le benchmark a échoué : {err_msg}")
        st.code(err_tb, language="text")
        return

    if st.session_state.get("t1_stopped"):
        st.warning("🛑 Benchmark arrêté par l'utilisateur.")
        cleanup_info = st.session_state.pop("t1_cleanup_info", {})
        if cleanup_info:
            _cleanup_stopped_run(cleanup_info)
        st.session_state["t1_stopped"] = False
        st.session_state["t1_results"] = []
        return

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
        summary = st.session_state.get("t1_summary", {})
        global_fps = summary.get("global_pipeline_fps")
        if global_fps is not None:
            st.success(
                f"✅ Benchmark terminé en {elapsed:.1f}s "
                f"— FPS pipeline global : **{global_fps:.2f}**"
            )
        else:
            st.success(f"✅ Benchmark terminé en {elapsed:.1f}s !")

        df_done = pd.DataFrame(results)
        best_thresholds = summary.get("best_thresholds", {})

        st.subheader("📈 Moyennes par modèle")
        avg = _model_summary(df_done, best_thresholds)
        if not avg.empty:
            st.table(avg)

        st.subheader("⚡ Qualité vs Latence")
        _scatter_chart(df_done)

        if any(r.get("threshold_analysis") for r in results):
            st.subheader("🔬 Sensibilité au seuil de binarisation")
            st.markdown(
                "Évolution de l'IoU et du Boundary F en fonction du seuil — "
                "le meilleur seuil par modèle (argmax IoU moyenné sur les vidéos) est utilisé "
                "dans le tableau 'Moyennes par modèle' et dans le CSV exporté."
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Sidebar globale
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.sidebar.title("⚙️ Configuration")

# Dataset hard-codé : on utilise le dataset par défaut défini dans config.py.
curr_videos_dir = VIDEOS_DIR
curr_gt_dir = GROUND_TRUTH_DIR

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab3 = st.tabs(
    [
        "🚀 Benchmark complet",
        "🖼️ Comparateur visuel",
    ]
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 1 — Full benchmark
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.title("🚀 Benchmark complet")

    # Sidebar de Tab 1
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

        video_indices: list[int] | None = None
        num_videos = 0
        use_shuffle = False

        if sel_mode == "Toutes":
            num_videos = 0
        elif sel_mode == "Aléatoire":
            num_videos = st.number_input(
                "Nombre de vidéos",
                1,
                max(1, total_videos),
                min(10, max(1, total_videos)),
                key="t1_num_v",
            )
            use_shuffle = True
        elif sel_mode == "Plage (Index)":
            if total_videos > 1:
                idx_range = st.slider(
                    "Choisissez la plage",
                    0,
                    total_videos - 1,
                    (0, min(9, total_videos - 1)),
                    key="t1_range",
                )
                video_indices = list(range(idx_range[0], idx_range[1] + 1))
            else:
                st.warning("Pas assez de vidéos pour utiliser une plage.")
                video_indices = []
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
                    video_indices = [i for i in video_indices if 0 <= i < total_videos]
                    video_indices = sorted(set(video_indices))
                except ValueError:
                    st.error(
                        "Format invalide. Utilisez des nombres, virgules ou tirets (ex: 0, 5-10)."
                    )

        st.divider()
        st.subheader("🔬 Analyse avancée")
        analyze_thresholds = st.checkbox(
            "Analyser tous les seuils (0.1 → 0.9)",
            value=False,
            key="t1_thresh_analysis",
            help="Calcule IoU, Boundary F et Flow Warping Error pour 9 seuils. "
            "Le meilleur seuil par modèle est ensuite calculé en maximisant l'IoU "
            "moyenné sur l'ensemble des vidéos.",
        )

        # Le seuil de binarisation n'est affiché que lorsque l'analyse multi-seuils est désactivée.
        if not analyze_thresholds:
            threshold = st.slider(
                "Seuil de binarisation",
                min_value=0.10,
                max_value=0.90,
                value=0.50,
                step=0.05,
                key="t1_threshold",
                help="Seuil appliqué pour binariser les masques (foreground ≥ seuil).",
            )
        else:
            threshold = 0.5  # Valeur sentinelle non utilisée en mode sweep.

    # ── Sélection des modèles ──
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
        mode_str = "Analyse multi-seuils" if analyze_thresholds else f"Seuil = {threshold:.2f}"
        st.info(f"""
- **Vidéos détectées** : {total_videos}
- **Séquences à traiter** : {n_seq}
- **Combinaisons totales** : {len(selected_models) * n_seq}
- **Mode** : {mode_str}
- **Dossier sortie** : `{OUTPUT_DIR}`
        """)

        launch_btn = st.button("🚀 LANCER LE BENCHMARK", type="primary", key="t1_launch")

        if st.session_state.get("t1_running"):
            if st.button("🛑 Arrêter le benchmark", type="secondary", key="t1_stop"):
                stop_ev = st.session_state.get("t1_stop_event")
                if stop_ev is not None:
                    stop_ev.set()
                st.warning("Arrêt en cours… la frame en cours se termine avant de s'arrêter.")

    with st.expander("📖 Guide des métriques", expanded=False):
        st.markdown(METRIC_GUIDE)

    # ── Exécution ──
    if launch_btn:
        if not selected_models:
            st.error("Sélectionnez au moins un modèle.")
        else:
            bg_queue: queue.Queue = queue.Queue()
            _stop_event = threading.Event()
            st.session_state["t1_queue"] = bg_queue
            st.session_state["t1_stop_event"] = _stop_event
            st.session_state["t1_results"] = []
            st.session_state["t1_running"] = True
            st.session_state["t1_stopped"] = False
            st.session_state["t1_elapsed"] = 0
            st.session_state["t1_summary"] = {}
            st.session_state["t1_error"] = None

            _models = [MODEL_REGISTRY[k]() for k in selected_models]
            _videos_dir = curr_videos_dir
            _gt_dir = curr_gt_dir
            _num_videos = num_videos
            _shuffle = use_shuffle
            _v_indices = video_indices
            _analyze = analyze_thresholds
            _threshold = threshold

            def _bg_benchmark():
                def _on_result(entry: dict):
                    bg_queue.put(("result", entry))

                t_start = time.time()
                try:
                    bench_result = run_benchmark(
                        models=_models,
                        videos_dir=_videos_dir,
                        gt_dir=_gt_dir,
                        num_videos=_num_videos,
                        random_selection=_shuffle,
                        video_indices=_v_indices,
                        save_masks=False,
                        save_video=True,
                        save_segmented=True,
                        on_result=_on_result,
                        analyze_thresholds=_analyze,
                        threshold=_threshold,
                        stop_event=_stop_event,
                    )
                    if bench_result.get("stopped"):
                        bg_queue.put(
                            (
                                "stopped",
                                {
                                    "model_dirs_created": [
                                        str(p) for p in bench_result.get("model_dirs_created", [])
                                    ],
                                },
                            )
                        )
                    else:
                        bg_queue.put(
                            (
                                "summary",
                                {
                                    "global_pipeline_fps": bench_result.get("global_pipeline_fps"),
                                    "best_thresholds": bench_result.get("best_thresholds", {}),
                                    "history_dir": str(bench_result.get("history_dir") or ""),
                                },
                            )
                        )
                except Exception as e:
                    bg_queue.put(("error", (str(e), traceback.format_exc())))
                finally:
                    bg_queue.put(("done", time.time() - t_start))

            threading.Thread(target=_bg_benchmark, daemon=True).start()
            st.rerun()

    # Live panel
    n_seq = num_videos if num_videos > 0 else total_videos
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

    # ── Historique des benchmarks ──
    st.markdown("---")
    with st.expander("📂 Historique des benchmarks", expanded=False):
        history_runs = _list_history_runs(OUTPUT_DIR)
        if not history_runs:
            st.info(
                "Aucun benchmark historique trouvé. "
                f"Les runs seront stockés dans `{OUTPUT_DIR / 'history'}`."
            )
        else:
            hist_df = pd.DataFrame(
                [
                    {
                        "#": r["id"],
                        "timestamp": r["timestamp"],
                        "modèles": ", ".join(r["models"]),
                        "vidéos": r["num_videos"],
                        "FPS pipeline global": r.get("global_pipeline_fps", "—"),
                        "analyse multi-seuils": r.get("analyze_thresholds", False),
                        "seuil": (r.get("threshold") if r.get("threshold") is not None else "—"),
                    }
                    for r in history_runs
                ]
            )
            st.dataframe(hist_df, width="stretch", hide_index=True)

            run_ids = [r["id"] for r in history_runs]
            chosen_id = st.selectbox(
                "Recharger le benchmark numéro",
                options=run_ids,
                index=len(run_ids) - 1,
                key="t1_history_pick",
            )
            chosen_run = next(r for r in history_runs if r["id"] == chosen_id)
            chosen_dir = chosen_run["_dir"]
            csv_path = chosen_dir / "benchmark_results.csv"
            if not csv_path.exists():
                err = f"benchmark_results.csv manquant dans {chosen_dir}"
                st.error(err)
                raise FileNotFoundError(err)

            df_hist = pd.read_csv(csv_path)
            st.subheader(f"Benchmark #{chosen_id} — détail")
            st.dataframe(df_hist, width="stretch", hide_index=True)
            st.subheader(f"Benchmark #{chosen_id} — Moyennes par modèle")
            best = chosen_run.get("best_thresholds") or {}
            st.table(_model_summary(df_hist, best))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 3 — Visual comparator (avec lecture vidéo synchronisée)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.title("🖼️ Comparateur visuel")
    st.markdown(
        "Visualisez côte à côte la **frame source**, le **masque prédit**, "
        "la **vérité terrain** et l'**overlay TP/FP/FN**."
    )

    # Index des (modèle, vidéo)
    masks_root3 = OUTPUT_DIR / "masks"
    pairs_index: dict = {}
    if masks_root3.is_dir():
        for md in sorted(masks_root3.iterdir()):
            if not md.is_dir():
                continue
            mask_stems = sorted(
                f.stem.replace(f"_{md.name}_mask", "") for f in md.glob("*_mask.mp4")
            )
            if mask_stems:
                pairs_index[md.name] = mask_stems
                continue
            seg_stems = sorted(
                f.stem.replace(f"_{md.name}_segmented", "") for f in md.glob("*_segmented.mp4")
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

        thresh_vis = st.slider(
            "Seuil de binarisation",
            min_value=0.10,
            max_value=0.90,
            value=float(st.session_state.get("t3_threshold", 0.50)),
            step=0.05,
            key="t3_threshold",
        )
        st.metric("Seuil de binarisation appliqué", f"{thresh_vis:.2f}")

        if chosen_model and chosen_video:
            src_path = curr_videos_dir / f"{chosen_video}.mp4"
            gt_path = curr_gt_dir / f"{chosen_video}.mp4"
            mask_path = masks_root3 / chosen_model / f"{chosen_video}_{chosen_model}_mask.mp4"
            seg_path = masks_root3 / chosen_model / f"{chosen_video}_{chosen_model}_segmented.mp4"

            ref_path = (
                mask_path if mask_path.exists() else (seg_path if seg_path.exists() else src_path)
            )
            n_frames_vis = 1
            if ref_path.exists():
                n_frames_vis, _fps, _res = _get_video_info(ref_path)

            frame_idx = st.slider(
                "Numéro de frame",
                min_value=0,
                max_value=max(0, n_frames_vis - 1),
                value=0,
                key="t3_frame_slider",
            )
            st.caption(f"Frame {frame_idx} / {max(0, n_frames_vis - 1)}")

            # Charger source + GT
            img_src = get_frame_at(src_path, frame_idx) if src_path.exists() else None
            img_gt_raw = get_frame_at(gt_path, frame_idx) if gt_path.exists() else None

            # Dimensions de référence : source d'abord, masque en fallback
            h_ref: int | None = None
            w_ref: int | None = None
            if img_src is not None:
                h_ref, w_ref = img_src.shape[:2]

            # Charger masque prédit — erreur explicite si ça échoue
            img_mask: np.ndarray | None = None
            mask_source = ""
            try:
                if mask_path.exists():
                    loaded = get_frame_at(mask_path, frame_idx)
                    if loaded is None:
                        raise RuntimeError(
                            f"get_frame_at a retourné None pour {mask_path} frame {frame_idx}"
                        )
                    img_mask = loaded
                    mask_source = "mask"
                elif seg_path.exists():
                    seg_frame = get_frame_at(seg_path, frame_idx)
                    if seg_frame is None:
                        raise RuntimeError(
                            f"get_frame_at a retourné None pour {seg_path} frame {frame_idx}"
                        )
                    seg_rgb = cv2.cvtColor(seg_frame, cv2.COLOR_BGR2RGB)
                    derived = np.any(seg_rgb > 10, axis=2).astype(np.uint8) * 255
                    img_mask = cv2.cvtColor(derived, cv2.COLOR_GRAY2BGR)
                    mask_source = "segmented"
            except Exception as _mask_err:
                logging.error(
                    "Erreur chargement masque pour le modèle '%s' frame %d : %s",
                    chosen_model,
                    frame_idx,
                    _mask_err,
                )
                st.error(
                    f"Impossible de charger le masque pour le modèle **{chosen_model}** "
                    f"(frame {frame_idx}) : {_mask_err}"
                )

            # Redimensionner le masque prédit aux proportions exactes de la source
            gray_pred: np.ndarray | None = None
            if img_mask is not None:
                try:
                    raw_gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
                    if h_ref is None or w_ref is None:
                        # Pas de source disponible : utiliser les dimensions natives du masque
                        h_ref, w_ref = raw_gray.shape[:2]
                    if raw_gray.shape != (h_ref, w_ref):
                        raw_gray = cv2.resize(
                            raw_gray, (w_ref, h_ref), interpolation=cv2.INTER_LINEAR
                        )
                    gray_pred = raw_gray
                except Exception as _resize_err:
                    logging.error(
                        "Erreur redimensionnement masque '%s' (%s → %dx%d) : %s",
                        chosen_model,
                        img_mask.shape,
                        w_ref or 0,
                        h_ref or 0,
                        _resize_err,
                    )
                    st.error(
                        f"Erreur de redimensionnement du masque pour **{chosen_model}** "
                        f"({img_mask.shape[:2]} → {h_ref}×{w_ref}) : {_resize_err}"
                    )

            # Convertir GT en masque et aligner sur les dimensions de référence
            img_gt_mask: np.ndarray | None = None
            if img_gt_raw is not None:
                if _is_chroma_key(img_gt_raw):
                    gt_bin = _chroma_key_to_mask(img_gt_raw)
                    img_gt_mask = (gt_bin * 255).astype(np.uint8)
                else:
                    img_gt_mask = cv2.cvtColor(img_gt_raw, cv2.COLOR_BGR2GRAY).astype(np.uint8)
                if h_ref is not None and w_ref is not None and img_gt_mask.shape != (h_ref, w_ref):
                    img_gt_mask = cv2.resize(
                        img_gt_mask, (w_ref, h_ref), interpolation=cv2.INTER_NEAREST
                    ).astype(np.uint8)

            # ── Ligne 1 : source, masque, GT ─────────────────────────────────
            c_src, c_pred, c_gt = st.columns(3)
            with c_src:
                st.markdown("**Frame source**")
                if img_src is not None:
                    st.image(cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB), width="stretch")
                else:
                    st.warning("Vidéo source introuvable.")
            with c_pred:
                label = "**Masque prédit**" + (
                    " *(dérivé du segmenté)*" if mask_source == "segmented" else ""
                )
                st.markdown(label)
                if gray_pred is not None:
                    st.image(gray_pred, width="stretch", clamp=True)
                else:
                    st.warning("Masque prédit introuvable.")
            with c_gt:
                st.markdown("**GT binarisé**")
                if img_gt_mask is not None:
                    st.image(img_gt_mask, width="stretch", clamp=True)
                else:
                    st.warning("GT introuvable.")

            # ── Ligne 2 : overlay TP/FP/FN ───────────────────────────────────
            if img_src is not None and gray_pred is not None and img_gt_mask is not None:
                st.markdown("---")
                bin_pred = gray_pred > int(thresh_vis * 255)
                bin_gt = img_gt_mask > int(thresh_vis * 255)

                tp = bin_pred & bin_gt
                fp = bin_pred & ~bin_gt
                fn = ~bin_pred & bin_gt

                alpha = 0.45
                overlay_f = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB).astype(np.float32)
                if overlay_f.shape[:2] != gray_pred.shape[:2]:
                    overlay_f = cv2.resize(
                        overlay_f,
                        (gray_pred.shape[1], gray_pred.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    ).astype(np.float32)
                overlay_f[tp] = (
                    overlay_f[tp] * (1 - alpha) + np.array([0, 200, 0], np.float32) * alpha
                )
                overlay_f[fp] = (
                    overlay_f[fp] * (1 - alpha) + np.array([220, 50, 50], np.float32) * alpha
                )
                overlay_f[fn] = (
                    overlay_f[fn] * (1 - alpha) + np.array([50, 50, 220], np.float32) * alpha
                )
                overlay_img = overlay_f.clip(0, 255).astype(np.uint8)

                from benchmark.metrics import compute_boundary_f_measure, compute_iou

                pred_f32 = gray_pred.astype(np.float32) / 255.0
                gt_f32 = img_gt_mask.astype(np.float32) / 255.0
                frame_iou = compute_iou([pred_f32], [gt_f32], threshold=thresh_vis)
                frame_bf = compute_boundary_f_measure(pred_f32, gt_f32, threshold=thresh_vis)

                col_ov, col_leg = st.columns([3, 1])
                with col_ov:
                    st.markdown("**Overlay TP / FP / FN**")
                    st.image(overlay_img, width="stretch")
                with col_leg:
                    st.markdown("**Légende**")
                    st.markdown(
                        """
- 🟢 **Vert** — Vrai positif (TP)
- 🔴 **Rouge** — Faux positif (FP)
- 🔵 **Bleu** — Faux négatif (FN)
- *(fond inchangé = vrai négatif)*
"""
                    )
                    st.markdown("**Métriques de cette frame**")
                    st.metric("IoU", f"{frame_iou:.3f}")
                    st.metric("Boundary F", f"{frame_bf:.3f}")
                    n_tp = int(tp.sum())
                    n_fp = int(fp.sum())
                    n_fn = int(fn.sum())
                    st.caption(f"TP={n_tp:,}  FP={n_fp:,}  FN={n_fn:,} px")
