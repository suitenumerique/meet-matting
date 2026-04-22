"""
Tableau de bord Streamlit pour le Benchmark Video Matting.

Offre une interface moderne pour :
  - Sélectionner les modèles.
  - Choisir le nombre de vidéos (aléatoire ou non).
  - Activer la sauvegarde des masques.
  - Visualiser les résultats en temps réel.
"""

import streamlit as st
import pandas as pd
import sys
import os
import warnings
from pathlib import Path
import time
import logging

# Silence Streamlit context warnings handled in benchmark/__init__.py


# Ajouter le parent au path pour importer les modules benchmark
sys.path.append(str(Path(__file__).parent.parent))

# Silence MediaPipe / TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

from benchmark.models import MODEL_REGISTRY
from benchmark.runner import run_benchmark, discover_datasets
from benchmark.config import VIDEOS_DIR, GROUND_TRUTH_DIR, OUTPUT_DIR, TEMP_RESULTS_DIR, DATASETS

# Configuration de la page
st.set_page_config(
    page_title="Video Matting Benchmark",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style Custom
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar : Configuration ──
st.sidebar.subheader("Dataset")
selected_ds_name = st.sidebar.selectbox(
    "Choisir le dataset",
    options=list(DATASETS.keys()),
    index=0
)
ds_path = DATASETS[selected_ds_name]
curr_videos_dir = ds_path / "videos"
curr_gt_dir = ds_path / "ground_truth"

@st.cache_data
def get_available_datasets(v_dir, g_dir):
    return discover_datasets(v_dir, g_dir)

available_datasets = get_available_datasets(curr_videos_dir, curr_gt_dir)

total_videos = len(available_datasets)

num_videos = st.sidebar.number_input(
    "Nombre de vidéos", 
    min_value=0, 
    max_value=total_videos, 
    value=min(1, total_videos),
    help="0 = traiter toutes les vidéos"
)

use_shuffle = st.sidebar.checkbox("Sélection aléatoire (Shuffle)", value=True)

st.sidebar.divider()

st.sidebar.subheader("Options")
save_masks = st.sidebar.checkbox("Sauvegarder les images (PNG)", value=False)
save_video = st.sidebar.checkbox("Sauvegarder les masques (.mp4)", value=False)
save_segmented = st.sidebar.checkbox("Sauvegarder le sujet (.mp4)", value=True, help="Affiche la personne sur fond noir")

st.sidebar.divider()

# ── Main Content ──
st.title("🎬 Video Matting Benchmark Dashboard")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🤖 Sélection des Modèles")
    st.write("Cochez les modèles que vous souhaitez inclure dans le benchmark.")
    
    # Barre d'outils de sélection
    c1, c2, c3 = st.columns([1, 1, 1])
    if c1.button("✅ Tout cocher"):
        for k in MODEL_REGISTRY.keys():
            st.session_state[f"cb_{k}"] = True
        st.session_state.selected_models = list(MODEL_REGISTRY.keys())
        st.rerun()
        
    if c2.button("❌ Tout décocher"):
        for k in MODEL_REGISTRY.keys():
            st.session_state[f"cb_{k}"] = False
        st.session_state.selected_models = []
        st.rerun()

    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = list(MODEL_REGISTRY.keys())

    # Grille de cartes de modèles
    st.markdown("---")
    
    # Métadonnées pour enrichir l'UI
    model_info = {
        "mediapipe": {"tag": "⚡ RAPIDE", "desc": "Solution Google optimisée mobile", "color": "#28a745"},
        "rvm": {"tag": "🎬 VIDÉO", "desc": "Cohérence temporelle récurrente", "color": "#007bff"},
        "modnet": {"tag": "💎 QUALITÉ", "desc": "Portrait matting haute résolution", "color": "#6f42c1"},
        "pphumanseg": {"tag": "📱 MOBILE", "desc": "Ultra-léger par PaddleSeg", "color": "#fd7e14"},
        "efficientvit": {"tag": "🚀 SOTA", "desc": "Transformer haute performance", "color": "#dc3545"},
        "mobilenetv3": {"tag": "⚖️ ÉQUILIBRE", "desc": "Standard industriel léger", "color": "#6c757d"},
    }

    selected_list = []
    
    # On itère par groupes pour faire des rangées de 2
    keys = list(MODEL_REGISTRY.keys())
    for i in range(0, len(keys), 2):
        row_cols = st.columns(2)
        for j in range(2):
            if i + j < len(keys):
                key = keys[i+j]
                m_instance = MODEL_REGISTRY[key]()
                info = model_info.get(key, {"tag": "MODÈLE", "desc": "Segmentation Personne", "color": "#333"})
                
                with row_cols[j]:
                    # Conteneur stylisé pour la "carte"
                    with st.container(border=True):
                        is_selected = st.checkbox(
                            f"**{m_instance.name}**", 
                            value=(key in st.session_state.selected_models),
                            key=f"cb_{key}",
                            help=info['desc']
                        )
                        st.markdown(f"<span style='background-color:{info['color']}; color:white; padding:2px 6px; border-radius:4px; font-size:10px; font-weight:bold;'>{info['tag']}</span>", unsafe_allow_html=True)
                        st.caption(info['desc'])
                        
                        if is_selected:
                            selected_list.append(key)

    # Mise à jour de l'état
    st.session_state.selected_models = selected_list
    selected_models = selected_list

with col2:
    st.subheader("ℹ️ Aperçu du Benchmark")
    st.info(f"""
    - **Vidéos détectées** : {total_videos}
    - **Séquences à traiter** : {num_videos if num_videos > 0 else total_videos}
    - **Combinaisons total** : {len(selected_models) * (num_videos if num_videos > 0 else total_videos)}
    - **Dossier Sortie** : `{OUTPUT_DIR.relative_to(Path.cwd())}`
    """)
    
    launch_btn = st.button("🚀 LANCER LE BENCHMARK", type="primary")

# ── Exécution ──
if launch_btn:
    if not selected_models:
        st.error("Veuillez sélectionner au moins un modèle.")
    else:
        with st.status("🛠️ Benchmark en cours...", expanded=True) as status:
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            def update_progress(current, total, message):
                progress_bar.progress(current / total)
                progress_text.text(message)

            st.write("Initialisation des modèles...")
            models = [MODEL_REGISTRY[key]() for key in selected_models]
            
            st.write("Lancement du moteur de benchmark...")
            t_start = time.time()
            
            results = run_benchmark(
                    models=models,
                    videos_dir=curr_videos_dir,
                    gt_dir=curr_gt_dir,
                    num_videos=num_videos if num_videos > 0 else None,
                    random_selection=use_shuffle,
                    save_masks=save_masks,
                    save_video=save_video,
                    save_segmented=save_segmented,
                    progress_callback=update_progress
                )
            
            t_end = time.time()
            progress_bar.empty()
            progress_text.empty()
            status.update(label=f"✅ Terminé en {t_end - t_start:.1f}s", state="complete")

        
        # ── Résultats ──
        st.success("Benchmark terminé avec succès !")
        
        if results:
            df = pd.DataFrame(results)
            # Nettoyage pour affichage : ne prendre que les colonnes qui existent
            all_potential_cols = [
                "model", "video", "status", "latency_p95_ms", 
                "iou_mean", "boundary_f_mean", "flow_warping_error", "flops_per_frame"
            ]
            display_cols = [c for c in all_potential_cols if c in df.columns]
            df_display = df[display_cols].copy()
            
            st.subheader("📊 Résultats")
            
            # Application du style seulement sur les colonnes numériques existantes
            numeric_cols = [c for c in ["iou_mean", "boundary_f_mean", "latency_p95_ms", "flow_warping_error"] if c in df_display.columns]
            
            styled_df = df_display.style
            if "iou_mean" in df_display.columns:
                styled_df = styled_df.highlight_max(subset=["iou_mean"], color='#d4edda')
            if "boundary_f_mean" in df_display.columns:
                styled_df = styled_df.highlight_max(subset=["boundary_f_mean"], color='#d4edda')
            if "latency_p95_ms" in df_display.columns:
                styled_df = styled_df.highlight_min(subset=["latency_p95_ms"], color='#d4edda')
            
            st.dataframe(styled_df, width='stretch')
            
            # Bouton de téléchargement CSV
            csv = df_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger les résultats (CSV)",
                data=csv,
                file_name=f"benchmark_results_{int(time.time())}.csv",
                mime='text/csv',
            )
            
            # Métriques moyennes par modèle
            st.subheader("📈 Moyennes par Modèle")
            avg_df = df_display.groupby("model").mean(numeric_only=True).reset_index()
            
            if not avg_df.empty:
                # Arrondir pour un affichage propre
                avg_df_styled = avg_df.style.format(precision=4)
                st.dataframe(avg_df_styled, use_container_width=True)
                
                # Bouton de téléchargement des moyennes
                avg_csv = avg_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 Télécharger les MOYENNES par modèle (CSV)",
                    data=avg_csv,
                    file_name=f"model_averages_{int(time.time())}.csv",
                    mime='text/csv',
                    key="download_avg"
                )

                if "iou_mean" in avg_df.columns:
                    st.bar_chart(avg_df, x="model", y="iou_mean")
            else:
                st.info("Pas assez de données valides pour calculer des moyennes.")

        else:
            st.warning("Aucun résultat n'a été produit. Vérifiez vos fichiers datasets.")

else:
    # Message si pas encore lancé
    st.info("Configurez les paramètres dans la barre latérale et cliquez sur Lancer.")
    
    # Afficher les anciens résultats si ils existent
    res_path = OUTPUT_DIR / "benchmark_results.csv"
    if res_path.exists():
        st.subheader("Last Run Results")
        old_df = pd.read_csv(res_path)
        st.dataframe(old_df.head(10), width='stretch')
