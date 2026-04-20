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
from pathlib import Path
import time

# Ajouter le parent au path pour importer les modules benchmark
sys.path.append(str(Path(__file__).parent.parent))

from benchmark.models import MODEL_REGISTRY
from benchmark.runner import run_benchmark, discover_datasets
from benchmark.config import VIDEOS_DIR, GROUND_TRUTH_DIR, OUTPUT_DIR, TEMP_RESULTS_DIR

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
st.sidebar.title("⚙️ Configuration")

st.sidebar.subheader("Vidéos")
available_datasets = discover_datasets(VIDEOS_DIR, GROUND_TRUTH_DIR)
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
    st.subheader("🤖 Modèles")
    st.write("Sélectionnez les modèles à évaluer :")
    
    # Boutons de sélection groupée
    c1, c2 = st.columns(2)
    if c1.button("Tout sélectionner"):
        st.session_state.selected_models = list(MODEL_REGISTRY.keys())
    if c2.button("Tout désélectionner"):
        st.session_state.selected_models = []

    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = list(MODEL_REGISTRY.keys())

    selected_models = st.multiselect(
        "Modèles actifs",
        options=list(MODEL_REGISTRY.keys()),
        default=st.session_state.selected_models,
        format_func=lambda x: MODEL_REGISTRY[x]().name,
        key="model_multiselect"
    )

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
            st.write("Initialisation des modèles...")
            models = [MODEL_REGISTRY[key]() for key in selected_models]
            
            st.write("Lancement du moteur de benchmark...")
            t_start = time.time()
            
            # On redirige les logs de runner.py vers Streamlit si possible ou on attend juste la fin
            # Pour l'instant on exécute simplement la fonction
            results = run_benchmark(
                    models=models,
                    num_videos=num_videos if num_videos > 0 else None,
                    random_selection=use_shuffle,
                    save_masks=save_masks,
                    save_video=save_video,
                    save_segmented=save_segmented
                )
            
            t_end = time.time()
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
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Métriques moyennes par modèle
            st.subheader("📈 Moyennes par Modèle")
            avg_df = df_display.groupby("model").mean(numeric_only=True).reset_index()
            if not avg_df.empty:
                st.table(avg_df)
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
        st.dataframe(old_df.head(10), use_container_width=True)
