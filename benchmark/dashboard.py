"""
Streamlit dashboard for Video Matting Benchmark.

Provides a modern interface for:
  - Selecting models.
  - Choosing the number of videos (random or not).
  - Enabling mask saving.
  - Viewing results in real time.
"""

import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
import time
import logging

# Silence Streamlit context warnings that spam the console during parallel processing
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)

# Add the parent to the path to import the benchmark modules
sys.path.append(str(Path(__file__).parent.parent))

from benchmark.models import MODEL_REGISTRY
from benchmark.runner import run_benchmark, discover_datasets
from benchmark.config import VIDEOS_DIR, GROUND_TRUTH_DIR, OUTPUT_DIR, TEMP_RESULTS_DIR

# Page configuration
st.set_page_config(
    page_title="Video Matting Benchmark",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
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

# ── Sidebar: Configuration ──
st.sidebar.title("⚙️ Configuration")

st.sidebar.subheader("Videos")
available_datasets = discover_datasets(VIDEOS_DIR, GROUND_TRUTH_DIR)
total_videos = len(available_datasets)

num_videos = st.sidebar.number_input(
    "Number of videos",
    min_value=0,
    max_value=total_videos,
    value=min(1, total_videos),
    help="0 = process all videos"
)

use_shuffle = st.sidebar.checkbox("Random selection (Shuffle)", value=True)

st.sidebar.divider()

st.sidebar.subheader("Options")
save_masks = st.sidebar.checkbox("Save images (PNG)", value=False)
save_video = st.sidebar.checkbox("Save masks (.mp4)", value=False)
save_segmented = st.sidebar.checkbox("Save subject (.mp4)", value=True, help="Display person on black background")

st.sidebar.divider()

# ── Main content ──
st.title("🎬 Video Matting Benchmark Dashboard")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🤖 Model selection")
    st.write("Check the models you want to include in the benchmark.")

    # Selection toolbar
    c1, c2, c3 = st.columns([1, 1, 1])
    if c1.button("✅ Check all"):
        for k in MODEL_REGISTRY.keys():
            st.session_state[f"cb_{k}"] = True
        st.session_state.selected_models = list(MODEL_REGISTRY.keys())
        st.rerun()

    if c2.button("❌ Uncheck all"):
        for k in MODEL_REGISTRY.keys():
            st.session_state[f"cb_{k}"] = False
        st.session_state.selected_models = []
        st.rerun()

    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = list(MODEL_REGISTRY.keys())

    # Model cards grid
    st.markdown("---")

    # Metadata to enhance the UI
    model_info = {
        "mediapipe": {"tag": "⚡ FAST", "desc": "Google's mobile-optimised solution", "color": "#28a745"},
        "rvm": {"tag": "🎬 VIDEO", "desc": "Recurrent temporal coherence", "color": "#007bff"},
        "modnet": {"tag": "💎 QUALITY", "desc": "High-resolution portrait matting", "color": "#6f42c1"},
        "pphumanseg": {"tag": "📱 MOBILE", "desc": "Ultra-lightweight from PaddleSeg", "color": "#fd7e14"},
        "efficientvit": {"tag": "🚀 SOTA", "desc": "High-performance Transformer", "color": "#dc3545"},
        "mobilenetv3": {"tag": "⚖️ BALANCE", "desc": "Lightweight industry standard", "color": "#6c757d"},
    }

    selected_list = []

    # Iterate by groups to make rows of 2
    keys = list(MODEL_REGISTRY.keys())
    for i in range(0, len(keys), 2):
        row_cols = st.columns(2)
        for j in range(2):
            if i + j < len(keys):
                key = keys[i+j]
                m_instance = MODEL_REGISTRY[key]()
                info = model_info.get(key, {"tag": "MODEL", "desc": "Person Segmentation", "color": "#333"})

                with row_cols[j]:
                    # Styled card container
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

    # Update the state
    st.session_state.selected_models = selected_list
    selected_models = selected_list

with col2:
    st.subheader("ℹ️ Benchmark preview")
    st.info(f"""
    - **Videos detected**: {total_videos}
    - **Sequences to process**: {num_videos if num_videos > 0 else total_videos}
    - **Total combinations**: {len(selected_models) * (num_videos if num_videos > 0 else total_videos)}
    - **Output folder**: `{OUTPUT_DIR.relative_to(Path.cwd())}`
    """)

    launch_btn = st.button("🚀 LAUNCH BENCHMARK", type="primary")

# ── Execution ──
if launch_btn:
    if not selected_models:
        st.error("Please select at least one model.")
    else:
        with st.status("🛠️ Benchmark in progress...", expanded=True) as status:
            st.write("Initialising models...")
            models = [MODEL_REGISTRY[key]() for key in selected_models]

            st.write("Starting the benchmark engine...")
            t_start = time.time()

            # Redirect logs from runner.py to Streamlit if possible or just wait for completion
            # For now we just run the function
            results = run_benchmark(
                    models=models,
                    num_videos=num_videos if num_videos > 0 else None,
                    random_selection=use_shuffle,
                    save_masks=save_masks,
                    save_video=save_video,
                    save_segmented=save_segmented
                )

            t_end = time.time()
            status.update(label=f"✅ Completed in {t_end - t_start:.1f}s", state="complete")

        # ── Results ──
        st.success("Benchmark completed successfully!")

        if results:
            df = pd.DataFrame(results)
            # Clean up for display: only take columns that exist
            all_potential_cols = [
                "model", "video", "status", "latency_p95_ms",
                "iou_mean", "boundary_f_mean", "flow_warping_error", "flops_per_frame"
            ]
            display_cols = [c for c in all_potential_cols if c in df.columns]
            df_display = df[display_cols].copy()

            st.subheader("📊 Results")

            # Apply styling only to existing numeric columns
            numeric_cols = [c for c in ["iou_mean", "boundary_f_mean", "latency_p95_ms", "flow_warping_error"] if c in df_display.columns]

            styled_df = df_display.style
            if "iou_mean" in df_display.columns:
                styled_df = styled_df.highlight_max(subset=["iou_mean"], color='#d4edda')
            if "boundary_f_mean" in df_display.columns:
                styled_df = styled_df.highlight_max(subset=["boundary_f_mean"], color='#d4edda')
            if "latency_p95_ms" in df_display.columns:
                styled_df = styled_df.highlight_min(subset=["latency_p95_ms"], color='#d4edda')

            st.dataframe(styled_df, width='stretch')

            # CSV download button
            csv = df_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download results (CSV)",
                data=csv,
                file_name=f"benchmark_results_{int(time.time())}.csv",
                mime='text/csv',
            )

            # Average metrics per model
            st.subheader("📈 Averages per model")
            avg_df = df_display.groupby("model").mean(numeric_only=True).reset_index()
            if not avg_df.empty:
                st.table(avg_df)
                if "iou_mean" in avg_df.columns:
                    st.bar_chart(avg_df, x="model", y="iou_mean")
            else:
                st.info("Not enough valid data to compute averages.")
        else:
            st.warning("No results produced. Check your datasets.")

else:
    # Message if not yet launched
    st.info("Configure the parameters in the sidebar and click Launch.")

    # Display old results if they exist
    res_path = OUTPUT_DIR / "benchmark_results.csv"
    if res_path.exists():
        st.subheader("Last Run Results")
        old_df = pd.read_csv(res_path)
        st.dataframe(old_df.head(10), width='stretch')
