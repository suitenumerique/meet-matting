from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st
from config import VIDEO_DIR
from core.registry import models, postprocessors, preprocessors
from core.video_io import list_videos

from ui.widgets import render_component_config


@dataclass
class SidebarSelection:
    video_path: Path | None
    model_name: str
    model_params: dict
    weights_path: str | None
    preprocessors: list[tuple[str, dict]] = field(default_factory=list)
    postprocessors: list[tuple[str, dict]] = field(default_factory=list)


def render_sidebar() -> SidebarSelection:
    """Render the full sidebar and return a :class:`SidebarSelection`.

    Sections:
        1. Video selector
        2. Model selector + parameters
        3. Preprocessor multiselect + per-step parameters
        4. Postprocessor multiselect + per-step parameters
    """
    # --- 1. Video ---
    with st.sidebar.expander("Video", expanded=True):
        st.caption("Pick the source video to process.")
        videos = list_videos(VIDEO_DIR)
        if videos:
            chosen = st.selectbox(
                "Select video",
                options=videos,
                format_func=lambda p: p.name,
                key="video_select",
            )
            video_path: Path | None = chosen
        else:
            st.info("No videos found in `data/videos/`.")
            video_path = None

    # --- 2. Model ---
    with st.sidebar.expander("Model", expanded=True):
        st.caption("The segmentation model that produces the alpha matte.")
        model_names = models.names()
        model_name = st.selectbox("Model", model_names, key="model_select")
        model_cls = models.get(model_name)
        st.caption(model_cls.description)

        weights_input = st.text_input(
            "Weights path (optional)",
            value="",
            help="Absolute or relative path to a weights file. Leave empty to use default.",
            key="weights_path",
        )
        weights_path: str | None = weights_input.strip() or None

        model_params = render_component_config(model_cls, key_prefix="model")

    # --- 3. Preprocessors ---
    with st.sidebar.expander("Preprocessors", expanded=False):
        st.caption(
            "Transforms applied to each frame *before* the model sees it. Does not affect the final composite."
        )
        pre_names = preprocessors.names()
        chosen_pre = st.multiselect(
            "Preprocessing steps (applied in order)",
            options=pre_names,
            default=[],
            key="pre_select",
        )
        pre_configs: list[tuple[str, dict]] = []
        for i, name in enumerate(chosen_pre):
            cls = preprocessors.get(name)
            st.markdown(f"**{name}** — {cls.description}")
            params = render_component_config(cls, key_prefix=f"pre_{i}")
            pre_configs.append((name, params))

    # --- 4. Postprocessors ---
    with st.sidebar.expander("Postprocessors", expanded=False):
        st.caption("Refinements applied to the raw mask *after* the model, before compositing.")
        post_names = postprocessors.names()
        chosen_post = st.multiselect(
            "Postprocessing steps (applied in order)",
            options=post_names,
            default=[],
            key="post_select",
        )
        post_configs: list[tuple[str, dict]] = []
        for i, name in enumerate(chosen_post):
            cls = postprocessors.get(name)
            st.markdown(f"**{name}** — {cls.description}")
            params = render_component_config(cls, key_prefix=f"post_{i}")
            post_configs.append((name, params))

    return SidebarSelection(
        video_path=video_path,
        model_name=model_name,
        model_params=model_params,
        weights_path=weights_path,
        preprocessors=pre_configs,
        postprocessors=post_configs,
    )
