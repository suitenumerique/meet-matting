from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st
from config import VIDEO_DIR
from core.registry import models, postprocessors, preprocessors, upsamplers
from core.video_io import list_videos

from ui.widgets import render_component_config


@dataclass
class SidebarSelection:
    video_path: Path | None
    model_name: str
    model_params: dict
    weights_path: str | None
    upsampler: tuple[str, dict]
    bg_color: tuple[int, int, int]
    preprocessors: list[tuple[str, dict]] = field(default_factory=list)
    postprocessors: list[tuple[str, dict]] = field(default_factory=list)


def render_sidebar() -> SidebarSelection:
    """Render the full sidebar and return a :class:`SidebarSelection`.

    Sections:
        1. Video selector
        2. Model selector + parameters
        3. Upsampling method + parameters
        4. Preprocessor multiselect + per-step parameters
        5. Postprocessor multiselect + per-step parameters
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

    # --- 3. Upsampling ---
    with st.sidebar.expander("Upsampling", expanded=False):
        st.caption(
            "How the low-resolution mask is scaled back up to the original frame resolution. "
            "Only applies to models that run inference at a reduced resolution."
        )
        upsampler_names = upsamplers.names()
        chosen_upsampler = st.selectbox(
            "Method",
            options=upsampler_names,
            key="upsampler_select",
        )
        upsampler_cls = upsamplers.get(chosen_upsampler)
        st.caption(upsampler_cls.description)
        upsampler_params = render_component_config(upsampler_cls, key_prefix="upsampler")
        upsampler: tuple[str, dict] = (chosen_upsampler, upsampler_params)

    # --- 4. Preprocessors ---
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

    # --- 5. Background colour ---
    with st.sidebar.expander("Background colour", expanded=False):
        st.caption("Colour used for pixels identified as background in the final composite.")
        hex_color = st.color_picker("Background", value="#000000", key="bg_color")
        h = hex_color.lstrip("#")
        bg_color: tuple[int, int, int] = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    # --- 6. Postprocessors ---
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
        upsampler=upsampler,
        bg_color=bg_color,
        preprocessors=pre_configs,
        postprocessors=post_configs,
    )
