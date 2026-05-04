from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st
from config import VIDEO_DIR
from core.registry import (
    compositors,
    models,
    postprocessors,
    preprocessors,
    skip_strategies,
    upsamplers,
)
from core.video_io import list_videos

from ui.widgets import render_component_config

_BG_IMAGE_URLS: dict[str, str] = {
    "Bureau Moderne": "https://images.unsplash.com/photo-1497366216548-37526070297c?auto=format&fit=crop&w=1280&q=80",
    "Nature": "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?auto=format&fit=crop&w=1280&q=80",
}


@dataclass
class SidebarSelection:
    video_path: Path | None
    model_name: str
    model_params: dict
    weights_path: str | None
    upsampler: tuple[str, dict]
    bg_color: tuple[int, int, int]
    bg_image_name: str | None
    compositor: tuple[str, dict]
    skip_frames: int
    skip_strategy: tuple[str, dict]
    preprocessors: list[tuple[str, dict]] = field(default_factory=list)
    postprocessors: list[tuple[str, dict]] = field(default_factory=list)


def render_sidebar() -> SidebarSelection:
    """Render the full sidebar and return a :class:`SidebarSelection`.

    Sections (in order):
        1. Video selector
        2. Preprocessors
        3. Model selector + parameters
        4. Upsampling method + parameters
        5. Postprocessors
        6. Background colour
        7. Skip Frames
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

    # --- 2. Preprocessors ---
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
            if cls.details:
                st.caption(cls.details)
            params = render_component_config(cls, key_prefix=f"pre_{i}")
            pre_configs.append((name, params))

    # --- 3. Model ---
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
        # Ensure we don't crash if session_state was set to None during import
        weights_input = weights_input or ""
        weights_path: str | None = weights_input.strip() or None

        model_params = render_component_config(model_cls, key_prefix="model")

    # --- 4. Upsampling ---
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

    # --- 5. Postprocessors ---
    with st.sidebar.expander("Postprocessors", expanded=False):
        st.caption("Refinements applied to the raw mask *after* the model, before compositing.")
        post_names = [
            n for n in postprocessors.names() if not getattr(postprocessors.get(n), "hidden", False)
        ]
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
            if cls.details:
                st.caption(cls.details)
            params = render_component_config(cls, key_prefix=f"post_{i}")
            post_configs.append((name, params))

    # --- 6. Background ---
    with st.sidebar.expander("Background", expanded=False):
        bg_type = st.selectbox(
            "Type",
            options=["Couleur unie"] + list(_BG_IMAGE_URLS.keys()),
            key="bg_type",
        )
        if bg_type == "Couleur unie":
            hex_color = st.color_picker("Couleur", value="#000000", key="bg_color")
            h = hex_color.lstrip("#")
            bg_color: tuple[int, int, int] = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
            bg_image_name: str | None = None
        else:
            bg_color = (0, 0, 0)
            bg_image_name = bg_type

    # --- 7. Composition ---
    with st.sidebar.expander("Composition", expanded=False):
        st.caption(
            "Assemblage visuel final du sujet sur le fond. "
            "L'**Alpha blending** est la fusion classique par transparence. "
            "Le **Light wrapping** laisse la lumière du fond déborder subtilement "
            "sur les contours du sujet pour une intégration plus naturelle."
        )
        compositor_names = compositors.names()
        chosen_compositor = st.selectbox(
            "Méthode",
            options=compositor_names,
            key="compositor_select",
        )
        compositor_cls = compositors.get(chosen_compositor)
        st.caption(compositor_cls.description)
        compositor_params = render_component_config(compositor_cls, key_prefix="compositor")
        compositor: tuple[str, dict] = (chosen_compositor, compositor_params)

    # --- 8. Skip Frames ---
    with st.sidebar.expander("Skip Frames", expanded=False):
        st.caption(
            "Speed up export by running the model only every N frames. "
            "Choose how skipped frames are filled."
        )
        skip_frames = st.number_input(
            "Infer every N frames",
            min_value=1,
            value=1,
            help="1 = every frame, 2 = every other frame, etc.",
            key="skip_frames_n",
        )
        strategy_names = skip_strategies.names()
        chosen_strategy = st.selectbox(
            "Strategy for skipped frames",
            options=strategy_names,
            key="skip_strategy_select",
        )
        strategy_cls = skip_strategies.get(chosen_strategy)
        st.caption(strategy_cls.description)
        strategy_params = render_component_config(strategy_cls, key_prefix="skip_strategy")
        skip_strategy: tuple[str, dict] = (chosen_strategy, strategy_params)

    return SidebarSelection(
        video_path=video_path,
        model_name=model_name,
        model_params=model_params,
        weights_path=weights_path,
        upsampler=upsampler,
        bg_color=bg_color,
        bg_image_name=bg_image_name,
        compositor=compositor,
        skip_frames=skip_frames,
        skip_strategy=skip_strategy,
        preprocessors=pre_configs,
        postprocessors=post_configs,
    )
