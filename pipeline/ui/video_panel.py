import numpy as np
import streamlit as st


def display_four_panels(result: dict) -> None:
    """Display pipeline results as a 2x2 grid of images.

    Args:
        result: Dict returned by :meth:`MattingPipeline.process_frame`, containing
                ``original``, ``preprocessed``, ``raw_mask``, and ``final`` arrays.
    """
    col1, col2 = st.columns(2)

    with col1:
        st.image(result["original"], caption="Original", use_container_width=True)

    with col2:
        st.image(result["preprocessed"], caption="After preprocessing", use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        mask_uint8 = (result["raw_mask"] * 255).astype(np.uint8)
        st.image(mask_uint8, caption="Model output (raw mask)", use_container_width=True)

    with col4:
        st.image(
            result["final"], caption="Final (mask applied to original)", use_container_width=True
        )
