"""Streamlit 2×2 panel for comparing the original frame, composite, raw mask, and final mask."""

import numpy as np
import streamlit as st


def display_four_panels(result: dict) -> None:
    """Display pipeline results as a 2x2 grid of images.

    Args:
        result: Dict returned by :meth:`MattingPipeline.process_frame`, containing
                ``original``, ``preprocessed``, ``raw_mask``, ``final_mask`` and ``final`` arrays.
    """
    # ── Postprocessing delta indicator ────────────────────────────────────────
    raw = result["raw_mask"].astype(np.float32)
    final_m = result["final_mask"].astype(np.float32)
    delta_max = float(np.max(np.abs(final_m - raw)))
    delta_mean = float(np.mean(np.abs(final_m - raw)))

    if delta_max > 1e-4:
        st.success(
            f"Post-processing actif — delta masque : max={delta_max:.4f} | moy={delta_mean:.4f}"
        )
    else:
        st.info(
            "Post-processing : aucun changement visible sur cette frame. "
            "Les filtres temporels (Kalman, One Euro) nécessitent plusieurs frames pour converger — "
            "l'effet sera visible dans la vidéo exportée."
        )

    # ── Row 1 : images complètes ──────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.image(
            result["original"],
            caption="Original (image complète)",
            width="stretch",
            output_format="JPEG",
        )

    with col2:
        st.image(
            result["final"],
            caption="Après post-processing (composite)",
            width="stretch",
            output_format="JPEG",
        )

    # ── Row 2 : masques pour comparaison ─────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        raw_uint8 = (result["raw_mask"] * 255).astype(np.uint8)
        st.image(
            raw_uint8,
            caption="Masque brut (avant post-processing)",
            width="stretch",
            output_format="JPEG",
        )

    with col4:
        final_uint8 = (result["final_mask"] * 255).astype(np.uint8)
        st.image(
            final_uint8,
            caption="Masque final (après post-processing)",
            width="stretch",
            output_format="JPEG",
        )
