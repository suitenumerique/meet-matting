import streamlit as st
from core.parameters import ParameterSpec


def render_widget(spec: ParameterSpec, key: str):
    """Render a single Streamlit widget from a ParameterSpec. Returns the user's value."""
    if spec.type == "int":
        return st.slider(
            spec.label,
            int(spec.min_value or 0),
            int(spec.max_value or 100),
            int(spec.default),
            int(spec.step or 1),
            help=spec.help,
            key=key,
        )
    if spec.type == "float":
        return st.slider(
            spec.label,
            float(spec.min_value or 0.0),
            float(spec.max_value or 1.0),
            float(spec.default),
            float(spec.step or 0.01),
            help=spec.help,
            key=key,
        )
    if spec.type == "bool":
        return st.checkbox(spec.label, bool(spec.default), help=spec.help, key=key)
    if spec.type == "choice":
        choices = spec.choices or []
        idx = choices.index(spec.default) if spec.default in choices else 0
        return st.selectbox(spec.label, choices, index=idx, help=spec.help, key=key)
    return st.text_input(spec.label, str(spec.default), help=spec.help, key=key)


def render_component_config(component_cls, key_prefix: str) -> dict:
    """Render all parameter widgets for a component. Returns kwargs dict ready for instantiation."""
    return {
        spec.name: render_widget(spec, f"{key_prefix}_{spec.name}")
        for spec in component_cls.parameter_specs()
    }
