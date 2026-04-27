import streamlit as st
from core.pipeline import MattingPipeline
from core.registry import models, postprocessors, preprocessors
from core.video_io import frame_count, read_frame
from ui.sidebar import render_sidebar
from ui.video_panel import display_four_panels

# Auto-discover all plugins.
preprocessors.discover("preprocessing")
models.discover("models")
postprocessors.discover("postprocessing")

st.set_page_config(layout="wide", page_title="Matting Pipeline Lab")
st.title("Background matting pipeline")

selection = render_sidebar()

if selection.video_path is None:
    st.info("Drop a video into `data/videos/` to begin.")
    st.stop()

# Build pipeline from selection.
pre_components = [preprocessors.get(name)(**params) for name, params in selection.preprocessors]
model = models.get(selection.model_name)(**selection.model_params)
model.load(selection.weights_path)  # may be None
post_components = [postprocessors.get(name)(**params) for name, params in selection.postprocessors]
pipeline = MattingPipeline(pre_components, model, post_components)

total = frame_count(selection.video_path)
idx = st.slider("Frame", 0, max(total - 1, 0), 0)
frame = read_frame(selection.video_path, idx)
result = pipeline.process_frame(frame)
display_four_panels(result)
