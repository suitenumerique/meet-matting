"""Benchmark package — suppresses noisy Streamlit and TFLite warnings at import time."""

import logging
import warnings

# Hard-disable Streamlit context warnings
for logger_name in [
    "streamlit.runtime.scriptrunner_utils.script_run_context",
    "streamlit.runtime.scriptrunner",
    "streamlit.runtime.state.session_state_proxy",
    "streamlit",
]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.ERROR)
    logger.disabled = True

warnings.filterwarnings("ignore", message="missing ScriptRunContext")
