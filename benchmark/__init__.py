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

__version__ = "1.0.0"
