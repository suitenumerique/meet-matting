"""
Shared context for inter-component communication within a single frame.
Uses a strict Singleton pattern to ensure all components share the same data,
even if module reloads occur in Streamlit.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SharedContext:
    _instance = None
    _data: dict[str, Any] = {}

    def __new__(cls):
        """Return the singleton instance, creating it on first call."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            import random

            cls._instance.id = random.randint(1000, 9999)
            logger.info(f"--- NEW SHARED CONTEXT CREATED (ID: {cls._instance.id}) ---")
        return cls._instance

    def set_val(self, key: str, value: Any) -> None:
        """Store *value* under *key* in the shared data dict."""
        self._data[key] = value

    def get_val(self, key: str, default: Any = None) -> Any:
        """Return the value stored under *key*, or *default* if absent."""
        return self._data.get(key, default)

    def clear(self):
        """Clear all stored values; called once per frame by the pipeline."""
        self._data.clear()


# Global instance
_ctx = SharedContext()


def set_val(key, value):
    """Store *value* under *key* in the per-frame shared context."""
    _ctx.set_val(key, value)


def get_val(key, default=None):
    """Return the value stored under *key*, or *default* if absent."""
    return _ctx.get_val(key, default)


def clear():
    """Clear all per-frame context values."""
    _ctx.clear()
