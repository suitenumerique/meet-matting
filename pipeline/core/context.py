"""
Shared context for inter-component communication within a single frame.
Uses a strict Singleton pattern to ensure all components share the same data,
even if module reloads occur in Streamlit.
"""
import logging

logger = logging.getLogger(__name__)

class SharedContext:
    _instance = None
    _data = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SharedContext, cls).__new__(cls)
            import random
            cls._instance.id = random.randint(1000, 9999)
            logger.info(f"--- NEW SHARED CONTEXT CREATED (ID: {cls._instance.id}) ---")
        return cls._instance

    def set_val(self, key: str, value: any):
        self._data[key] = value

    def get_val(self, key: str, default: any = None):
        return self._data.get(key, default)

    def clear(self):
        self._data.clear()

# Global instance
_ctx = SharedContext()

def set_val(key, value):
    _ctx.set_val(key, value)

def get_val(key, default=None):
    return _ctx.get_val(key, default)

def clear():
    _ctx.clear()
