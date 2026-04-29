from __future__ import annotations

import importlib
import pkgutil


class Registry:
    """Auto-discovery registry for pipeline components."""

    def __init__(self, label: str):
        self._label = label
        self._store: dict[str, type] = {}

    def register(self, cls: type) -> type:
        """Decorator that registers *cls* under ``cls.name``.

        Raises:
            ValueError: if a class with the same name is already registered.
        """
        if cls.name in self._store:
            raise ValueError(
                f"[{self._label}] Name collision: '{cls.name}' is already registered "
                f"by {self._store[cls.name].__qualname__}."
            )
        self._store[cls.name] = cls
        return cls

    def discover(self, package_name: str) -> None:
        """Import every non-underscore module in *package_name*.

        This triggers the ``@register`` decorators on the way, populating the
        registry without any explicit import list.

        Args:
            package_name: Top-level package name (e.g. ``"preprocessing"``).
        """
        package = importlib.import_module(package_name)
        for _finder, module_name, _is_pkg in pkgutil.iter_modules(package.__path__):
            if not module_name.startswith("_"):
                importlib.import_module(f"{package_name}.{module_name}")

    def get(self, name: str) -> type:
        """Return the class registered under *name*.

        Raises:
            KeyError: with a message listing available names if *name* is unknown.
        """
        if name not in self._store:
            available = ", ".join(self.names()) or "(none registered yet)"
            raise KeyError(f"[{self._label}] Unknown component '{name}'. Available: {available}")
        return self._store[name]

    def names(self) -> list[str]:
        """Return a sorted list of registered component names."""
        return sorted(self._store)


preprocessors = Registry("preprocessors")
models = Registry("models")
postprocessors = Registry("postprocessors")
upsamplers = Registry("upsamplers")
skip_strategies = Registry("skip_strategies")
