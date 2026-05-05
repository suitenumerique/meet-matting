"""Unit tests for the Registry class — registration, discovery, and error cases."""

import pytest
from core.registry import Registry


def test_register_and_get():
    """A class decorated with @reg.register must be retrievable via reg.get(name)."""
    reg = Registry("test")

    @reg.register
    class Foo:
        name = "foo"

    assert reg.get("foo") is Foo


def test_discover_imports_non_underscore_modules():
    """discover() must import every non-underscore module and populate the registry."""
    # Use the real preprocessors registry after discovery.
    from core.registry import preprocessors

    preprocessors.discover("preprocessing")
    assert "identity" in preprocessors.names()
    assert "gaussian_blur" in preprocessors.names()


def test_name_collision_raises():
    """Registering two classes with the same name must raise ValueError."""
    reg = Registry("test")

    @reg.register
    class Bar:
        name = "bar"

    with pytest.raises(ValueError, match="bar"):

        @reg.register
        class Bar2:
            name = "bar"


def test_get_unknown_raises_with_list():
    """get() for an unknown name must raise KeyError listing available names in the message."""
    reg = Registry("test")

    @reg.register
    class Baz:
        name = "baz"

    with pytest.raises(KeyError) as exc_info:
        reg.get("nonexistent")

    assert "baz" in str(exc_info.value)
    assert "nonexistent" in str(exc_info.value)
