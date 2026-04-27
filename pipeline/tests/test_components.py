import numpy as np
import pytest
from core.parameters import ParameterSpec
from core.registry import models, postprocessors, preprocessors

# Ensure all components are registered before tests run.
preprocessors.discover("preprocessing")
models.discover("models")
postprocessors.discover("postprocessing")

REGISTRIES = [preprocessors, models, postprocessors]


def _default_params(cls):
    return {s.name: s.default for s in cls.parameter_specs()}


@pytest.mark.parametrize("registry", REGISTRIES, ids=["preprocessors", "models", "postprocessors"])
def test_parameter_specs_return_list_of_spec_objects(registry):
    for name in registry.names():
        cls = registry.get(name)
        specs = cls.parameter_specs()
        assert isinstance(specs, list), f"{cls.__name__}.parameter_specs() must return a list"
        for spec in specs:
            assert isinstance(spec, ParameterSpec), (
                f"{cls.__name__}.parameter_specs() must contain ParameterSpec objects, "
                f"got {type(spec)}"
            )


@pytest.mark.parametrize("registry", REGISTRIES, ids=["preprocessors", "models", "postprocessors"])
def test_instantiation_with_defaults_succeeds(registry):
    for name in registry.names():
        cls = registry.get(name)
        params = _default_params(cls)
        instance = cls(**params)
        assert instance is not None


def test_gaussian_blur_even_kernel_does_not_crash():
    from preprocessing.gaussian_blur import GaussianBlur

    blur = GaussianBlur(kernel_size=4, sigma=1.0)  # 4 is even — should be corrected internally
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    result = blur(frame)
    assert result.shape == (64, 64, 3)
    assert result.dtype == np.uint8
