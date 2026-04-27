# Adding a Preprocessor

Preprocessors transform the input frame **before** it is passed to the matting model. They are useful for noise reduction, colour normalisation, or any frame-level augmentation the model benefits from.

## Data contract

```
__call__(frame: np.ndarray) -> np.ndarray
```

- **Input**: RGB image, shape `(H, W, 3)`, dtype `uint8`.
- **Output**: RGB image, shape `(H, W, 3)`, dtype `uint8`.

## Minimal example

```python
# preprocessing/my_blur.py
import cv2
from core.base import Preprocessor
from core.parameters import ParameterSpec
from core.registry import preprocessors

@preprocessors.register
class MyBlur(Preprocessor):
    name = "my_blur"          # unique key; appears in the UI dropdown
    description = "Applies a median blur to reduce salt-and-pepper noise."

    @classmethod
    def parameter_specs(cls):
        return [
            ParameterSpec(
                name="ksize", type="int", default=3,
                label="Kernel size", min_value=1, max_value=15, step=2,
                help="Must be odd.",
            ),
        ]

    def __call__(self, frame):
        return cv2.medianBlur(frame, self.params["ksize"])
```

## ParameterSpec types and their UI widgets

| `type` | Widget | Required extra fields |
|--------|--------|-----------------------|
| `"int"` | `st.slider` (integer) | `min_value`, `max_value`, `step` |
| `"float"` | `st.slider` (float) | `min_value`, `max_value`, `step` |
| `"bool"` | `st.checkbox` | — |
| `"choice"` | `st.selectbox` | `choices` (list) |
| `"str"` | `st.text_input` | — |

## Registration rules

1. The file **must not** start with `_` (underscore-prefixed files are skipped by auto-discovery).
2. The class **must** be decorated with `@preprocessors.register`.
3. `name` must be unique across all registered preprocessors — a `ValueError` is raised on collision.
