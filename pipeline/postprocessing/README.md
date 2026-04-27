# Adding a Postprocessor

Postprocessors refine the raw alpha matte after model inference. They receive both the mask **and** the original (un-preprocessed) frame, so colour-guided filters such as guided filtering or edge detection have access to unmodified pixel values.

## Data contract

```
__call__(mask: np.ndarray, original_frame: np.ndarray) -> np.ndarray
```

- **`mask`**: alpha matte, shape `(H, W)`, dtype `float32`, range `[0, 1]`.
- **`original_frame`**: RGB image, shape `(H, W, 3)`, dtype `uint8` — un-preprocessed.
- **Output**: refined alpha matte, shape `(H, W)`, dtype `float32`, range `[0, 1]`.

## Minimal example

```python
# postprocessing/erode.py
import cv2
import numpy as np
from core.base import Postprocessor
from core.parameters import ParameterSpec
from core.registry import postprocessors

@postprocessors.register
class Erode(Postprocessor):
    name = "erode"
    description = "Shrinks the foreground mask by eroding its boundary."

    @classmethod
    def parameter_specs(cls):
        return [
            ParameterSpec(
                name="iterations", type="int", default=1,
                label="Iterations", min_value=1, max_value=10, step=1,
            ),
        ]

    def __call__(self, mask, original_frame):
        kernel = np.ones((3, 3), np.uint8)
        mask_u8 = (mask * 255).astype(np.uint8)
        eroded = cv2.erode(mask_u8, kernel, iterations=self.params["iterations"])
        return (eroded / 255.0).astype(np.float32)
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

1. The file **must not** start with `_`.
2. The class **must** be decorated with `@postprocessors.register`.
3. `name` must be unique — a `ValueError` is raised on collision.
