# Adding a Compositing Technique

Compositors blend the segmented foreground over a background using the final alpha matte. This is the last step of the pipeline and produces the output frame shown to the user.

## Data contract

```
composite(fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray) -> np.ndarray
```

- **`fg`**: foreground RGB frame, shape `(H, W, 3)`, dtype `uint8`.
- **`bg`**: background, shape `(H, W, 3)`, dtype `float32`, range `[0, 255]`. Already resized to match `fg` by the pipeline.
- **`alpha`**: alpha matte, shape `(H, W)`, dtype `float32`, range `[0, 1]`.
- **Output**: composited image, shape `(H, W, 3)`, dtype `uint8`.

Note: `bg` is `float32` (not `uint8`) — cast `fg` to `float32` before arithmetic to avoid integer overflow.

## Minimal example

```python
# compositing/additive.py
import numpy as np
from core.base import Compositor
from core.parameters import ParameterSpec
from core.registry import compositors


@compositors.register
class Additive(Compositor):
    name = "additive"
    description = "Adds fg·α on top of bg·(1−α) with no colour correction."

    @classmethod
    def parameter_specs(cls) -> list[ParameterSpec]:
        return []

    def composite(self, fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        mask3 = alpha[..., np.newaxis]
        result = fg.astype(np.float32) * mask3 + bg * (1.0 - mask3)
        return np.clip(result, 0, 255).astype(np.uint8)
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
2. The class **must** be decorated with `@compositors.register`.
3. `name` must be unique — a `ValueError` is raised on collision.
