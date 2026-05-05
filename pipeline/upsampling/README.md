# Adding an Upsampling Method

Upsampling methods rescale a low-resolution mask (produced by a model that runs at reduced resolution) back to the full frame resolution. Guidance-based upsamplers use the full-resolution RGB frame to recover edge detail lost at the lower resolution.

## Data contract

```
_upsample_impl(low_res_mask: np.ndarray, guide: np.ndarray) -> np.ndarray
```

- **`low_res_mask`**: alpha matte, shape `(H_l, W_l)`, dtype `float32`, range `[0, 1]`.
- **`guide`**: full-resolution RGB frame, shape `(H_h, W_h, 3)`, dtype `uint8`.
- **Output**: upsampled mask, shape `(H_h, W_h)`, dtype `float32`, range `[0, 1]`.

Implement `_upsample_impl`, **not** `upsample`. The public `upsample()` method wraps `_upsample_impl` and handles timing; subclasses must not override it.

## Minimal example

```python
# upsampling/nearest.py
import cv2
from core.base import UpsamplingMethod
from core.parameters import ParameterSpec
from core.registry import upsamplers


@upsamplers.register
class Nearest(UpsamplingMethod):
    name = "nearest"
    description = "Nearest-neighbour interpolation — fastest, hard edges."

    @classmethod
    def parameter_specs(cls) -> list[ParameterSpec]:
        return []

    def _upsample_impl(self, low_res_mask, guide):
        H_h, W_h = guide.shape[:2]
        if low_res_mask.shape[:2] == (H_h, W_h):
            return low_res_mask
        return cv2.resize(low_res_mask, (W_h, H_h), interpolation=cv2.INTER_NEAREST)
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
2. The class **must** be decorated with `@upsamplers.register`.
3. `name` must be unique — a `ValueError` is raised on collision.
4. Implement `_upsample_impl`, not `upsample`.
