# Adding a Skip Strategy

Skip strategies produce a mask for frames that are deliberately not sent through the model. When the pipeline is configured to run inference only every N frames (to reduce CPU/GPU load), a skip strategy fills the gaps by deriving a substitute mask from the previous inferred result.

## Data contract

```
__call__(current_frame: np.ndarray, prev_frame: np.ndarray, prev_mask: np.ndarray) -> np.ndarray
```

- **`current_frame`**: the frame to fill, shape `(H, W, 3)`, dtype `uint8`, RGB.
- **`prev_frame`**: the last frame that was sent to the model, shape `(H, W, 3)`, dtype `uint8`, RGB.
- **`prev_mask`**: the mask produced for `prev_frame`, shape `(H, W)`, dtype `float32`, range `[0, 1]`.
- **Output**: estimated mask for `current_frame`, shape `(H, W)`, dtype `float32`, range `[0, 1]`.

## Minimal example

```python
# skip_strategies/fade.py
import numpy as np
from core.base import SkipStrategy
from core.parameters import ParameterSpec
from core.registry import skip_strategies


@skip_strategies.register
class Fade(SkipStrategy):
    name = "fade"
    description = "Linearly decays the previous mask towards zero over skipped frames."

    @classmethod
    def parameter_specs(cls) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                name="decay", type="float", default=0.9,
                label="Decay factor", min_value=0.0, max_value=1.0, step=0.01,
                help="Multiplied on the mask each skipped frame. 1.0 = no decay (same as Reuse).",
            ),
        ]

    def __call__(self, current_frame, prev_frame, prev_mask):
        return (prev_mask * self.params["decay"]).astype(np.float32)
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
2. The class **must** be decorated with `@skip_strategies.register`.
3. `name` must be unique — a `ValueError` is raised on collision.
