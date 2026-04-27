# Adding a Matting Model

Model wrappers load weights and run inference to produce an alpha matte from a single frame.

## Data contract

```
load(weights_path: str | None) -> None
infer(frame: np.ndarray) -> np.ndarray
```

- **`load`**: called once before inference. `weights_path` may be `None` for models with bundled weights.
- **`infer` input**: RGB image, shape `(H, W, 3)`, dtype `uint8`.
- **`infer` output**: alpha matte, shape `(H, W)`, dtype `float32`, range `[0, 1]`.

## Minimal example

```python
# models/my_model.py
import numpy as np
from core.base import MattingModel
from core.parameters import ParameterSpec
from core.registry import models

@models.register
class MyModel(MattingModel):
    name = "my_model"
    description = "Short description shown in the UI."

    @classmethod
    def parameter_specs(cls):
        return [
            ParameterSpec(
                name="threshold", type="float", default=0.5,
                label="Threshold", min_value=0.0, max_value=1.0, step=0.01,
            ),
        ]

    def load(self, weights_path=None):
        # Load ONNX / PyTorch / TFLite weights here.
        pass

    def infer(self, frame):
        h, w = frame.shape[:2]
        # Replace with real inference.
        return np.ones((h, w), dtype=np.float32)
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
2. The class **must** be decorated with `@models.register`.
3. `name` must be unique — a `ValueError` is raised on collision.
4. Do **not** import `torch`, `onnxruntime`, or any heavy ML library in `core/` or `ui/`. Keep those imports local to the model file.
