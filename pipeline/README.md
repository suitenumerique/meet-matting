# Matting Pipeline Lab

A modular, plug-and-play background-matting pipeline with a Streamlit UI for testing configurations interactively.

## Purpose

The pipeline lets you chain any combination of **preprocessors → matting model → postprocessors** and inspect the result frame-by-frame. Components are auto-discovered: dropping a new Python file into the right folder is all it takes to make a new step appear in the UI.

## Quickstart

```bash
cd meet-matting/pipeline
pip install -r requirements.txt          # or: uv pip install -r requirements.txt
# Drop a video into data/videos/
streamlit run app.py
```

The sidebar will show the available models, preprocessors, and postprocessors automatically.

## How to add a new component

See the per-folder READMEs for worked examples and the exact method signatures:

- **Preprocessor** → [`preprocessing/README.md`](preprocessing/README.md)
- **Model** → [`models/README.md`](models/README.md)
- **Postprocessor** → [`postprocessing/README.md`](postprocessing/README.md)

Key rule: the filename must **not** start with `_`, and the class must be decorated with the appropriate registry decorator (`@preprocessors.register`, `@models.register`, `@postprocessors.register`).

## Data contracts

| Data | Shape | dtype | Range |
|------|-------|-------|-------|
| Frame (input/output) | `(H, W, 3)` | `uint8` | `[0, 255]` — RGB |
| Mask | `(H, W)` | `float32` | `[0, 1]` |

Postprocessors receive the **original, un-preprocessed** frame alongside the mask so that colour-guided filters have access to unmodified pixel values.
