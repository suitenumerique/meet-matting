# Dev Guide

Quick reference for working on **meet-matting**. Python >= 3.11, `uv` as package manager.

## Overview

Two-phase project for real-time video matting:

- **Phase 1** — `benchmark/`: comparative evaluation of 10 segmentation models with five metrics (IoU, BF-measure, flow warping error, P95 latency, FLOPs).
- **Phase 2-3** — `pipeline/`: modular matting pipeline (preprocessing -> inference -> postprocessing) with a Streamlit interface. Runs locally; web integration is the next deployment target.

## Project structure

```
background-segmentation/
├── benchmark/            # Phase 1: model evaluation suite
│   ├── models/           # 10 model wrappers (MediaPipe, RVM, ModNet, ...)
│   ├── metrics.py        # IoU, BF-measure, FWE, P95 latency, FLOPs
│   ├── runner.py         # Benchmark executor
│   ├── dashboard.py      # Streamlit UI
│   └── config.py         # Paths and parameters
│
└── pipeline/             # Phase 2-3: modular matting pipeline
    ├── app.py            # Streamlit entry point
    ├── config.py         # Paths and supported video extensions
    ├── core/             # Base abstractions, orchestrator, registries
    ├── models/           # Model wrappers (auto-discovered)
    ├── preprocessing/    # Frame preprocessors (auto-discovered)
    ├── postprocessing/   # Mask postprocessors (auto-discovered)
    ├── upsampling/       # Upsampling methods (auto-discovered)
    ├── compositing/      # Compositing techniques (auto-discovered)
    ├── skip_strategies/  # Frame-skip strategies (auto-discovered)
    ├── ui/               # Streamlit UI components
    └── tests/            # pytest test suite
```

### Auto-discovery

Components in `models/`, `preprocessing/`, `postprocessing/`, `upsampling/`, `compositing/`, and `skip_strategies/` are loaded automatically by the registry system at startup. Three rules apply:

1. The filename must not start with `_`.
2. The class must carry the `@<registry>.register` decorator.
3. The `name` attribute must be unique within the registry.

## Install

```bash
uv sync                  # runtime deps
uv sync --all-groups     # runtime + dev (ruff, mypy, pre-commit, pytest)
```

`uv sync` creates `.venv/` and installs from `uv.lock`. No need to activate the venv — prefix commands with `uv run`.

To add a dependency:

```bash
uv add <package>                # runtime
uv add --group dev <package>    # dev only
```

Commit both `pyproject.toml` and `uv.lock`.

## Run the pipeline

```bash
uv run streamlit run pipeline/app.py
```

## Run the benchmark

```bash
uv run streamlit run benchmark/dashboard.py     # web UI
uv run python -m benchmark.run_benchmark --help # CLI
```

## Tests

```bash
uv run pytest
```

## Pipeline components

### Naming conventions

| Element             | Convention | Example                                    |
|---------------------|------------|--------------------------------------------|
| Classes             | PascalCase | `TemporalSmoothing`, `MattingModel`        |
| Functions / methods | snake_case | `parameter_specs()`, `process_frame()`     |
| Module files        | snake_case | `guided_filter.py`, `optical_flow_warp.py` |
| Class attributes    | snake_case | `name`, `description`, `params`            |
| ParameterSpec keys  | snake_case | `mask_value`, `cutoff`, `alpha`            |

### Data contracts

All pipeline stages exchange data under the following invariants:

- **Frames**: `np.ndarray`, shape `(H, W, 3)`, dtype `uint8`, RGB color space.
- **Masks**: `np.ndarray`, shape `(H, W)`, dtype `float32`, values in `[0.0, 1.0]`.

Postprocessors receive both the output mask **and** the original un-preprocessed frame. This enables color-guided filtering (guided filter, boundary blur, etc.) to operate on the unmodified source image rather than the preprocessed one.

### Component template

The pattern below applies to all auto-discovered components. The postprocessor variant is shown; replace `Postprocessor` / `postprocessors` with the appropriate base class and registry for other component types.

```python
from __future__ import annotations

import numpy as np

from core.base import Postprocessor
from core.parameters import ParameterSpec
from core.registry import postprocessors


@postprocessors.register
class MyPostprocessor(Postprocessor):
    name = "my_postprocessor"
    description = "Short description of the transformation applied."

    @classmethod
    def parameter_specs(cls) -> list[ParameterSpec]:
        return [
            ParameterSpec(name="alpha", type=float, default=0.5, min_value=0.0, max_value=1.0),
        ]

    def __call__(self, mask: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        alpha = self.params["alpha"]
        return mask
```

Stateful components (e.g. temporal smoothers) must implement `reset()` to clear internal state between videos.

Heavy third-party imports (`torch`, `onnxruntime`, `mediapipe`) belong inside the component's own module file — never in `core/` or `ui/` — to avoid loading them in unrelated contexts.

## Lint & format — ruff

Config lives in [pyproject.toml](pyproject.toml) (`[tool.ruff]`). Line length 100, rules `E/W/F/I/UP/B`.

```bash
uv run ruff check benchmark/             # lint
uv run ruff check --fix benchmark/       # lint + autofix
uv run ruff format benchmark/            # format
uv run ruff format --check benchmark/    # CI-style: fail if not formatted
```

## Type-check — mypy

Config in `[tool.mypy]`. Missing stubs are ignored (`cv2`, `mediapipe`, `fvcore`...).

```bash
uv run mypy benchmark/
uv run mypy pipeline/
```

If a third-party type is wrong but the runtime is fine, prefer a narrow `cast(...)` or `.astype(...)` over `# type: ignore`. `warn_unused_ignores = true` will flag stale ignores.

## Pre-commit hooks

Configured in [.pre-commit-config.yaml](.pre-commit-config.yaml): `ruff --fix`, `ruff-format`, `mypy`.

```bash
uv run pre-commit install            # one-time: register the git hook
uv run pre-commit run --all-files    # run all hooks now
uv run pre-commit autoupdate         # bump hook versions
```

Hooks run automatically on `git commit`. If a hook rewrites files (ruff `--fix`, ruff-format), the commit aborts — re-stage and commit again.

### About the mypy hook

The mypy hook is a `local` / `language: system` hook that calls `uv run mypy benchmark/` directly. This is intentional:

- It uses the project `.venv`, so it sees `numpy`, `pandas`, `torch`, etc. — the same environment as `uv run mypy` on the CLI. The default mirror-mypy hook runs in an isolated venv with no project deps and silently turns every third-party symbol into `Any`, which produces different errors than the CLI.
- It always checks the whole `benchmark/` package (`pass_filenames: false`), not just the staged files. Avoids cross-file inference differences between a partial and a full check.

Trade-off: `uv` must be on `PATH` for anyone running pre-commit. If you don't have `uv` installed, follow [the uv install guide](https://docs.astral.sh/uv/getting-started/installation/) before `pre-commit install`.

### Why `ruff check` passes locally but the hook fails

`ruff check` is the **linter**; `ruff format` is the **formatter** — separate tools. The hook runs both. To match it locally:

```bash
uv run ruff format --check benchmark/
```

## Git workflow

### Branch naming

```
feature/<description>    # new feature
fix/<description>        # bug fix
refactor/<description>   # refactoring with no behavior change
chore/<description>      # maintenance (deps, config, CI, tooling)
```

Never commit directly to `main`.

### Conventional Commits

Format: `<type>(<optional scope>): <imperative description>`

```
feat: add guided-filter postprocessor
fix(cca): handle empty frame in connected-component analysis
refactor(pipeline): move frame-skip logic into skip_strategies
chore: upgrade pre-commit to 4.6.1
docs: document mask data contract
test: add edge cases for TemporalSmoothing
perf: vectorize EMA computation on GPU
```

Valid types: `feat`, `fix`, `refactor`, `chore`, `docs`, `test`, `perf`.

## Making a clean commit

1. Inspect the working tree:
   ```bash
   git status
   git diff
   ```
2. Run the full check locally before staging — same checks as the hooks:
   ```bash
   uv run ruff check --fix .
   uv run ruff format .
   uv run mypy benchmark/ pipeline/
   ```
3. Stage only the files you want — avoid `git add -A` (sweeps `dataset/`, `weights/`, `output/` if they leak past `.gitignore`):
   ```bash
   git add benchmark/metrics.py benchmark/runner.py
   ```
4. Commit using Conventional Commits. The pre-commit hooks run here:
   ```bash
   git commit -m "fix(metrics): narrow None-check before flow warping"
   ```
   If a hook fails, fix the issue, re-stage, and run `git commit` again — do **not** use `--amend` after a failed hook (the previous commit didn't include your changes).
5. Push:
   ```bash
   git push
   ```

## Gitignored paths

`dataset/`, `weights/`, `output/`, `temp_results/`, `.venv/` — never commit these. Heavy binaries belong outside git.

## Troubleshooting

- **`uv sync` warns about a `VIRTUAL_ENV` mismatch** — harmless if you have a system venv active. To silence it: `deactivate`, or pass `--active` to target it explicitly.
- **mypy complains after upgrading numpy / opencv** — stubs change between versions. Re-run `uv sync` to align with `uv.lock`, then re-check.
- **pre-commit is slow on first run** — it's building hook environments under `~/.cache/pre-commit`. Subsequent runs are cached.
