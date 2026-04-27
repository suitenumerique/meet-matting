# Dev Guide

Quick reference for working on **meet-matting**. Python ≥ 3.11, `uv` as package manager.

## Install

```bash
uv sync                  # runtime deps
uv sync --all-groups     # runtime + dev (ruff, mypy, pre-commit)
```

`uv sync` creates `.venv/` and installs from `uv.lock`. No need to activate the venv — prefix commands with `uv run`.

To add a dependency:

```bash
uv add <package>                # runtime
uv add --group dev <package>    # dev only
```

Commit both `pyproject.toml` and `uv.lock`.

## Run the benchmark

```bash
uv run streamlit run benchmark/dashboard.py     # web UI
uv run python -m benchmark.run_benchmark --help # CLI
```

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

## Making a clean commit

1. Inspect the working tree:
   ```bash
   git status
   git diff
   ```
2. Run the full check locally before staging — same checks as the hooks:
   ```bash
   uv run ruff check --fix benchmark/
   uv run ruff format benchmark/
   uv run mypy benchmark/
   ```
3. Stage only the files you want — avoid `git add -A` (sweeps `dataset/`, `weights/`, `output/` if they leak past `.gitignore`):
   ```bash
   git add benchmark/metrics.py benchmark/runner.py
   ```
4. Commit. The pre-commit hooks run here:
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
