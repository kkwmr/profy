# Repository Guidelines

This guide helps contributors work consistently across the Profy repository.

## Project Structure & Module Organization
- `src/`: Core Python code
  - `models/`, `training/`, `evaluation/`, `utils/`, `visualization/`
- `configs/`: YAML configs (e.g., `configs/config.yaml`)
- `scripts/`: Execution scripts (only `run_full_experiment.sh`)
- `data -> /home/kazuki/Projects/Profy/data`: Symlink to real dataset
- `results/`: Timestamped experiment outputs and logs
- `webapp/`: FastAPI UI (`main.py`, `static/`)
- `paper/`: LaTeX sources and figures

## Build, Test, and Development Commands
- Environment (Python 3.10):
  - `python -m venv .venv && source .venv/bin/activate`
  - Install app deps as needed (web UI): `pip install -r webapp/requirements.txt`
- Reproducible experiment (end‑to‑end validation):
  - `./scripts/run_full_experiment.sh`
- Run web UI locally:
  - `cd webapp && uvicorn main:app --reload`
- Docker (optional):
  - `docker build -t profy .` (CUDA base; adjust as needed)

## Coding Style & Naming Conventions
- Python: PEP 8, 4‑space indent, type hints when practical, docstrings in English.
- Filenames/modules: `snake_case.py`; classes `CamelCase`; functions/vars `snake_case`.
- Prefer editing existing files over adding new ones.
- Do not create files with names like `*_new.py`, `*_improved.py`, `*_v2.py`.
- Keep `scripts/` limited to the single `run_full_experiment.sh`.

## Testing Guidelines
- No formal unit test suite yet; use the experiment script as the primary E2E test.
- Verify outputs under `results/experiment_YYYYMMDD_HHMMSS/` (figures, logs, model artifacts).
- Spot‑check visualizations in `src/visualization/` and metrics in `evaluation/` outputs.

## Commit & Pull Request Guidelines
- Commit messages: short, imperative, present tense (e.g., “Fix training script”, “Update README”).
- PRs must include: clear description (what/why), linked issues, and relevant result snapshots (e.g., figures, key metrics).
- Run `./scripts/run_full_experiment.sh` before opening a PR; do not commit raw data.

## Security & Configuration Tips
- Use only real Profy data; do not introduce synthetic data.
- Do not modify the `data` symlink target or commit dataset files.
- GPU/CUDA recommended; set `CUDA_VISIBLE_DEVICES` as needed.
- The webapp enables permissive CORS for local dev; restrict origins for deployments.

