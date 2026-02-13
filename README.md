# CRSD Inspector

CRSD Inspector defines CRSD workflows and uses `renderflow` for runtime/rendering.

## Install

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ~/git/renderflow
uv pip install -e .
```

## Run

```bash
# Provider-scoped CLI (delegates to renderflow)
crsd-inspector list
crsd-inspector show-params --workflow signal_analysis
crsd-inspector execute --workflow signal_analysis --init crsd_directory=examples

# GUI (renderflow-backed)
crsd-inspector-gui

# Equivalent direct renderflow calls
renderflow run --provider crsd-inspector
renderflow list-workflows --provider crsd-inspector
```

## Architecture

`crsd_inspector` contains:
- `workflows/`: domain workflow definitions (`run_workflow` + `workflow.params`)
- `renderflow.py`: minimal provider contract (`initialize`, `INIT_PARAMS`, `WORKFLOWS_PACKAGE`)
- `cli.py`: thin wrapper over `renderflow.cli`
- `gui.py`: GUI bridge to renderflow Streamlit renderer

The package no longer defines its own renderer/runtime layer.

## Provider Registration

`pyproject.toml` registers this package under:

`[project.entry-points."renderflow.providers"]`

`crsd-inspector = "crsd_inspector.renderflow:get_app_spec"`
