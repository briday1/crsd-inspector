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
renderflow list-workflows --provider crsd-inspector
renderflow show-params --provider crsd-inspector --workflow signal_analysis
renderflow execute --provider crsd-inspector --workflow signal_analysis --init crsd_directory=examples
renderflow run --provider crsd-inspector
```

## Architecture

`crsd_inspector` contains:
- `workflows/`: domain workflow definitions (`run_workflow` + `workflow.params`)
- `renderflow.py`: minimal provider contract (`initialize`, `INIT_PARAMS`, `WORKFLOWS_PACKAGE`)

The package no longer defines its own renderer/runtime layer.

## Provider Registration

`pyproject.toml` registers this package under:

`[project.entry-points."renderflow.providers"]`

`crsd-inspector = "crsd_inspector.renderflow:get_app_spec"`
