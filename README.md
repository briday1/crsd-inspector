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
crsd-inspector list
crsd-inspector show-params --workflow signal_analysis

# Workflow-only params (no separate initialization stage)
crsd-inspector execute \
  --workflow signal_analysis \
  --param crsd_directory=examples \
  --param crsd_file=uniform_prf_1target_1ch_external.crsd

crsd-inspector run
```

## Architecture

`crsd_inspector` contains:
- `workflows/`: domain workflow definitions (`run_workflow` + `workflow.params`)
- `workflows/util/input_loader.py`: shared CRSD file loading from workflow params
- `renderflow.py`: minimal provider contract (`APP_NAME`, `WORKFLOWS_PACKAGE`)

The package no longer defines its own renderer/runtime layer and does not use a separate initialization stage.

## Provider Registration

`pyproject.toml` registers this package under:

`[project.entry-points."renderflow.providers"]`

`crsd-inspector = "crsd_inspector.renderflow:get_app_spec"`
