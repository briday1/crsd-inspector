# CRSD Inspector

CRSD Inspector is a CRSD analysis package that exposes:
- domain CLI commands (`crsd-inspector`)
- workflow/provider definitions for `renderflow`
- a GUI launcher (`crsd-inspector-gui`) that delegates rendering to `renderflow`

## Install

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ~/git/renderflow
uv pip install -e .
```

## Run

```bash
# Domain CLI
crsd-inspector --help

# GUI (renderflow-backed)
crsd-inspector-gui
```

You can also launch renderflow directly:

```bash
renderflow run --provider crsd-inspector
```

## Architecture

`crsd_inspector` contains:
- `cli.py`: domain CLI
- `gui.py`: GUI bridge API (calls renderflow Streamlit renderer)
- `app_definition.py`: renderflow provider contract
- `workflows/`: analysis workflows

This repo no longer contains an in-repo renderer/runtime implementation.

## Provider Registration

`pyproject.toml` registers this package as a renderflow provider:

`[project.entry-points."renderflow.providers"]`

`crsd-inspector = "crsd_inspector.app_definition:get_app_spec"`

## Notes

- CRSD reading/writing is through `sarkit`.
- Workflow execution/processing uses existing workflow modules under `crsd_inspector/workflows/`.
