# CRSD Workflows

This directory contains dagex workflow definitions for CRSD processing.

## Workflow Structure

Each workflow is a Python module that defines:

1. `create_workflow()` - Returns a dagex Graph
2. `format_results(context)` - Formats workflow outputs for display
3. `WORKFLOW_NAME` - Display name
4. `WORKFLOW_DESCRIPTION` - Short description

## Available Workflows

### basic_stats.py
Simple workflow computing basic amplitude and phase statistics.

### range_doppler.py
2D FFT processing to generate range-doppler maps.

### quality_assessment.py
Comprehensive quality metrics including SNR, dynamic range, phase coherence, and clipping detection.

## Creating Custom Workflows

```python
from dagex import Graph

def create_workflow():
    graph = Graph()
    
    # Add your processing nodes
    graph.add(your_function, label="Step 1", inputs=..., outputs=...)
    
    return graph

def format_results(context):
    return {
        "tables": [{"title": "Results", "data": {...}}],
        "plots": [{"type": "heatmap", "data": array, ...}],
        "text": ["Additional information"]
    }

WORKFLOW_NAME = "My Workflow"
WORKFLOW_DESCRIPTION = "What it does"
```

## Input Data

All workflows receive a dictionary with:
- `signal_data`: Complex numpy array of CRSD signal data
- `metadata`: File metadata dictionary

## Output Format

The `format_results()` function should return a dictionary with:
- `tables`: List of `{title, data}` dictionaries
- `plots`: List of plot specifications
- `text`: List of text strings for additional info
