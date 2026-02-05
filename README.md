# CRSD Inspector

A comprehensive radar signal analysis toolkit for inspecting and visualizing CRSD (Compensated Radar Signal Data) files. Features a Streamlit web interface with dagex workflow orchestration, interactive Plotly visualizations, and a CLI for file generation.

## Features

### CLI Tool
- **`crsd-inspector`**: Launch the Streamlit app instantly
- **`crsd-inspector generate`**: Create realistic test CRSD files with configurable scenes
- pip-installable package with proper entry points

### Workflow-Based Architecture
- **Modular Processing**: Self-contained workflow modules for different analyses
- **Pluggable Design**: Workflows discover and register automatically
- **Clean Separation**: App.py is a pure dispatcher - all logic in workflows
- **Simple Interface**: Workflows run code and return results - no framework required

### Analysis Workflows
- **Basic Statistics**: Comprehensive signal statistics with amplitude/phase heatmaps
- **Quality Assessment**: SNR estimation, dynamic range, clipping detection, histogram plots
- **Range-Doppler Processing**: 2D FFT analysis with matched filtering support

### Professional CRSD Support
- **Real CRSD Files**: Full support via sarkit (NGA.STND.0080 specification)
- **Multi-Channel**: Handle files with multiple receiver channels
- **TX Waveforms**: Extract and use transmit waveforms for matched filtering
- **Metadata Extraction**: Sample rate, PRF, bandwidth, and other radar parameters
- **Support Arrays**: Access calibration data and auxiliary information

### Interactive Visualizations
- **Plotly Integration**: All plots generated as interactive Figure objects
- **Workflow Output**: Plots returned directly from workflows (not assembled in app)
- **Zoom/Pan/Hover**: Full interactivity with pixel-level inspection
- **Multiple Views**: Heatmaps, histograms, profiles, and range-Doppler maps

## Installation

Install as a package with pip:

```bash
pip install -e .
```

Or for development:

```bash
git clone https://github.com/yourusername/crsd-inspector.git
cd crsd-inspector
pip install -e .
```

## Usage

### Launch the App

Simply run:

```bash
crsd-inspector
```

The Streamlit app will open in your browser at `http://localhost:8501`.

### Generate Example CRSD Files

Create realistic test files with configurable radar scenes:

```bash
crsd-inspector generate
```

This generates 3 example files in `./examples/`:
- **example_1.crsd**: Simple scene (3 targets, 1 channel)
- **example_2.crsd**: Complex scene (5 targets, 1 channel)  
- **example_3.crsd**: Multi-channel (2 targets, 2 channels)

Generate to a custom directory:

```bash
crsd-inspector generate --output-dir /path/to/output
```

### Using the Application

1. **Select CRSD File**: Use the dropdown to select from available files in `examples/`
2. **Choose Workflow**: Select an analysis workflow (Basic Stats, Quality Assessment, Range-Doppler)
3. **View Results**: Explore interactive plots and statistics tables
4. **Navigate**: Use ◀ ↻ ▶ buttons to move between files

## Architecture

### Package Structure

```
crsd-inspector/
├── pyproject.toml              # Package configuration
├── README.md                   # This file
├── examples/                   # Generated CRSD files
│   ├── example_1.crsd
│   ├── example_2.crsd
│   └── example_3.crsd
└── crsd_inspector/             # Main package
    ├── __init__.py
    ├── cli.py                  # CLI entry point
    ├── app.py                  # Streamlit app (pure dispatcher)
    ├── generate.py             # CRSD file generator
    └── workflows/              # Analysis workflows
        ├── basic_stats.py      # Signal statistics + plots
        ├── quality_assessment.py  # Quality metrics + histograms
        └── range_doppler.py    # 2D FFT processing
```

### Workflow Pattern

Each workflow module is self-contained and requires only a single function:

```python
# workflows/my_workflow.py
WORKFLOW_NAME = "My Workflow"
WORKFLOW_DESCRIPTION = "What it does"

def run_workflow(signal_data, metadata=None):
    """Run analysis and return formatted results"""
    # Process the signal data
    result_value = compute_something(signal_data)
    
    # Create visualizations
    fig = create_plot(result_value)
    
    # Return results dictionary
    return {
        'tables': [{
            'title': 'Results',
            'data': {'metric': result_value}
        }],
        'plots': [fig],        # Plotly Figure objects
        'text': ["## Analysis complete"]  # Markdown strings
    }
```

**Key Points:**
- Workflows can use any processing approach (simple functions, dagex graphs, etc.)
- Must return a dictionary with `tables`, `plots`, and `text` keys
- Tables are dicts with `'title'` and `'data'` (dict of metric: value pairs)
- Plots are Plotly Figure objects
- Text entries are markdown strings
- The app discovers and loads workflows automatically - just drop a new module in `workflows/`

### CRSD Generator

The generator creates realistic synthetic CRSD files conforming to NGA.STND.0080:

- **TX Waveforms**: LFM chirps with configurable time-bandwidth product
- **Point Targets**: Delayed waveform + Doppler modulation for realistic returns
- **Multi-Channel**: Independent noise per channel
- **Full Metadata**: Sample rate, PRF, bandwidth stored in file header
- **Support Arrays**: TX waveform available for matched filtering

```python
from crsd_inspector.generate import SceneConfig, RadarTarget, CRSDGenerator

scene = SceneConfig(
    num_pulses=256,
    samples_per_pulse=512,
    sample_rate_hz=100e6,
    prf_hz=1000.0,
    bandwidth_hz=10e6,
    targets=[
        RadarTarget(range_m=3000, doppler_hz=50, rcs_dbsm=10, label="Vehicle"),
    ],
    output_file="custom.crsd"
)

generator = CRSDGenerator(scene)
stats = generator.generate()
```

## Dependencies

- **streamlit** (>=1.31.0): Web framework for interactive applications
- **sarkit** (>=1.0.0): Library for reading/writing CRSD files (NGA.STND.0080)
- **numpy** (>=1.24.0): Numerical computations
- **plotly** (>=5.18.0): Interactive visualization library
- **matplotlib** (>=3.7.0): Additional plotting support
- **lxml** (>=4.9.0): XML processing for CRSD metadata
- **dagex** (>=2026.1): Optional - DAG execution engine (used internally by example workflows)

All dependencies are automatically installed when you `pip install` the package.

## CRSD File Format

CRSD (Compensated Radar Signal Data) is a standardized format for Synthetic Aperture Radar (SAR) signal data defined by NGA.STND.0080. It contains:
- **Signal Data**: Complex-valued radar signal samples (multi-channel support)
- **XML Metadata**: Collection parameters, platform information, geometry, radar parameters
- **PVP Arrays**: Per-Vector Parameters for each pulse
- **PPP Arrays**: Per-Pulse Parameters shared across channels
- **Support Arrays**: TX waveforms, calibration data, and auxiliary information

CRSD Inspector supports real CRSD files via sarkit, with full access to TX waveforms and metadata for advanced processing workflows.

## CLI Reference

```bash
# Show help
crsd-inspector --help

# Launch app (default)
crsd-inspector
crsd-inspector app

# Generate examples
crsd-inspector generate
crsd-inspector generate --output-dir ./my-test-files
```

## Development

The package uses a modular workflow architecture. To add a new analysis:

1. Create `crsd_inspector/workflows/my_analysis.py`
2. Define `WORKFLOW_NAME` and `WORKFLOW_DESCRIPTION` constants
3. Implement `run_workflow(signal_data, metadata=None)` that returns results dict
4. The app will auto-discover and load it

Workflow return format:
```python
{
    'tables': [
        {
            'title': 'Statistics',
            'data': {'Mean': 1.23, 'StdDev': 0.45}  # dict of metric: value
        }
    ],
    'plots': [fig1, fig2],  # List of Plotly Figure objects
    'text': ["## Header", "Analysis description"]  # List of markdown strings
}

## License

See LICENSE file for details.
