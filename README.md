# CRSD Inspector

A Streamlit web application for comprehensive inspection and visualization of CRSD (Compensated Radar Signal Data) files using dagex workflow orchestration and Plotly interactive visualizations.

## Features

### 🔄 dagex Workflow Orchestration
- 4-node DAG pipeline for parallel CRSD processing
- Load → Extract Metadata → Read Signal → Compute Statistics
- Configurable parallel execution with thread control

### 📊 Interactive Plotly Visualizations
- **2D Amplitude Heatmaps**: Interactive dB-scale signal amplitude with zoom/pan
- **2D Phase Heatmaps**: Complex phase visualization with HSV colormap
- **Distribution Histograms**: Amplitude and phase statistical distributions
- **Signal Profiles**: Azimuth and range cut analysis
- **Hover Details**: Pixel-level inspection with coordinates and values

### 🔍 Comprehensive Diagnostics
- **File Information**: Size, format version, data dimensions
- **Signal Statistics**: Min, max, mean, std, median, percentiles
- **Quality Metrics**: SNR estimation, dynamic range analysis
- **Channel Information**: Multi-channel support and details
- **Metadata Extraction**: Full XML metadata parsing
- **Collection Details**: Platform, geometry, timing information

### 📈 Multiple Diagnostic Views
- **Overview Tab**: Quick metrics and file summary
- **Amplitude Tab**: Interactive signal amplitude visualization
- **Phase Tab**: Complex phase analysis
- **Statistics Tab**: Comprehensive statistical analysis with histograms
- **Profiles Tab**: Azimuth and range profile cuts
- **Metadata Tab**: Detailed CRSD metadata inspection

## Installation

1. Clone this repository:
```bash
git clone https://github.com/briday1/crsd-inspector.git
cd crsd-inspector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Using the Application

1. **Select a File Source**:
   - Upload your own CRSD file (supports .crsd, .nitf, .ntf formats)
   - Use an example file (if available in `examples/` directory)
   - App will generate synthetic data for demonstration if no file is provided

2. **Configure Execution**:
   - Choose parallel or sequential execution
   - Adjust number of threads for parallel processing

3. **Run Analysis**:
   - Click "Run Comprehensive Analysis" button
   - Explore results across multiple diagnostic tabs

4. **Interactive Exploration**:
   - Zoom and pan in Plotly visualizations
   - Hover over data points for detailed information
   - Download data arrays for further analysis

## Example Files

To generate a synthetic example CRSD file:

```bash
cd examples
python3 create_example_crsd.py
```

This creates a small (256×256) synthetic CRSD file with realistic characteristics for testing.

## How It Works

The application uses **dagex** (a DAG execution engine) to orchestrate the CRSD processing workflow:

```python
graph = Graph()
graph.add(load_crsd_file(path), label="Load CRSD", ...)
graph.add(extract_metadata, label="Extract Metadata", ...)
graph.add(read_signal_data, label="Read Signal", ...)
graph.add(compute_statistics, label="Compute Statistics", ...)

dag = graph.build()
context = dag.execute(parallel=True, num_threads=4)
```

The workflow nodes can execute in parallel where dependencies allow, significantly improving performance for large files.

## Dependencies

- **dagex** (>=2026.1): DAG execution engine (from github.com/briday1/graph-sp, installed as `dagex` on PyPI)
- **sarkit** (>=2024.1): Library for reading/writing CRSD files (NGA Standard 0080)
- **streamlit** (>=1.31.0): Web framework for interactive applications
- **plotly** (>=5.18.0): Interactive visualization library
- **numpy** (>=1.24.0): Numerical computations
- **matplotlib** (>=3.7.0): Additional plotting support

## CRSD File Format

CRSD (Compensated Radar Signal Data) is a standardized format for Synthetic Aperture Radar (SAR) signal data defined by NGA.STND.0080. It contains:
- **Signal Data**: Complex-valued radar signal samples
- **XML Metadata**: Collection parameters, platform information, geometry
- **PVP Arrays**: Per-Vector Parameters for each pulse
- **Support Arrays**: Calibration and auxiliary data

## Architecture

### Workflow Nodes

1. **Load CRSD**: Opens file using sarkit Reader, extracts file information
2. **Extract Metadata**: Parses comprehensive CRSD metadata (collection, global params, channels, etc.)
3. **Read Signal**: Reads signal data block with configurable size
4. **Compute Statistics**: Calculates amplitude/phase statistics, quality metrics (runs in parallel)

### Visualization Pipeline

- Amplitude data → log scale (dB) → Plotly heatmap
- Phase data → radian scale → Plotly heatmap with HSV colormap
- Statistics → multiple views (distributions, profiles, metrics)

## Screenshots

![CRSD Inspector - Main Interface](https://github.com/user-attachments/assets/a6d50c51-6776-429a-93d7-60ab1aea3c10)

## License

See LICENSE file for details.