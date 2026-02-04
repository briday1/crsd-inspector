# CRSD Inspector

A Streamlit web application for inspecting and visualizing CRSD (Compensated Radar Signal Data) files using dagex workflow orchestration.

## Features

- 📡 Load and inspect CRSD files
- 🔄 Process data through a dagex DAG workflow
- 📊 Visualize metadata, amplitude, and phase information
- ⚡ Parallel execution support for faster processing
- 📈 Interactive visualizations with matplotlib

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

## How It Works

The application uses **dagex** (a DAG execution engine) to create a processing workflow:

1. **Load CRSD**: Reads the CRSD file using sarkit
2. **Extract Metadata**: Parses XML metadata and collection information
3. **Read Signal**: Reads a block of signal data from the file
4. **Compute Amplitude**: Calculates amplitude from complex signal data
5. **Compute Phase**: Calculates phase from complex signal data

The amplitude and phase computations run in parallel, demonstrating dagex's parallel execution capabilities.

## Dependencies

- **dagex**: DAG execution engine (installed as `dagex` from PyPI, source at github.com/briday1/graph-sp)
- **sarkit**: Library for reading CRSD files
- **streamlit**: Web framework for the interface
- **numpy**: Numerical computations
- **matplotlib**: Visualization

## CRSD File Format

CRSD (Compensated Radar Signal Data) is a standardized format for Synthetic Aperture Radar (SAR) signal data. It contains:
- Complex-valued signal samples
- XML metadata describing the radar collection
- Platform and acquisition parameters

## License

See LICENSE file for details.