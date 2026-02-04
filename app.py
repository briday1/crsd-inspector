"""
CRSD Inspector - Enhanced Streamlit app for comprehensive CRSD file diagnostics
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tempfile
import os
from datetime import datetime

# Constants
EPSILON = 1e-10  # Small value to prevent log(0) and division by zero

try:
    import dagex
    from dagex import Graph
except ImportError:
    st.error("dagex is not installed. Please run: pip install dagex")
    st.stop()

try:
    from sarkit.crsd import Reader as CRSDReader
    sarkit_available = True
except ImportError:
    try:
        from sarpy.io.complex.converter import open_complex as sarpy_open
        sarkit_available = False
    except ImportError:
        st.error("Neither sarkit nor sarpy is installed. Please run: pip install sarkit")
        st.stop()


st.set_page_config(page_title="CRSD Inspector", page_icon="📡", layout="wide")

st.title("📡 CRSD Inspector - Comprehensive Diagnostics")
st.markdown("""
This application provides **comprehensive diagnostics** for **CRSD** (Compensated Radar Signal Data) files
using **dagex** workflow orchestration and **Plotly** interactive visualizations.
""")


def load_crsd_file(file_path):
    """Load CRSD file node for dagex workflow"""
    def loader(_inputs):
        try:
            if sarkit_available:
                crsd_obj = CRSDReader(file_path)
            else:
                crsd_obj = sarpy_open(file_path)
            
            return {
                "crsd_object": crsd_obj,
                "file_path": file_path,
                "file_size": os.path.getsize(file_path)
            }
        except Exception as e:
            st.error(f"Error loading CRSD file: {e}")
            return {}
    return loader


def extract_comprehensive_metadata(inputs):
    """Extract comprehensive metadata from CRSD file"""
    crsd_obj = inputs.get("crsd_object")
    if crsd_obj is None:
        return {}
    
    metadata_dict = {}
    
    try:
        # Basic file info
        metadata_dict["file_info"] = {
            "path": inputs.get("file_path", "Unknown"),
            "size_bytes": inputs.get("file_size", 0),
            "size_mb": inputs.get("file_size", 0) / (1024 * 1024)
        }
        
        # Get CRSD metadata
        if hasattr(crsd_obj, 'crsd_meta'):
            meta = crsd_obj.crsd_meta
            
            # Collection Info
            if hasattr(meta, 'CollectionInfo'):
                coll_info = meta.CollectionInfo
                metadata_dict["collection"] = {
                    "collector_name": getattr(coll_info, 'CollectorName', 'N/A'),
                    "core_name": getattr(coll_info, 'CoreName', 'N/A'),
                    "collect_type": getattr(coll_info, 'CollectType', 'N/A'),
                    "classification": getattr(coll_info, 'Classification', 'N/A'),
                }
                if hasattr(coll_info, 'RadarMode'):
                    metadata_dict["collection"]["radar_mode"] = getattr(coll_info.RadarMode, 'ModeID', 'N/A')
            
            # Global parameters
            if hasattr(meta, 'Global'):
                glob = meta.Global
                metadata_dict["global"] = {
                    "collect_start": getattr(glob, 'CollectStart', 'N/A'),
                    "collect_duration": getattr(glob, 'CollectDuration', 'N/A'),
                    "tx_time_1": getattr(glob, 'TxTime1', 'N/A'),
                    "tx_time_2": getattr(glob, 'TxTime2', 'N/A'),
                }
            
            # Channel Info
            if hasattr(meta, 'Channel'):
                channels = meta.Channel
                metadata_dict["channels"] = {
                    "num_channels": len(channels),
                    "channel_ids": [getattr(ch, 'Identifier', f'Channel{i}') for i, ch in enumerate(channels)]
                }
            
            # Data format info
            if hasattr(meta, 'Data'):
                data_meta = meta.Data
                metadata_dict["data_format"] = {
                    "signal_array_format": getattr(data_meta, 'SignalArrayFormat', 'N/A'),
                    "num_bytes_pvp": getattr(data_meta, 'NumBytesPVP', 0),
                }
                if hasattr(data_meta, 'Channel') and len(data_meta.Channel) > 0:
                    ch0 = data_meta.Channel[0]
                    metadata_dict["data_format"]["num_vectors"] = getattr(ch0, 'NumVectors', 0)
                    metadata_dict["data_format"]["num_samples"] = getattr(ch0, 'NumSamples', 0)
        
        # File header info
        if hasattr(crsd_obj, 'file_header'):
            header = crsd_obj.file_header
            metadata_dict["file_header"] = {
                "xml_offset": getattr(header, 'xml_section_byte_offset', 'N/A'),
                "pvp_offset": getattr(header, 'pvp_section_byte_offset', 'N/A'),
                "signal_offset": getattr(header, 'signal_section_byte_offset', 'N/A'),
            }
        
        return {
            "metadata": metadata_dict,
            "crsd_object": crsd_obj
        }
    except Exception as e:
        return {
            "metadata": {"error": str(e)},
            "crsd_object": crsd_obj
        }


def read_signal_data(inputs):
    """Read signal data from CRSD file"""
    crsd_obj = inputs.get("crsd_object")
    if crsd_obj is None:
        return {}
    
    try:
        # Try to read signal array
        if hasattr(crsd_obj, 'read_signal_block'):
            try:
                # Read a reasonably sized block for visualization
                signal_block = crsd_obj.read_signal_block(0, 0, 512, 512)
                return {
                    "signal_data": signal_block,
                    "shape": signal_block.shape,
                    "data_source": "real"
                }
            except Exception:
                try:
                    signal_block = crsd_obj.read_signal_block(0, 0, 256, 256)
                    return {
                        "signal_data": signal_block,
                        "shape": signal_block.shape,
                        "data_source": "real"
                    }
                except:
                    pass
        
        if hasattr(crsd_obj, '__getitem__'):
            signal_block = np.array(crsd_obj[0][:512, :512])
            return {
                "signal_data": signal_block,
                "shape": signal_block.shape,
                "data_source": "real"
            }
        
        # Fallback: generate synthetic data
        sample_data = generate_synthetic_signal(256, 256)
        return {
            "signal_data": sample_data,
            "shape": sample_data.shape,
            "data_source": "synthetic"
        }
    except Exception as e:
        sample_data = generate_synthetic_signal(256, 256)
        return {
            "signal_data": sample_data,
            "shape": sample_data.shape,
            "data_source": "synthetic_error",
            "error": str(e)
        }


def generate_synthetic_signal(rows, cols):
    """Generate synthetic signal data for demo"""
    signal = np.zeros((rows, cols), dtype=np.complex64)
    
    # Add synthetic targets
    for i in range(3):
        target_row = 80 + i * 50
        target_col = 100 + i * 40
        for dr in range(-5, 6):
            for dc in range(-5, 6):
                r = target_row + dr
                c = target_col + dc
                if 0 <= r < rows and 0 <= c < cols:
                    amp = 10.0 * np.exp(-0.1 * (dr**2 + dc**2))
                    phase = np.random.uniform(-np.pi, np.pi)
                    signal[r, c] += amp * np.exp(1j * phase)
    
    # Add noise
    noise = 0.1 * (np.random.randn(rows, cols) + 1j * np.random.randn(rows, cols))
    signal += noise.astype(np.complex64)
    
    return signal


def compute_signal_statistics(inputs):
    """Compute comprehensive signal statistics"""
    signal_data = inputs.get("signal_data")
    if signal_data is None:
        return {"statistics": None}
    
    try:
        amplitude = np.abs(signal_data)
        phase = np.angle(signal_data)
        
        stats = {
            "amplitude": {
                "min": float(np.min(amplitude)),
                "max": float(np.max(amplitude)),
                "mean": float(np.mean(amplitude)),
                "std": float(np.std(amplitude)),
                "median": float(np.median(amplitude)),
                "percentile_95": float(np.percentile(amplitude, 95)),
                "percentile_99": float(np.percentile(amplitude, 99)),
            },
            "phase": {
                "min": float(np.min(phase)),
                "max": float(np.max(phase)),
                "mean": float(np.mean(phase)),
                "std": float(np.std(phase)),
            },
            "complex": {
                "real_mean": float(np.mean(signal_data.real)),
                "imag_mean": float(np.mean(signal_data.imag)),
                "real_std": float(np.std(signal_data.real)),
                "imag_std": float(np.std(signal_data.imag)),
            },
            "quality": {
                # Simplified SNR estimation: ratio of mean to std deviation
                # Note: This is an approximation and may not represent true SNR
                # as it doesn't account for actual noise floor measurements
                "snr_estimate_db": 20 * np.log10(np.mean(amplitude) / (np.std(amplitude) + EPSILON)),
                "dynamic_range_db": 20 * np.log10(np.max(amplitude) / (np.min(amplitude) + EPSILON)),
            }
        }
        
        return {
            "statistics": stats,
            "amplitude": amplitude,
            "phase": phase
        }
    except Exception as e:
        return {"statistics": None, "error": str(e)}


def create_crsd_workflow(file_path):
    """Create enhanced dagex workflow for CRSD processing"""
    graph = Graph()
    
    # Node 0: Load CRSD file
    graph.add(
        load_crsd_file(file_path),
        label="Load CRSD",
        inputs=None,
        outputs=[("crsd_object", "crsd"), ("file_path", "path"), ("file_size", "size")]
    )
    
    # Node 1: Extract comprehensive metadata
    graph.add(
        extract_comprehensive_metadata,
        label="Extract Metadata",
        inputs=[("crsd", "crsd_object"), ("path", "file_path"), ("size", "file_size")],
        outputs=[("metadata", "meta"), ("crsd_object", "crsd_pass")]
    )
    
    # Node 2: Read signal data
    graph.add(
        read_signal_data,
        label="Read Signal",
        inputs=[("crsd_pass", "crsd_object")],
        outputs=[("signal_data", "signal"), ("shape", "signal_shape"), ("data_source", "source")]
    )
    
    # Node 3: Compute comprehensive statistics
    graph.add(
        compute_signal_statistics,
        label="Compute Statistics",
        inputs=[("signal", "signal_data")],
        outputs=[("statistics", "stats"), ("amplitude", "amp"), ("phase", "phase_out")]
    )
    
    return graph


def create_plotly_amplitude_plot(amplitude):
    """Create interactive Plotly amplitude plot"""
    amp_db = 20 * np.log10(np.abs(amplitude) + EPSILON)
    
    fig = go.Figure(data=go.Heatmap(
        z=amp_db,
        colorscale='Gray',
        colorbar=dict(title="Amplitude (dB)"),
        hovertemplate='Vector: %{y}<br>Sample: %{x}<br>Amplitude: %{z:.2f} dB<extra></extra>'
    ))
    
    fig.update_layout(
        title="Signal Amplitude (dB)",
        xaxis_title="Range Sample",
        yaxis_title="Azimuth Vector",
        height=600,
        hovermode='closest'
    )
    
    return fig


def create_plotly_phase_plot(phase):
    """Create interactive Plotly phase plot"""
    fig = go.Figure(data=go.Heatmap(
        z=phase,
        colorscale='HSV',
        colorbar=dict(title="Phase (rad)"),
        hovertemplate='Vector: %{y}<br>Sample: %{x}<br>Phase: %{z:.3f} rad<extra></extra>',
        zmid=0
    ))
    
    fig.update_layout(
        title="Signal Phase",
        xaxis_title="Range Sample",
        yaxis_title="Azimuth Vector",
        height=600,
        hovermode='closest'
    )
    
    return fig


def create_histogram_plots(amplitude, phase):
    """Create amplitude and phase histograms"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Amplitude Distribution", "Phase Distribution")
    )
    
    # Amplitude histogram
    fig.add_trace(
        go.Histogram(x=amplitude.flatten(), nbinsx=100, name="Amplitude", showlegend=False),
        row=1, col=1
    )
    
    # Phase histogram
    fig.add_trace(
        go.Histogram(x=phase.flatten(), nbinsx=100, name="Phase", showlegend=False),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Amplitude", row=1, col=1)
    fig.update_xaxes(title_text="Phase (rad)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig


def create_signal_profiles(amplitude):
    """Create signal profile plots (range and azimuth cuts)"""
    mid_row = amplitude.shape[0] // 2
    mid_col = amplitude.shape[1] // 2
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Azimuth Profile (Mid-Range)", "Range Profile (Mid-Azimuth)")
    )
    
    # Azimuth profile
    fig.add_trace(
        go.Scatter(y=20 * np.log10(amplitude[:, mid_col] + EPSILON), mode='lines', name="Azimuth"),
        row=1, col=1
    )
    
    # Range profile
    fig.add_trace(
        go.Scatter(x=np.arange(amplitude.shape[1]), 
                   y=20 * np.log10(amplitude[mid_row, :] + EPSILON), 
                   mode='lines', name="Range"),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Vector Index", row=1, col=1)
    fig.update_xaxes(title_text="Sample Index", row=1, col=2)
    fig.update_yaxes(title_text="Amplitude (dB)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude (dB)", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig


# Sidebar for file selection
st.sidebar.header("📁 Select CRSD File")

# Option to use example file or upload
file_source = st.sidebar.radio(
    "File Source",
    ["Upload File", "Use Example File"],
    index=0
)

uploaded_file = None
example_file_path = None

if file_source == "Upload File":
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CRSD file", 
        type=['crsd', 'nitf', 'ntf']
    )
elif file_source == "Use Example File":
    example_path = os.path.join(os.path.dirname(__file__), "examples", "example_small.crsd")
    if os.path.exists(example_path):
        example_file_path = example_path
        st.sidebar.success(f"✅ Using example file")
    else:
        st.sidebar.info("Example file not found. Generate synthetic data or upload a file.")

# Process file
if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.crsd') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    
    st.success(f"✅ Uploaded: {uploaded_file.name}")
    file_to_process = tmp_file_path
    
elif example_file_path is not None:
    file_to_process = example_file_path
else:
    file_to_process = None

if file_to_process:
    # Create and visualize the workflow
    with st.expander("🔄 View dagex Workflow Structure", expanded=False):
        with st.spinner("Building workflow..."):
            graph = create_crsd_workflow(file_to_process)
            dag = graph.build()
            
            try:
                mermaid_diagram = dag.to_mermaid()
                st.code(mermaid_diagram, language="mermaid")
            except Exception as e:
                st.warning(f"Could not generate Mermaid diagram: {e}")
    
    # Execute the workflow
    st.header("⚡ Execute Workflow and View Diagnostics")
    
    col1, col2 = st.columns(2)
    with col1:
        parallel_exec = st.checkbox("Execute in parallel", value=True)
    with col2:
        num_threads = st.slider("Number of threads", 1, 8, 4)
    
    if st.button("🚀 Run Comprehensive Analysis", type="primary"):
        with st.spinner("Executing workflow and generating diagnostics..."):
            try:
                # Execute the DAG
                if parallel_exec:
                    context = dag.execute(True, num_threads)
                else:
                    context = dag.execute(False, None)
                
                st.success("✅ Workflow executed successfully!")
                
                # Create tabs for different diagnostic views
                tabs = st.tabs([
                    "📊 Overview", 
                    "📈 Amplitude", 
                    "🌊 Phase", 
                    "📉 Statistics",
                    "📐 Profiles",
                    "🔢 Metadata"
                ])
                
                # Tab 1: Overview
                with tabs[0]:
                    st.subheader("File Overview")
                    
                    metadata = context.get("meta")
                    if metadata and "file_info" in metadata:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("File Size", f"{metadata['file_info']['size_mb']:.2f} MB")
                        with col2:
                            signal_shape = context.get("signal_shape")
                            if signal_shape:
                                st.metric("Data Shape", f"{signal_shape[0]} × {signal_shape[1]}")
                        with col3:
                            source = context.get("source", "unknown")
                            st.metric("Data Source", source.title())
                    
                    # Quick stats
                    stats = context.get("stats")
                    if stats:
                        st.subheader("Quick Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mean Amplitude", f"{stats['amplitude']['mean']:.3f}")
                        with col2:
                            st.metric("Max Amplitude", f"{stats['amplitude']['max']:.3f}")
                        with col3:
                            st.metric("SNR Estimate", f"{stats['quality']['snr_estimate_db']:.1f} dB")
                        with col4:
                            st.metric("Dynamic Range", f"{stats['quality']['dynamic_range_db']:.1f} dB")
                
                # Tab 2: Amplitude Visualization
                with tabs[1]:
                    st.subheader("Amplitude Visualization (dB)")
                    amp = context.get("amp")
                    if amp is not None:
                        fig = create_plotly_amplitude_plot(amp)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download option
                        if st.button("💾 Download Amplitude Data"):
                            amp_db = 20 * np.log10(np.abs(amp) + EPSILON)
                            np.save("amplitude_db.npy", amp_db)
                            st.success("Saved to amplitude_db.npy")
                    else:
                        st.info("No amplitude data available")
                
                # Tab 3: Phase Visualization
                with tabs[2]:
                    st.subheader("Phase Visualization")
                    phase = context.get("phase_out")
                    if phase is not None:
                        fig = create_plotly_phase_plot(phase)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No phase data available")
                
                # Tab 4: Statistics
                with tabs[3]:
                    st.subheader("Statistical Analysis")
                    stats = context.get("stats")
                    if stats:
                        # Display statistics in organized sections
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### Amplitude Statistics")
                            st.json(stats['amplitude'])
                            
                            st.markdown("### Quality Metrics")
                            st.json(stats['quality'])
                        
                        with col2:
                            st.markdown("### Phase Statistics")
                            st.json(stats['phase'])
                            
                            st.markdown("### Complex Signal Statistics")
                            st.json(stats['complex'])
                        
                        # Histograms
                        st.markdown("### Distribution Histograms")
                        amp = context.get("amp")
                        phase = context.get("phase_out")
                        if amp is not None and phase is not None:
                            fig = create_histogram_plots(amp, phase)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No statistics available")
                
                # Tab 5: Profiles
                with tabs[4]:
                    st.subheader("Signal Profiles")
                    amp = context.get("amp")
                    if amp is not None:
                        fig = create_signal_profiles(amp)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("""
                        **Profile Analysis:**
                        - Left: Amplitude variation along azimuth (at mid-range)
                        - Right: Amplitude variation along range (at mid-azimuth)
                        - Look for peaks indicating strong targets or features
                        """)
                    else:
                        st.info("No amplitude data available")
                
                # Tab 6: Metadata
                with tabs[5]:
                    st.subheader("CRSD Metadata")
                    metadata = context.get("meta")
                    if metadata:
                        for key, value in metadata.items():
                            with st.expander(f"📋 {key.replace('_', ' ').title()}"):
                                st.json(value)
                    else:
                        st.info("No metadata available")
                
            except Exception as e:
                st.error(f"Error executing workflow: {e}")
                st.exception(e)
    
    # Clean up temp file if it was created
    try:
        if uploaded_file is not None:
            os.unlink(tmp_file_path)
    except:
        pass

else:
    st.info("👈 Please select or upload a CRSD file using the sidebar to begin comprehensive inspection")
    
    # Show demo information
    st.header("📋 Comprehensive Diagnostics Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Visualizations
        - 🎨 Interactive Plotly plots with zoom/pan
        - 📊 2D amplitude heatmaps (dB scale)
        - 🌊 2D phase heatmaps
        - 📈 Distribution histograms
        - 📐 Azimuth and range profiles
        """)
    
    with col2:
        st.markdown("""
        ### Diagnostics
        - 📏 Comprehensive signal statistics
        - 🎯 SNR and dynamic range estimation
        - 📋 Full metadata extraction
        - 🔢 File format details
        - ⚡ Quality metrics
        """)
    
    st.markdown("""
    ### dagex Workflow
    The analysis uses a 4-node DAG:
    1. **Load CRSD**: Reads file using sarkit
    2. **Extract Metadata**: Comprehensive metadata parsing
    3. **Read Signal**: Signal data block extraction
    4. **Compute Statistics**: Parallel statistical analysis
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
**CRSD Inspector v2.0**
- **dagex**: Workflow orchestration
- **sarkit**: CRSD file I/O
- **Plotly**: Interactive visualizations
- **streamlit**: Web interface
""")
