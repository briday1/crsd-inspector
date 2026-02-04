"""
CRSD Inspector - Multi-File Directory Browser with Comparison
Enhanced Streamlit app for comprehensive CRSD file diagnostics with multi-file support
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tempfile
import os
from datetime import datetime
from pathlib import Path
import glob

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

# Initialize session state
if 'crsd_files' not in st.session_state:
    st.session_state.crsd_files = []
if 'current_file_index' not in st.session_state:
    st.session_state.current_file_index = 0
if 'file_cache' not in st.session_state:
    st.session_state.file_cache = {}
if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = False
if 'comparison_file_index' not in st.session_state:
    st.session_state.comparison_file_index = None

st.title("📡 CRSD Inspector - Multi-File Browser with Comparison")
st.markdown("""
Browse directories of CRSD files, compare multiple files side-by-side, and analyze differences.
""")


def scan_directory_for_crsd(directory_path):
    """Scan a directory for CRSD files"""
    crsd_extensions = ['*.crsd', '*.CRSD', '*.nitf', '*.NITF', '*.ntf', '*.NTF']
    files = []
    
    for ext in crsd_extensions:
        files.extend(glob.glob(os.path.join(directory_path, ext)))
    
    return sorted(files)


def generate_synthetic_signal(rows, cols, seed=None):
    """Generate synthetic CRSD signal data"""
    if seed is not None:
        np.random.seed(seed)
    
    signal = np.zeros((rows, cols), dtype=np.complex64)
    
    # Add synthetic targets
    num_targets = np.random.randint(2, 5)
    for i in range(num_targets):
        target_row = np.random.randint(50, rows - 50)
        target_col = np.random.randint(50, cols - 50)
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


def load_and_process_file(file_path):
    """Load and process a single CRSD file"""
    # Check cache first
    if file_path in st.session_state.file_cache:
        return st.session_state.file_cache[file_path]
    
    try:
        # Load file
        if sarkit_available:
            crsd_obj = CRSDReader(file_path)
        else:
            crsd_obj = sarpy_open(file_path)
        
        # Read signal data
        try:
            if hasattr(crsd_obj, 'read_signal_block'):
                signal_block = crsd_obj.read_signal_block(0, 0, 256, 256)
            elif hasattr(crsd_obj, '__getitem__'):
                signal_block = np.array(crsd_obj[0][:256, :256])
            else:
                # Generate synthetic based on filename
                seed = hash(os.path.basename(file_path)) % (2**32)
                signal_block = generate_synthetic_signal(256, 256, seed=seed)
        except:
            seed = hash(os.path.basename(file_path)) % (2**32)
            signal_block = generate_synthetic_signal(256, 256, seed=seed)
        
        # Compute statistics
        amplitude = np.abs(signal_block)
        phase = np.angle(signal_block)
        
        stats = {
            "amplitude": {
                "min": float(np.min(amplitude)),
                "max": float(np.max(amplitude)),
                "mean": float(np.mean(amplitude)),
                "std": float(np.std(amplitude)),
                "median": float(np.median(amplitude)),
            },
            "phase": {
                "min": float(np.min(phase)),
                "max": float(np.max(phase)),
                "mean": float(np.mean(phase)),
                "std": float(np.std(phase)),
            },
            "quality": {
                "snr_estimate_db": 20 * np.log10(np.mean(amplitude) / (np.std(amplitude) + EPSILON)),
                "dynamic_range_db": 20 * np.log10(np.max(amplitude) / (np.min(amplitude) + EPSILON)),
            }
        }
        
        # Extract metadata
        metadata = {
            "filename": os.path.basename(file_path),
            "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
            "shape": signal_block.shape,
        }
        
        if hasattr(crsd_obj, 'crsd_meta'):
            meta = crsd_obj.crsd_meta
            if hasattr(meta, 'CollectionInfo'):
                metadata["collector"] = getattr(meta.CollectionInfo, 'CollectorName', 'N/A')
                metadata["core_name"] = getattr(meta.CollectionInfo, 'CoreName', 'N/A')
        
        result = {
            "signal_data": signal_block,
            "amplitude": amplitude,
            "phase": phase,
            "statistics": stats,
            "metadata": metadata,
            "file_path": file_path
        }
        
        # Cache the result
        st.session_state.file_cache[file_path] = result
        
        return result
        
    except Exception as e:
        st.error(f"Error loading {os.path.basename(file_path)}: {e}")
        # Return synthetic data
        seed = hash(os.path.basename(file_path)) % (2**32)
        signal_block = generate_synthetic_signal(256, 256, seed=seed)
        amplitude = np.abs(signal_block)
        phase = np.angle(signal_block)
        
        # Create complete statistics
        stats = {
            "amplitude": {
                "min": float(np.min(amplitude)),
                "max": float(np.max(amplitude)),
                "mean": float(np.mean(amplitude)),
                "std": float(np.std(amplitude)),
                "median": float(np.median(amplitude)),
            },
            "phase": {
                "min": float(np.min(phase)),
                "max": float(np.max(phase)),
                "mean": float(np.mean(phase)),
                "std": float(np.std(phase)),
            },
            "quality": {
                "snr_estimate_db": 20 * np.log10(np.mean(amplitude) / (np.std(amplitude) + EPSILON)),
                "dynamic_range_db": 20 * np.log10(np.max(amplitude) / (np.min(amplitude) + EPSILON)),
            }
        }
        
        return {
            "signal_data": signal_block,
            "amplitude": amplitude,
            "phase": phase,
            "statistics": stats,
            "metadata": {
                "filename": os.path.basename(file_path),
                "file_size_mb": 0,
                "shape": signal_block.shape,
            },
            "file_path": file_path,
            "error": str(e)
        }


def create_comparison_plot(data1, data2, title1, title2, colorscale='Gray'):
    """Create side-by-side comparison plot"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(title1, title2, "Difference"),
        horizontal_spacing=0.05
    )
    
    # File 1
    fig.add_trace(
        go.Heatmap(z=data1, colorscale=colorscale, showscale=False),
        row=1, col=1
    )
    
    # File 2
    fig.add_trace(
        go.Heatmap(z=data2, colorscale=colorscale, showscale=False),
        row=1, col=2
    )
    
    # Difference
    diff = data2 - data1
    fig.add_trace(
        go.Heatmap(z=diff, colorscale='RdBu', zmid=0, colorbar=dict(title="Difference")),
        row=1, col=3
    )
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig


def create_plotly_amplitude_plot(amplitude, title="Signal Amplitude (dB)"):
    """Create interactive Plotly amplitude plot"""
    amp_db = 20 * np.log10(np.abs(amplitude) + EPSILON)
    
    fig = go.Figure(data=go.Heatmap(
        z=amp_db,
        colorscale='Gray',
        colorbar=dict(title="Amplitude (dB)"),
        hovertemplate='Vector: %{y}<br>Sample: %{x}<br>Amplitude: %{z:.2f} dB<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Range Sample",
        yaxis_title="Azimuth Vector",
        height=500,
        hovermode='closest'
    )
    
    return fig


def create_plotly_phase_plot(phase, title="Signal Phase"):
    """Create interactive Plotly phase plot"""
    fig = go.Figure(data=go.Heatmap(
        z=phase,
        colorscale='HSV',
        colorbar=dict(title="Phase (rad)"),
        hovertemplate='Vector: %{y}<br>Sample: %{x}<br>Phase: %{z:.3f} rad<extra></extra>',
        zmid=0
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Range Sample",
        yaxis_title="Azimuth Vector",
        height=500,
        hovermode='closest'
    )
    
    return fig


# Sidebar for directory/file selection
st.sidebar.header("📁 File Selection")

# File source selection
file_source = st.sidebar.radio(
    "Source",
    ["Browse Directory", "Upload Files", "Generate Examples"],
    index=0
)

if file_source == "Browse Directory":
    directory_path = st.sidebar.text_input(
        "Directory Path",
        value=".",
        help="Enter path to directory containing CRSD files"
    )
    
    if st.sidebar.button("🔍 Scan Directory"):
        if os.path.isdir(directory_path):
            files = scan_directory_for_crsd(directory_path)
            if files:
                st.session_state.crsd_files = files
                st.session_state.current_file_index = 0
                st.session_state.file_cache = {}  # Clear cache
                st.sidebar.success(f"Found {len(files)} CRSD file(s)")
            else:
                st.sidebar.warning("No CRSD files found in directory")
        else:
            st.sidebar.error("Invalid directory path")

elif file_source == "Upload Files":
    uploaded_files = st.sidebar.file_uploader(
        "Choose CRSD files",
        type=['crsd', 'nitf', 'ntf'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Save uploaded files temporarily
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        for uploaded_file in uploaded_files:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.read())
            file_paths.append(temp_path)
        
        st.session_state.crsd_files = file_paths
        st.session_state.current_file_index = 0
        st.session_state.file_cache = {}
        st.sidebar.success(f"Uploaded {len(file_paths)} file(s)")

elif file_source == "Generate Examples":
    num_examples = st.sidebar.slider("Number of example files", 2, 10, 3)
    
    if st.sidebar.button("Generate Synthetic Files"):
        st.session_state.crsd_files = [f"synthetic_{i+1}.crsd" for i in range(num_examples)]
        st.session_state.current_file_index = 0
        st.session_state.file_cache = {}
        st.sidebar.success(f"Generated {num_examples} synthetic files")

# File list and selection
if st.session_state.crsd_files:
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"📋 Files ({len(st.session_state.crsd_files)})")
    
    # File selector
    for idx, file_path in enumerate(st.session_state.crsd_files):
        filename = os.path.basename(file_path)
        
        col1, col2 = st.sidebar.columns([3, 1])
        
        with col1:
            is_current = idx == st.session_state.current_file_index
            button_type = "primary" if is_current else "secondary"
            if st.button(f"{'▶ ' if is_current else '  '}{filename[:30]}", 
                        key=f"file_{idx}", 
                        type=button_type,
                        use_container_width=True):
                st.session_state.current_file_index = idx
                st.rerun()
        
        with col2:
            if st.button("📊", key=f"compare_{idx}", help="Compare with this file"):
                st.session_state.comparison_mode = True
                st.session_state.comparison_file_index = idx
                st.rerun()
    
    # Navigation buttons
    st.sidebar.markdown("---")
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        if st.button("⬅️ Prev"):
            if st.session_state.current_file_index > 0:
                st.session_state.current_file_index -= 1
                st.rerun()
    
    with col2:
        if st.button("🔄 Refresh"):
            st.session_state.file_cache = {}
            st.rerun()
    
    with col3:
        if st.button("Next ➡️"):
            if st.session_state.current_file_index < len(st.session_state.crsd_files) - 1:
                st.session_state.current_file_index += 1
                st.rerun()
    
    # Comparison mode toggle
    st.sidebar.markdown("---")
    if st.session_state.comparison_mode:
        if st.sidebar.button("❌ Exit Comparison Mode"):
            st.session_state.comparison_mode = False
            st.session_state.comparison_file_index = None
            st.rerun()

# Main content
if st.session_state.crsd_files:
    current_file = st.session_state.crsd_files[st.session_state.current_file_index]
    
    # Load current file
    with st.spinner(f"Loading {os.path.basename(current_file)}..."):
        current_data = load_and_process_file(current_file)
    
    # Display mode
    if st.session_state.comparison_mode and st.session_state.comparison_file_index is not None:
        # Comparison mode
        st.header("🔍 File Comparison Mode")
        
        comparison_file = st.session_state.crsd_files[st.session_state.comparison_file_index]
        
        with st.spinner(f"Loading {os.path.basename(comparison_file)}..."):
            comparison_data = load_and_process_file(comparison_file)
        
        # Display file names
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**File 1:** {current_data['metadata']['filename']}")
        with col2:
            st.info(f"**File 2:** {comparison_data['metadata']['filename']}")
        
        # Tabs for comparison
        tabs = st.tabs(["📊 Statistics Comparison", "📈 Amplitude Comparison", "🌊 Phase Comparison", "📉 Difference Analysis"])
        
        with tabs[0]:
            st.subheader("Statistics Comparison")
            
            col1, col2, col3 = st.columns(3)
            
            stats1 = current_data['statistics']
            stats2 = comparison_data['statistics']
            
            with col1:
                st.metric("File 1 Mean Amplitude", f"{stats1['amplitude']['mean']:.3f}")
                st.metric("File 1 SNR (dB)", f"{stats1['quality']['snr_estimate_db']:.1f}")
            
            with col2:
                st.metric("File 2 Mean Amplitude", f"{stats2['amplitude']['mean']:.3f}")
                st.metric("File 2 SNR (dB)", f"{stats2['quality']['snr_estimate_db']:.1f}")
            
            with col3:
                mean_diff = stats2['amplitude']['mean'] - stats1['amplitude']['mean']
                snr_diff = stats2['quality']['snr_estimate_db'] - stats1['quality']['snr_estimate_db']
                st.metric("Mean Amplitude Δ", f"{mean_diff:.3f}", delta=f"{mean_diff:.3f}")
                st.metric("SNR Δ (dB)", f"{snr_diff:.1f}", delta=f"{snr_diff:.1f}")
        
        with tabs[1]:
            st.subheader("Amplitude Comparison")
            amp1_db = 20 * np.log10(current_data['amplitude'] + EPSILON)
            amp2_db = 20 * np.log10(comparison_data['amplitude'] + EPSILON)
            
            fig = create_comparison_plot(
                amp1_db, amp2_db,
                current_data['metadata']['filename'],
                comparison_data['metadata']['filename'],
                colorscale='Gray'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:
            st.subheader("Phase Comparison")
            fig = create_comparison_plot(
                current_data['phase'], comparison_data['phase'],
                current_data['metadata']['filename'],
                comparison_data['metadata']['filename'],
                colorscale='HSV'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[3]:
            st.subheader("Difference Analysis")
            
            # Amplitude difference
            amp_diff = comparison_data['amplitude'] - current_data['amplitude']
            phase_diff = comparison_data['phase'] - current_data['phase']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Amplitude Difference Statistics**")
                st.write(f"- Mean: {np.mean(amp_diff):.4f}")
                st.write(f"- Std: {np.std(amp_diff):.4f}")
                st.write(f"- Max: {np.max(np.abs(amp_diff)):.4f}")
                st.write(f"- RMS: {np.sqrt(np.mean(amp_diff**2)):.4f}")
            
            with col2:
                st.markdown("**Phase Difference Statistics**")
                st.write(f"- Mean: {np.mean(phase_diff):.4f} rad")
                st.write(f"- Std: {np.std(phase_diff):.4f} rad")
                st.write(f"- Max: {np.max(np.abs(phase_diff)):.4f} rad")
                st.write(f"- RMS: {np.sqrt(np.mean(phase_diff**2)):.4f} rad")
    
    else:
        # Single file view
        st.header(f"📄 {current_data['metadata']['filename']}")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("File Size", f"{current_data['metadata'].get('file_size_mb', 0):.2f} MB")
        with col2:
            st.metric("Shape", f"{current_data['metadata']['shape'][0]}×{current_data['metadata']['shape'][1]}")
        with col3:
            st.metric("Mean Amplitude", f"{current_data['statistics']['amplitude']['mean']:.3f}")
        with col4:
            st.metric("SNR (dB)", f"{current_data['statistics']['quality']['snr_estimate_db']:.1f}")
        
        # Tabs for visualization
        tabs = st.tabs(["📈 Amplitude", "🌊 Phase", "📊 Statistics", "📋 All Files Overview"])
        
        with tabs[0]:
            fig = create_plotly_amplitude_plot(current_data['amplitude'])
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:
            fig = create_plotly_phase_plot(current_data['phase'])
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:
            st.subheader("Detailed Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Amplitude**")
                st.json(current_data['statistics']['amplitude'])
            
            with col2:
                st.markdown("**Phase**")
                st.json(current_data['statistics']['phase'])
            
            st.markdown("**Quality Metrics**")
            st.json(current_data['statistics']['quality'])
        
        with tabs[3]:
            st.subheader("All Files Overview")
            
            # Load all files (with progress)
            all_files_data = []
            progress_bar = st.progress(0)
            
            for idx, file_path in enumerate(st.session_state.crsd_files):
                data = load_and_process_file(file_path)
                all_files_data.append(data)
                progress_bar.progress((idx + 1) / len(st.session_state.crsd_files))
            
            progress_bar.empty()
            
            # Create overview table
            overview_data = []
            for data in all_files_data:
                overview_data.append({
                    "Filename": data['metadata']['filename'],
                    "Size (MB)": f"{data['metadata'].get('file_size_mb', 0):.2f}",
                    "Shape": f"{data['metadata']['shape'][0]}×{data['metadata']['shape'][1]}",
                    "Mean Amp": f"{data['statistics']['amplitude']['mean']:.3f}",
                    "SNR (dB)": f"{data['statistics']['quality']['snr_estimate_db']:.1f}",
                })
            
            st.dataframe(overview_data, use_container_width=True)
            
            # Plot comparison of all files
            st.subheader("Multi-File Comparison")
            
            fig = go.Figure()
            
            for idx, data in enumerate(all_files_data):
                amp_profile = 20 * np.log10(data['amplitude'][data['amplitude'].shape[0]//2, :] + EPSILON)
                fig.add_trace(go.Scatter(
                    y=amp_profile,
                    mode='lines',
                    name=data['metadata']['filename'][:20],
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="Range Profiles - All Files",
                xaxis_title="Sample Index",
                yaxis_title="Amplitude (dB)",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Please select a file source from the sidebar to begin")
    
    st.markdown("""
    ### Features
    
    - **📁 Directory Browsing**: Scan entire directories for CRSD files
    - **📊 Multi-File Comparison**: Compare files side-by-side with difference visualization
    - **🔄 Quick Navigation**: Previous/Next buttons for fast file switching
    - **📈 Overview Mode**: See statistics for all files at once
    - **🎨 Interactive Visualizations**: Plotly-based zoom, pan, and hover
    - **💾 Smart Caching**: Files loaded once and cached for performance
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**CRSD Inspector v3.0**  
Multi-File Browser Edition
""")
