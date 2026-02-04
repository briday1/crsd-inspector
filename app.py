"""
CRSD Inspector - A Streamlit app for inspecting CRSD files using dagex workflows
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import tempfile
import os

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

st.title("📡 CRSD Inspector with dagex Workflow")
st.markdown("""
This application uses **dagex** (DAG executor) to create a workflow for processing and visualizing 
**CRSD** (Compensated Radar Signal Data) files.
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
                "file_path": file_path
            }
        except Exception as e:
            st.error(f"Error loading CRSD file: {e}")
            return {}
    return loader


def extract_metadata(inputs):
    """Extract metadata from CRSD file"""
    crsd_obj = inputs.get("crsd_object")
    if crsd_obj is None:
        return {}
    
    try:
        # Get CRSD metadata
        metadata_parts = ["### CRSD Metadata"]
        
        if hasattr(crsd_obj, 'crsd_meta'):
            # sarkit Reader
            meta = crsd_obj.crsd_meta
            if hasattr(meta, 'CollectionInfo'):
                metadata_parts.append(f"- **Collection Info**: {meta.CollectionInfo}")
            if hasattr(meta, 'Channel'):
                metadata_parts.append(f"- **Channels**: {len(meta.Channel)}")
        elif hasattr(crsd_obj, 'sicd_meta'):
            # sarpy format
            metadata_parts.append("- Using sarpy reader")
        else:
            metadata_parts.append("- Basic CRSD file loaded")
        
        # Try to get file header info
        if hasattr(crsd_obj, 'file_header'):
            header = crsd_obj.file_header
            if hasattr(header, 'xml_section_byte_offset'):
                metadata_parts.append(f"- **XML Offset**: {header.xml_section_byte_offset}")
        
        metadata_str = "\n".join(metadata_parts)
        
        return {
            "metadata": metadata_str,
            "crsd_object": crsd_obj  # Pass through
        }
    except Exception as e:
        return {
            "metadata": f"Error extracting metadata: {e}",
            "crsd_object": crsd_obj
        }


def read_signal_data(inputs):
    """Read signal data from CRSD file"""
    crsd_obj = inputs.get("crsd_object")
    if crsd_obj is None:
        return {}
    
    try:
        # Try to read signal array (first channel, limited samples)
        if hasattr(crsd_obj, 'read_signal_block'):
            # sarkit Reader API
            try:
                signal_block = crsd_obj.read_signal_block(0, 0, 512, 512)
                return {
                    "signal_data": signal_block,
                    "shape": str(signal_block.shape)
                }
            except Exception as e:
                # Try reading smaller block
                try:
                    signal_block = crsd_obj.read_signal_block(0, 0, 256, 256)
                    return {
                        "signal_data": signal_block,
                        "shape": str(signal_block.shape)
                    }
                except:
                    pass
        
        if hasattr(crsd_obj, '__getitem__'):
            # Try array-like access
            signal_block = np.array(crsd_obj[0][:512, :512])
            return {
                "signal_data": signal_block,
                "shape": str(signal_block.shape)
            }
        
        # Fallback: generate sample data for demo
        sample_data = np.random.randn(128, 128) + 1j * np.random.randn(128, 128)
        return {
            "signal_data": sample_data,
            "shape": f"{sample_data.shape} (sample data)"
        }
    except Exception as e:
        # Generate sample data on error
        sample_data = np.random.randn(128, 128) + 1j * np.random.randn(128, 128)
        return {
            "signal_data": sample_data,
            "shape": f"{sample_data.shape} (sample data - error: {str(e)[:50]})"
        }


def compute_amplitude(inputs):
    """Compute amplitude from complex signal data"""
    signal_data = inputs.get("signal_data")
    if signal_data is None:
        return {"amplitude": None}
    
    try:
        if signal_data is not None and hasattr(signal_data, '__len__'):
            amplitude = np.abs(signal_data)
            return {"amplitude": amplitude}
        else:
            return {"amplitude": None}
    except Exception as e:
        st.error(f"Error computing amplitude: {e}")
        return {"amplitude": None}


def compute_phase(inputs):
    """Compute phase from complex signal data"""
    signal_data = inputs.get("signal_data")
    if signal_data is None:
        return {"phase": None}
    
    try:
        if signal_data is not None and hasattr(signal_data, '__len__'):
            phase = np.angle(signal_data)
            return {"phase": phase}
        else:
            return {"phase": None}
    except Exception as e:
        st.error(f"Error computing phase: {e}")
        return {"phase": None}


def create_crsd_workflow(file_path):
    """Create a dagex workflow for CRSD processing"""
    graph = Graph()
    
    # Node 0: Load CRSD file
    graph.add(
        load_crsd_file(file_path),
        label="Load CRSD",
        inputs=None,
        outputs=[("crsd_object", "crsd"), ("file_path", "path")]
    )
    
    # Node 1: Extract metadata
    graph.add(
        extract_metadata,
        label="Extract Metadata",
        inputs=[("crsd", "crsd_object")],
        outputs=[("metadata", "meta"), ("crsd_object", "crsd_pass")]
    )
    
    # Node 2: Read signal data
    graph.add(
        read_signal_data,
        label="Read Signal",
        inputs=[("crsd_pass", "crsd_object")],
        outputs=[("signal_data", "signal"), ("shape", "signal_shape")]
    )
    
    # Node 3: Compute amplitude
    graph.add(
        compute_amplitude,
        label="Compute Amplitude",
        inputs=[("signal", "signal_data")],
        outputs=[("amplitude", "amp")]
    )
    
    # Node 4: Compute phase
    graph.add(
        compute_phase,
        label="Compute Phase",
        inputs=[("signal", "signal_data")],
        outputs=[("phase", "phase_out")]
    )
    
    return graph


# Sidebar for file upload
st.sidebar.header("📁 Upload CRSD File")
uploaded_file = st.sidebar.file_uploader("Choose a CRSD file", type=['crsd', 'nitf', 'ntf'])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.crsd') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    
    st.success(f"✅ Uploaded: {uploaded_file.name}")
    
    # Create and visualize the workflow
    st.header("🔄 dagex Workflow Structure")
    
    with st.spinner("Building workflow..."):
        graph = create_crsd_workflow(tmp_file_path)
        dag = graph.build()
        
        # Get Mermaid diagram
        try:
            mermaid_diagram = dag.to_mermaid()
            st.markdown("### Workflow DAG")
            st.code(mermaid_diagram, language="mermaid")
        except Exception as e:
            st.warning(f"Could not generate Mermaid diagram: {e}")
    
    # Execute the workflow
    st.header("⚡ Execute Workflow")
    
    col1, col2 = st.columns(2)
    with col1:
        parallel_exec = st.checkbox("Execute in parallel", value=True)
    with col2:
        num_threads = st.slider("Number of threads", 1, 8, 4)
    
    if st.button("🚀 Run Workflow", type="primary"):
        with st.spinner("Executing workflow..."):
            try:
                # Execute the DAG
                if parallel_exec:
                    context = dag.execute(True, num_threads)
                else:
                    context = dag.execute(False, None)
                
                st.success("✅ Workflow executed successfully!")
                
                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(["📊 Metadata", "📈 Amplitude", "🌊 Phase"])
                
                with tab1:
                    st.subheader("CRSD Metadata")
                    metadata = context.get("meta")
                    if metadata:
                        st.markdown(metadata)
                    else:
                        st.info("No metadata available")
                    
                    signal_shape = context.get("signal_shape")
                    if signal_shape:
                        st.markdown(f"**Signal Shape**: {signal_shape}")
                
                with tab2:
                    st.subheader("Amplitude Visualization")
                    amp = context.get("amp")
                    if amp is not None:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        im = ax.imshow(20 * np.log10(np.abs(amp) + 1e-10), 
                                     cmap='gray', aspect='auto')
                        ax.set_title("Amplitude (dB)")
                        ax.set_xlabel("Range")
                        ax.set_ylabel("Azimuth")
                        plt.colorbar(im, ax=ax, label="dB")
                        st.pyplot(fig)
                    else:
                        st.info("No amplitude data available")
                
                with tab3:
                    st.subheader("Phase Visualization")
                    phase = context.get("phase_out")
                    if phase is not None:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        im = ax.imshow(phase, cmap='hsv', aspect='auto')
                        ax.set_title("Phase")
                        ax.set_xlabel("Range")
                        ax.set_ylabel("Azimuth")
                        plt.colorbar(im, ax=ax, label="Phase (rad)")
                        st.pyplot(fig)
                    else:
                        st.info("No phase data available")
                
            except Exception as e:
                st.error(f"Error executing workflow: {e}")
                st.exception(e)
    
    # Clean up temp file
    try:
        os.unlink(tmp_file_path)
    except:
        pass

else:
    st.info("👈 Please upload a CRSD file using the sidebar to begin inspection")
    
    # Show demo workflow structure
    st.header("📋 Example Workflow Structure")
    st.markdown("""
    The CRSD inspection workflow consists of the following nodes:
    
    1. **Load CRSD**: Reads the CRSD file using sarkit/sarpy
    2. **Extract Metadata**: Parses XML metadata and collection information
    3. **Read Signal**: Reads a block of signal data from the file
    4. **Compute Amplitude**: Calculates amplitude from complex signal (parallel)
    5. **Compute Phase**: Calculates phase from complex signal (parallel)
    
    The workflow leverages dagex's parallel execution capabilities to process 
    amplitude and phase computations simultaneously.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
This app uses:
- **dagex**: DAG execution engine
- **sarkit**: CRSD file reading
- **streamlit**: Web interface
""")
