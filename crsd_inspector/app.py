"""
CRSD Inspector - Multi-File Browser
Streamlit app for CRSD file analysis with workflow-based processing
"""
import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os
import sys
import importlib.util
from datetime import datetime
from pathlib import Path
import glob

try:
    import dagex
    from dagex import Graph
except ImportError:
    st.error("dagex is not installed. Please run: pip install dagex")
    st.stop()

try:
    import sarkit.crsd as crsd
except ImportError:
    st.error("sarkit is required. Please run: pip install sarkit")
    st.stop()

st.set_page_config(page_title="CRSD Inspector", layout="wide")

# Initialize session state
if 'crsd_files' not in st.session_state:
    st.session_state.crsd_files = []
if 'current_file_index' not in st.session_state:
    st.session_state.current_file_index = 0
if 'file_cache' not in st.session_state:
    st.session_state.file_cache = {}
if 'selected_workflow' not in st.session_state:
    st.session_state.selected_workflow = None
if 'workflows' not in st.session_state:
    st.session_state.workflows = {}

st.title("CRSD Inspector")
st.markdown("""
Analyze CRSD files using modular workflows.
""")


def discover_workflows():
    """Discover available workflow modules"""
    workflows = {}
    workflows_dir = os.path.join(os.path.dirname(__file__), "workflows")
    
    if not os.path.isdir(workflows_dir):
        return workflows
    
    # Find all Python files in workflows directory
    workflow_files = glob.glob(os.path.join(workflows_dir, "*.py"))
    
    for filepath in workflow_files:
        filename = os.path.basename(filepath)
        if filename.startswith("_") or filename == "README.md":
            continue
        
        module_name = filename[:-3]  # Remove .py
        
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check for required functions/variables
            if hasattr(module, 'create_workflow') and hasattr(module, 'format_results'):
                workflow_name = getattr(module, 'WORKFLOW_NAME', module_name)
                workflow_desc = getattr(module, 'WORKFLOW_DESCRIPTION', '')
                
                workflows[workflow_name] = {
                    'module': module,
                    'name': workflow_name,
                    'description': workflow_desc,
                    'filepath': filepath
                }
        except Exception as e:
            st.warning(f"Failed to load workflow {filename}: {e}")
    
    return workflows


def execute_workflow(workflow_module, signal_data, metadata):
    """Execute a workflow and return formatted results"""
    try:
        # Create the workflow graph with data
        graph = workflow_module.create_workflow(signal_data=signal_data)
        dag = graph.build()
        
        # Execute
        context = dag.execute(True, 4)
        
        # Format results
        results = workflow_module.format_results(context)
        
        return results, context
    except Exception as e:
        import traceback
        st.error(f"Workflow execution failed: {e}")
        st.code(traceback.format_exc())
        return None, None


def scan_directory_for_crsd(directory_path):
    """Scan a directory for CRSD files"""
    crsd_extensions = ['*.crsd', '*.CRSD', '*.nitf', '*.NITF', '*.ntf', '*.NTF']
    files = []
    
    for ext in crsd_extensions:
        files.extend(glob.glob(os.path.join(directory_path, ext)))
    
    return sorted(files)


def load_and_process_file(file_path):
    """Load raw signal data from CRSD file"""
    # Check cache first
    if file_path in st.session_state.file_cache:
        return st.session_state.file_cache[file_path]
    
    try:
        signal_block = None
        tx_wfm = None
        channel_ids = []
        sample_rate_hz = None
        prf_hz = None
        
        # Load CRSD file with sarkit
        with open(file_path, 'rb') as f:
                reader = crsd.Reader(f)
                
                # Get channel IDs
                root = reader.metadata.xmltree.getroot()
                channels = root.findall('.//{http://api.nsgreg.nga.mil/schema/crsd/1.0}Channel')
                channel_ids = [ch.find('{http://api.nsgreg.nga.mil/schema/crsd/1.0}ChId').text 
                              for ch in channels] if channels else []
                
                # Load first channel (or only channel)
                if channel_ids:
                    signal_block = reader.read_signal(channel_ids[0])
                elif hasattr(reader, 'read_signal_block'):
                    signal_block = reader.read_signal_block(0, 0, 256, 256)
                
                # Try to load TX waveform
                try:
                    tx_wfm_array = reader.read_support_array("TX_WFM")
                    tx_wfm = np.asarray(tx_wfm_array[0, :], dtype=np.complex64)
                except:
                    pass
                
                # Extract radar parameters from metadata
                try:
                    radar_params = root.find('.//{http://api.nsgreg.nga.mil/schema/crsd/1.0}RadarParameters')
                    if radar_params is not None:
                        sample_rate = radar_params.find('.//{http://api.nsgreg.nga.mil/schema/crsd/1.0}SampleRate')
                        if sample_rate is not None:
                            sample_rate_hz = float(sample_rate.text)
                        
                        prf = radar_params.find('.//{http://api.nsgreg.nga.mil/schema/crsd/1.0}PRF')
                        if prf is not None:
                            prf_hz = float(prf.text)
                except:
                    pass
        
        if signal_block is None:
            raise ValueError(f"Unable to read file format: {file_path}")
        
        # Extract minimal metadata
        metadata = {
            "filename": os.path.basename(file_path),
            "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
            "shape": signal_block.shape,
            "num_channels": len(channel_ids) if channel_ids else 1,
            "channel_ids": channel_ids if channel_ids else ["CHAN1"],
        }
        
        # Add radar parameters if available
        if sample_rate_hz:
            metadata["sample_rate_hz"] = sample_rate_hz
        if prf_hz:
            metadata["prf_hz"] = prf_hz
        
        result = {
            "signal_data": signal_block,
            "tx_wfm": tx_wfm,
            "metadata": metadata,
            "file_path": file_path
        }
        
        # Cache the result
        st.session_state.file_cache[file_path] = result
        
        return result
        
    except Exception as e:
        st.error(f"Error loading {os.path.basename(file_path)}: {e}")
        return None


# Sidebar for directory/file selection
st.sidebar.header("File Selection")

# File source selection
file_source = st.sidebar.radio(
    "Source",
    ["Browse Directory", "Upload Files"],
    index=0
)

if file_source == "Browse Directory":
    directory_path = st.sidebar.text_input(
        "Directory Path",
        value="./examples",
        help="Enter path to directory containing CRSD files"
    )
    
    if st.sidebar.button("Scan Directory"):
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

# Workflow selection
st.sidebar.markdown("---")
st.sidebar.subheader("Workflow Selection")

# Discover workflows if not already done
if not st.session_state.workflows:
    st.session_state.workflows = discover_workflows()

if st.session_state.workflows:
    workflow_names = ["None (Default View)"] + list(st.session_state.workflows.keys())
    selected_workflow_name = st.sidebar.selectbox(
        "Select Workflow",
        workflow_names,
        index=0,
        help="Choose a processing workflow to apply to the selected CRSD file"
    )
    
    if selected_workflow_name != "None (Default View)":
        st.session_state.selected_workflow = st.session_state.workflows[selected_workflow_name]
        # Show workflow description
        if st.session_state.selected_workflow['description']:
            st.sidebar.caption(st.session_state.selected_workflow['description'])
    else:
        st.session_state.selected_workflow = None
else:
    st.sidebar.info("No workflows found in workflows/ directory")

# File list and selection
if st.session_state.crsd_files:
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"Files ({len(st.session_state.crsd_files)})")
    
    # File selector - using selectbox for better scalability
    file_options = [os.path.basename(f) for f in st.session_state.crsd_files]
    current_selection = st.sidebar.selectbox(
        "Select file:",
        options=range(len(file_options)),
        index=st.session_state.current_file_index,
        format_func=lambda x: file_options[x],
        key="file_selector"
    )
    
    if current_selection != st.session_state.current_file_index:
        st.session_state.current_file_index = current_selection
        st.rerun()
    
    # Navigation buttons
    st.sidebar.markdown("---")
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        if st.button("◀", key="prev_btn", use_container_width=True):
            if st.session_state.current_file_index > 0:
                st.session_state.current_file_index -= 1
                st.rerun()
    
    with col2:
        if st.button("↻", key="refresh_btn", use_container_width=True):
            st.session_state.file_cache = {}
            st.rerun()
    
    with col3:
        if st.button("▶", key="next_btn", use_container_width=True):
            if st.session_state.current_file_index < len(st.session_state.crsd_files) - 1:
                st.session_state.current_file_index += 1
                st.rerun()

# Main content
if st.session_state.crsd_files:
    current_file = st.session_state.crsd_files[st.session_state.current_file_index]
    
    # Load current file
    with st.spinner(f"Loading {os.path.basename(current_file)}..."):
        current_data = load_and_process_file(current_file)
    
    if current_data is None:
        st.error("Failed to load file. Please check the file format.")
    else:
        # Execute workflow if selected
        workflow_results = None
        if st.session_state.selected_workflow:
            with st.spinner(f"Executing workflow..."):
                workflow_results, workflow_context = execute_workflow(
                    st.session_state.selected_workflow['module'],
                    current_data['signal_data'],
                    current_data['metadata']
                )
        
        # Display workflow results
        if workflow_results:
            # Display tables
            if workflow_results.get("tables"):
                for table_data in workflow_results["tables"]:
                    st.markdown(f"**{table_data['title']}**")
                    # Convert dict to dataframe for better display
                    if isinstance(table_data['data'], dict):
                        df = pd.DataFrame(list(table_data['data'].items()), columns=['Metric', 'Value'])
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.dataframe(table_data['data'], use_container_width=True)
            
            # Display plots (figure objects from workflow)
            if workflow_results.get("plots"):
                for fig in workflow_results["plots"]:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Display text
            if workflow_results.get("text"):
                for text in workflow_results["text"]:
                    st.markdown(text)
        else:
            st.info("Select a workflow from the sidebar to analyze this file")

else:
    st.info("Please select a file source from the sidebar to begin")
    
    st.markdown("""
    ### Features
    
    - **Workflow-Based Analysis**: Modular processing pipelines
    - **Directory Browsing**: Scan entire directories for CRSD files
    - **Quick Navigation**: Previous/Next buttons for fast file switching
    - **Smart Caching**: Files loaded once and cached for performance
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**CRSD Inspector**  
Workflow-Based Analysis
""")
