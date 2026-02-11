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
if 'selected_channel' not in st.session_state:
    st.session_state.selected_channel = None
if 'auto_scanned' not in st.session_state:
    st.session_state.auto_scanned = False

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
            if hasattr(module, 'run_workflow'):
                # Try to get workflow instance first, then fall back to module constants
                if hasattr(module, 'workflow'):
                    workflow_obj = getattr(module, 'workflow')
                    workflow_name = workflow_obj.name
                    workflow_desc = workflow_obj.description
                else:
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


def execute_workflow(workflow_module, channel_data, channel_id, tx_wfm, metadata, **kwargs):
    """Execute a workflow and return formatted results"""
    try:
        # Extract signal data for selected channel
        signal_data = channel_data[channel_id]
        
        # Add TX waveform to metadata
        if tx_wfm is not None:
            metadata['tx_wfm'] = tx_wfm
        
        # Add channel data and selected channel to metadata (for multi-channel workflows)
        metadata['channel_data'] = channel_data
        metadata['selected_channel'] = channel_id
        
        # Run the workflow (internally handles graph creation and formatting)
        results = workflow_module.run_workflow(signal_data=signal_data, metadata=metadata, **kwargs)
        return results
    except Exception as e:
        import traceback
        st.error(f"Workflow execution failed: {e}")
        st.code(traceback.format_exc())
        return None


def scan_directory_for_crsd(directory_path):
    """Scan a directory for CRSD files"""
    crsd_extensions = ['*.crsd', '*.CRSD', '*.nitf', '*.NITF', '*.ntf', '*.NTF']
    files = []
    
    for ext in crsd_extensions:
        files.extend(glob.glob(os.path.join(directory_path, ext)))
    
    return sorted(files)


def scan_directory_for_tx_files(directory_path):
    """Scan a directory for TX CRSD files (_tx.crsd suffix)"""
    if not directory_path:
        return []
    
    files = []
    for ext in ['*_tx.crsd', '*_tx.CRSD', '*_tx.nitf', '*_tx.NITF']:
        files.extend(glob.glob(os.path.join(directory_path, ext)))
    
    return sorted(files)


def load_tx_waveform(tx_file_path):
    """Load TX waveform from a CRSD file"""
    if not tx_file_path or not os.path.isfile(tx_file_path):
        return None
    
    try:
        with open(tx_file_path, 'rb') as f:
            reader = crsd.Reader(f)
            # Try to load TX waveform
            try:
                tx_wfm_array = reader.read_support_array("TX_WFM")
                tx_wfm = np.asarray(tx_wfm_array[0, :], dtype=np.complex64)
                return tx_wfm
            except:
                pass
    except Exception as e:
        st.warning(f"Could not load TX waveform from {os.path.basename(tx_file_path)}: {e}")
    
    return None


def load_and_process_file(file_path):
    """Load raw signal data from CRSD file"""
    # Check cache first
    if file_path in st.session_state.file_cache:
        return st.session_state.file_cache[file_path]
    
    try:
        channel_data = {}
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
                
                # Load ALL channels
                if channel_ids:
                    for ch_id in channel_ids:
                        channel_data[ch_id] = reader.read_signal(ch_id)
                else:
                    # Fallback for files without channel metadata
                    if hasattr(reader, 'read_signal_block'):
                        channel_data["CHAN1"] = reader.read_signal_block(0, 0, 256, 256)
                        channel_ids = ["CHAN1"]
                
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
        
        if not channel_data:
            raise ValueError(f"Unable to read file format: {file_path}")
        
        # Extract minimal metadata
        first_channel = list(channel_data.values())[0]
        metadata = {
            "filename": os.path.basename(file_path),
            "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
            "shape": first_channel.shape,
            "num_channels": len(channel_ids),
            "channel_ids": channel_ids,
        }
        
        # Add radar parameters if available
        if sample_rate_hz:
            metadata["sample_rate_hz"] = sample_rate_hz
        if prf_hz:
            metadata["prf_hz"] = prf_hz
        
        # Check if companion TX file exists
        file_dir = os.path.dirname(file_path)
        file_base = os.path.basename(file_path)
        # Replace .crsd/.CRSD with _tx.crsd
        if file_base.lower().endswith('.crsd'):
            base_without_ext = file_base[:-5]  # Remove .crsd
            tx_file_candidate = os.path.join(file_dir, f"{base_without_ext}_tx.crsd")
            if os.path.isfile(tx_file_candidate):
                metadata["suggested_tx_file"] = tx_file_candidate
        
        result = {
            "channel_data": channel_data,
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
st.sidebar.header("CRSD Inspector")

# Auto-scan directory on first load
if not st.session_state.auto_scanned:
    directory_path = "./examples"
    if os.path.isdir(directory_path):
        files = scan_directory_for_crsd(directory_path)
        if files:
            st.session_state.crsd_files = files
            st.session_state.current_file_index = 0
    st.session_state.auto_scanned = True

# Directory input with scan button
directory_path = st.sidebar.text_input(
    "Directory Path",
    value="./examples",
    help="Enter path to directory containing CRSD files"
)

if st.sidebar.button("Scan Directory", use_container_width=True):
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

# Initialize current_data
current_data = None

# File selection dropdown
if st.session_state.crsd_files:
    st.sidebar.markdown("---")
    file_options = [os.path.basename(f) for f in st.session_state.crsd_files]
    current_selection = st.sidebar.selectbox(
        "CRSD File",
        options=range(len(file_options)),
        index=st.session_state.current_file_index,
        format_func=lambda x: file_options[x]
    )
    
    if current_selection != st.session_state.current_file_index:
        st.session_state.current_file_index = current_selection
        st.session_state.file_cache = {}  # Clear cache when changing files
        st.rerun()
    
    # Load current file to get channel info
    current_file = st.session_state.crsd_files[st.session_state.current_file_index]
    current_data = load_and_process_file(current_file)
    
    if current_data:
        # Channel selection
        channel_ids = current_data['metadata'].get('channel_ids', [])
        if channel_ids:
            selected_channel = st.sidebar.selectbox(
                "Channel",
                options=channel_ids,
                index=0 if st.session_state.selected_channel not in channel_ids else channel_ids.index(st.session_state.selected_channel)
            )
            st.session_state.selected_channel = selected_channel
        else:
            st.session_state.selected_channel = None

# Workflow selection
st.sidebar.markdown("---")

# Discover workflows if not already done
if not st.session_state.workflows:
    st.session_state.workflows = discover_workflows()

if st.session_state.workflows:
    workflow_names = ["None (Default View)"] + list(st.session_state.workflows.keys())
    selected_workflow_name = st.sidebar.selectbox(
        "Workflow",
        workflow_names,
        index=0,
        help="Choose a processing workflow"
    )
    
    if selected_workflow_name != "None (Default View)":
        st.session_state.selected_workflow = st.session_state.workflows[selected_workflow_name]
        # Show workflow description
        if st.session_state.selected_workflow['description']:
            st.sidebar.caption(st.session_state.selected_workflow['description'])
    else:
        st.session_state.selected_workflow = None
else:
    st.sidebar.info("No workflows found")

# Workflow parameters (if workflow selected)
if st.session_state.selected_workflow:
    workflow_module = st.session_state.selected_workflow['module']
    
    if hasattr(workflow_module, 'PARAMS'):
        st.sidebar.markdown("---")
        st.sidebar.subheader("Parameters")
        
        # Initialize session state for workflow params if not exists
        if 'workflow_params' not in st.session_state:
            st.session_state.workflow_params = {}
        
        for param_key, param_config in workflow_module.PARAMS.items():
            label = param_config.get('label', param_key)
            param_type = param_config.get('type', 'text')
            default = param_config.get('default')
            
            if param_type == 'dropdown':
                options = [opt['value'] for opt in param_config.get('options', [])]
                labels_map = {opt['value']: opt['label'] for opt in param_config.get('options', [])}
                
                # Find default index
                try:
                    default_idx = options.index(default) if default in options else 0
                except ValueError:
                    default_idx = 0
                
                selected = st.sidebar.selectbox(
                    label,
                    options=options,
                    index=default_idx,
                    format_func=lambda x: labels_map.get(x, x),
                    key=f"param_{param_key}"
                )
                st.session_state.workflow_params[param_key] = selected
                
            elif param_type == 'number':
                min_val = param_config.get('min')
                max_val = param_config.get('max')
                step = param_config.get('step')
                
                value = st.sidebar.number_input(
                    label,
                    value=float(default) if default is not None else 0.0,
                    min_value=float(min_val) if min_val is not None else None,
                    max_value=float(max_val) if max_val is not None else None,
                    step=float(step) if step is not None else None,
                    key=f"param_{param_key}"
                )
                st.session_state.workflow_params[param_key] = value
                
            elif param_type == 'checkbox':
                value = st.sidebar.checkbox(
                    label,
                    value=bool(default) if default is not None else False,
                    key=f"param_{param_key}"
                )
                st.session_state.workflow_params[param_key] = value
                
            elif param_type == 'text':
                # Special handling for tx_crsd_file parameter - use file picker
                if param_key == 'tx_crsd_file' and current_data:
                    current_file_path = current_data['file_path']
                    file_dir = os.path.dirname(current_file_path)
                    
                    # Discover TX files in same directory
                    tx_files = scan_directory_for_tx_files(file_dir)
                    tx_file_options = ['(None)'] + [os.path.basename(f) for f in tx_files]
                    tx_file_paths = [''] + tx_files
                    
                    # Create display map
                    tx_display_map = {path: name for path, name in zip(tx_file_paths, tx_file_options)}
                    
                    # Try to find the companion file
                    selected_idx = 0
                    suggested_tx = current_data['metadata'].get('suggested_tx_file', '')
                    if suggested_tx and suggested_tx in tx_files:
                        selected_idx = tx_file_paths.index(suggested_tx)
                    
                    selected_tx_path = st.sidebar.selectbox(
                        label,
                        options=tx_file_paths,
                        index=selected_idx,
                        format_func=lambda x: tx_display_map.get(x, x),
                        key=f"param_{param_key}",
                        help="Select a TX waveform file from the same directory, or leave as (None)"
                    )
                    st.session_state.workflow_params[param_key] = selected_tx_path
                else:
                    value = st.sidebar.text_input(
                        label,
                        value=str(default) if default is not None else '',
                        key=f"param_{param_key}"
                    )
                    st.session_state.workflow_params[param_key] = value
    
    # Execute button at the end of sidebar
    st.sidebar.markdown("---")
    execute_button_disabled = not (st.session_state.selected_workflow and st.session_state.selected_channel)
    execute_clicked = st.sidebar.button(
        "Execute Workflow",
        disabled=execute_button_disabled,
        type="primary",
        use_container_width=True
    )
    
    if execute_clicked and current_data:
        status_panel = st.status(
            f"Executing workflow on channel {st.session_state.selected_channel}...",
            state="running",
            expanded=True
        )
        with status_panel:
            progress_window = st.empty()
        max_visible_entries = 5
        completed_entries = []
        progress_state = {'active_entry': None}

        def render_progress_window():
            active_entry = progress_state['active_entry']
            if active_entry is not None:
                visible_done = completed_entries[-(max_visible_entries - 1):]
                lines = visible_done + [active_entry]
            else:
                lines = completed_entries[-max_visible_entries:]
            progress_window.markdown("\n".join(f"- {line}" for line in lines))

        def progress_callback(step, status, detail=""):
            line = f"**{step}**"
            if detail:
                line += f" - {detail}"
            if status == "running":
                progress_state['active_entry'] = f"⏳ {line}"
            elif status == "done":
                completed_entries.append(f"✅ {line}")
                progress_state['active_entry'] = None
            elif status == "failed":
                completed_entries.append(f"❌ {line}")
                progress_state['active_entry'] = None
            render_progress_window()

        # Prepare metadata with workflow parameters
        workflow_metadata = current_data['metadata'].copy()
        
        # Add workflow parameters from session state
        if 'workflow_params' in st.session_state:
            workflow_metadata.update(st.session_state.workflow_params)
        
        # Attach progress callback so workflow can stream execution updates
        workflow_metadata['_progress_callback'] = progress_callback
        
        # Check if a TX file was selected and load its waveform
        tx_wfm_to_use = current_data.get('tx_wfm')
        tx_file_path = workflow_metadata.get('tx_crsd_file', '').strip()
        if tx_file_path and os.path.isfile(tx_file_path):
            tx_wfm_from_file = load_tx_waveform(tx_file_path)
            if tx_wfm_from_file is not None:
                tx_wfm_to_use = tx_wfm_from_file
                st.success(f"Loaded TX waveform from: {os.path.basename(tx_file_path)}")
        
        workflow_results = execute_workflow(
            st.session_state.selected_workflow['module'],
            current_data['channel_data'],
            st.session_state.selected_channel,
            tx_wfm_to_use,
            workflow_metadata
        )
        
        if workflow_results is None:
            status_panel.update(label="Workflow failed", state="error")
        else:
            status_panel.update(label="Workflow complete", state="complete")
        
        # Store results in session state to persist across reruns
        st.session_state.last_workflow_results = workflow_results

# Main content
if st.session_state.crsd_files and current_data:
    # Display last executed results if available
    if 'last_workflow_results' in st.session_state:
        workflow_results = st.session_state.last_workflow_results
    else:
        workflow_results = None
    
    # Display workflow results
    if workflow_results:
        # Check if new ordered format or legacy format
        if "results" in workflow_results:
            # New ordered format - display in order
            for item in workflow_results["results"]:
                if item["type"] == "text":
                    for text in item["content"]:
                        st.markdown(text)
                elif item["type"] == "table":
                    st.markdown(f"**{item['title']}**")
                    # Convert dict to dataframe for better display
                    if isinstance(item['data'], dict):
                        df = pd.DataFrame(item['data'])
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.dataframe(item['data'], use_container_width=True)
                elif item["type"] == "plot":
                    st.plotly_chart(item["figure"], use_container_width=True)
        else:
            # Legacy format - display grouped by type (tables, plots, text)
            if workflow_results.get("tables"):
                for table_data in workflow_results["tables"]:
                    st.markdown(f"**{table_data['title']}**")
                    # Convert dict to dataframe for better display
                    if isinstance(table_data['data'], dict):
                        df = pd.DataFrame(table_data['data'])
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
        st.info("Select a workflow and click 'Execute Workflow' to analyze this file")
elif not st.session_state.crsd_files:
    st.info("No CRSD files found. Please scan a directory containing CRSD files.")
else:
    st.info("Loading file...")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**CRSD Inspector**  
Workflow-Based Analysis
""")


def main():
    """Entry point for crsd-inspector command"""
    import sys
    import streamlit.web.cli as stcli
    
    sys.argv = ["streamlit", "run", __file__]
    sys.exit(stcli.main())


if __name__ == "__main__":
    pass  # Streamlit handles execution
