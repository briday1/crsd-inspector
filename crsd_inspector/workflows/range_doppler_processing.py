"""
Range Doppler Processing Workflow (Dagex version)

Modular matched filter processing for continuous staggered-PRF radar data.
"""

import numpy as np
import plotly.graph_objects as go
from dagex import Graph

from crsd_inspector.workflows.workflow import Workflow
from crsd_inspector.workflows.src.util.wrappers import (
    emit_progress, wrap_with_timing, make_window, downsample_heatmap, safe_plot_wrapper
)
from crsd_inspector.workflows.src.proc import range_doppler_nodes as nodes
from crsd_inspector.workflows.src.plot import range_doppler_plots as plots
from crsd_inspector.workflows.src.summary import range_doppler_summary as summary


# Create workflow instance
workflow = Workflow(
    name="Range Doppler Processing",
    description="Full processing pipeline: pulse extraction through Doppler compression"
)


# Workflow parameters
workflow.params = {
    'window_type': {
        'label': 'Slow-Time Window',
        'type': 'dropdown',
        'default': 'hamming',
        'options': [
            {'label': 'None', 'value': 'none'},
            {'label': 'Hamming', 'value': 'hamming'},
            {'label': 'Hanning', 'value': 'hanning'},
            {'label': 'Blackman', 'value': 'blackman'},
        ]
    },
    'range_window_type': {
        'label': 'Fast-Time (Range) Window',
        'type': 'dropdown',
        'default': 'hamming',
        'options': [
            {'label': 'None', 'value': 'none'},
            {'label': 'Hamming', 'value': 'hamming'},
            {'label': 'Hanning', 'value': 'hanning'},
            {'label': 'Blackman', 'value': 'blackman'},
        ]
    },
    'min_prf_hz': {
        'label': 'Min PRF (Hz)',
        'type': 'number',
        'default': 800,
    },
    'max_prf_hz': {
        'label': 'Max PRF (Hz)',
        'type': 'number',
        'default': 2400,
    },
    'auto_detect_prf': {
        'label': 'Auto-detect PRFs (cluster & snap)',
        'type': 'checkbox',
        'default': False,
    },
    'use_fixed_prfs': {
        'label': 'Use Fixed PRFs',
        'type': 'checkbox',
        'default': False,
    },
    'fixed_prfs_hz': {
        'label': 'Fixed PRFs (Hz, comma-separated)',
        'type': 'text',
        'default': '1000,1200,1500',
    },
    'tx_crsd_file': {
        'label': 'TX CRSD File (required)',
        'type': 'text',
        'default': '',
        'help': 'Path to transmit waveform CRSD file'
    },
    'nufft_kernel': {
        'label': 'NUFFT Kernel',
        'type': 'dropdown',
        'default': 'direct',
        'options': [
            {'label': 'Direct Computation (Exact)', 'value': 'direct'},
            {'label': 'Kaiser-Bessel (Approx, Fast)', 'value': 'kaiser_bessel'},
            {'label': 'Gaussian (Approx, Fast)', 'value': 'gaussian'},
        ]
    },
}


# ============================================================================
# WORKFLOW EXECUTION
# ============================================================================

def run_workflow(signal_data, metadata=None, **kwargs):
    """Execute pulse extraction workflow using dagex Graph"""
    workflow.clear()
    
    if metadata is None:
        metadata = {}
    
    tx_wfm = metadata.get('tx_wfm')
    if tx_wfm is None:
        workflow.add_text("❌ **Error:** TX waveform is required for Range-Doppler Processing workflow. Please select or provide a TX CRSD file.")
        return workflow.build()
    
    # Create and execute graph
    emit_progress(metadata, "Build Graph", "running", "Initialize processing pipeline")
    graph = _create_graph(signal_data, metadata)
    emit_progress(metadata, "Build Graph", "done", "Initialize processing pipeline")
    
    try:
        emit_progress(metadata, "Execute Graph", "running", "Run all processing nodes")
        dag = graph.build()
        # Run single-threaded (parallel=False) so progress callbacks work from main thread
        result = dag.execute(False, 1)
        emit_progress(metadata, "Execute Graph", "done", "Run all processing nodes")
        
        # Extract context - try multiple methods
        context = None
        if isinstance(result, dict):
            context = result
        elif hasattr(result, 'context'):
            context = result.context
        elif hasattr(result, 'results'):
            context = result.results
        else:
            context = dict(result) if hasattr(result, '__dict__') else {}
        
        # Ensure context is a dict
        if not isinstance(context, dict):
            context = {}
            
    except Exception as e:
        import traceback
        emit_progress(metadata, "Execute Graph", "failed", "Run all processing nodes")
        error_msg = f"Error executing graph: {str(e)}"
        workflow.add_text(error_msg)
        workflow.add_text("Stack trace:")
        workflow.add_code(traceback.format_exc())
        return workflow.build()
    
    # Check if context is empty
    if not context or not any(k for k in context.keys() if not k.startswith('_')):
        workflow.add_text("Warning: Graph execution returned empty results. Check input data and parameters.")
        return workflow.build()
    
    # Format and return results
    emit_progress(metadata, "Format Results", "running", "Build tables and plots")
    try:
        _format_results(context, metadata)
    except Exception as e:
        import traceback
        emit_progress(metadata, "Format Results", "failed", "Build tables and plots")
        workflow.add_text(f"❌ **Error formatting results:** {str(e)}")
        workflow.add_text("**Stack trace:**")
        workflow.add_code(traceback.format_exc())
    emit_progress(metadata, "Format Results", "done", "Build tables and plots")
    return workflow.build()


def _create_graph(signal_data, metadata):
    """Create pulse extraction workflow graph"""
    graph = Graph()
    
    # Extract parameters
    sample_rate_hz = float(metadata.get('sample_rate_hz', 100e6))
    window_type = metadata.get('window_type', 'hamming')
    range_window_type = metadata.get('range_window_type', 'hamming')
    min_prf_hz = float(metadata.get('min_prf_hz', 800))
    max_prf_hz = float(metadata.get('max_prf_hz', 2400))
    auto_detect_prf = bool(metadata.get('auto_detect_prf', False))
    use_fixed_prfs = bool(metadata.get('use_fixed_prfs', False))
    fixed_prfs_hz = str(metadata.get('fixed_prfs_hz', '1000,1200,1500'))
    nufft_kernel = metadata.get('nufft_kernel', 'direct')
    tx_wfm = metadata.get('tx_wfm')
    
    if signal_data.ndim == 1:
        signal_data = signal_data[None, :]
    
    total_samples = int(signal_data.shape[1])
    shortest_pri_samples = int(sample_rate_hz / max_prf_hz)
    
    # Provide initial data - no wrapper needed, manually track timing
    def provide_data(inputs):
        emit_progress(metadata, "Provide Data", "running", "Load signal, TX waveform, and processing parameters")
        import time
        start_time = time.perf_counter()
        
        result = {
            'signal_data': signal_data,
            'tx_wfm': tx_wfm,
            'window_type': window_type,
            'range_window_type': range_window_type,
            'sample_rate_hz': sample_rate_hz,
            'min_prf_hz': min_prf_hz,
            'max_prf_hz': max_prf_hz,
            'auto_detect_prf': auto_detect_prf,
            'use_fixed_prfs': use_fixed_prfs,
            'fixed_prfs_hz': fixed_prfs_hz,
            'nufft_kernel': nufft_kernel,
            'shortest_pri_samples': shortest_pri_samples,
            'total_samples': total_samples,
            'file_header_kvps': metadata.get('file_header_kvps', {})
        }
        
        elapsed_s = time.perf_counter() - start_time
        result['_timing_Provide Data'] = elapsed_s
        emit_progress(metadata, "Provide Data", "done", "Load signal, TX waveform, and processing parameters")
        
        return result
    
    graph.add(
        provide_data,
        label="Provide Data",
        inputs=[],
        outputs=[
            ('signal_data', 'signal_data'),
            ('tx_wfm', 'tx_wfm'),
            ('window_type', 'window_type'),
            ('range_window_type', 'range_window_type'),
            ('sample_rate_hz', 'sample_rate_hz'),
            ('min_prf_hz', 'min_prf_hz'),
            ('max_prf_hz', 'max_prf_hz'),
            ('auto_detect_prf', 'auto_detect_prf'),
            ('use_fixed_prfs', 'use_fixed_prfs'),
            ('fixed_prfs_hz', 'fixed_prfs_hz'),
            ('nufft_kernel', 'nufft_kernel'),
            ('shortest_pri_samples', 'shortest_pri_samples'),
            ('total_samples', 'total_samples'),
            ('file_header_kvps', 'file_header_kvps')
        ]
    )
    
    # Always use matched filter with TX waveform
    graph.add(
        wrap_with_timing(
            nodes.matched_filter,
            "Matched Filter",
            "Correlate RX signal with TX waveform",
            metadata
        ),
        label="Matched Filter",
        inputs=[
            ('signal_data', 'signal_data'),
            ('tx_wfm', 'tx_wfm'),
            ('range_window_type', 'range_window_type')
        ],
        outputs=[
            ('mf_output', 'mf_output'),
            ('mf_output_db', 'mf_output_db'),
            ('_timing', '_timing_Matched Filter')
        ]
    )
    
    # Fixed-PRF windowing
    graph.add(
        wrap_with_timing(
            nodes.fixed_prf_windows,
            "Fixed-PRF Windows",
            "Reshape matched-filter output into fixed windows",
            metadata
        ),
        label="Fixed-PRF Windows",
        inputs=[
            ('mf_output', 'mf_output'),
            ('shortest_pri_samples', 'shortest_pri_samples')
        ],
        outputs=[
            ('windows_2d', 'windows_2d'),
            ('windows_2d_db', 'windows_2d_db'),
            ('num_windows', 'num_windows'),
            ('_timing', '_timing_Fixed-PRF Windows')
        ]
    )
    
    # Pulse detection
    graph.add(
        wrap_with_timing(
            nodes.pulse_detection,
            "Pulse Detection",
            "Detect pulse-bearing windows via k-means",
            metadata
        ),
        label="Pulse Detection",
        inputs=[
            ('windows_2d', 'windows_2d'),
            ('shortest_pri_samples', 'shortest_pri_samples'),
            ('sample_rate_hz', 'sample_rate_hz')
        ],
        outputs=[
            ('pulses_2d', 'pulses_2d'),
            ('pulses_2d_db', 'pulses_2d_db'),
            ('num_pulses', 'num_pulses'),
            ('pulse_window_indices', 'pulse_window_indices'),
            ('pulse_start_times_s', 'pulse_start_times_s'),
            ('window_powers', 'window_powers'),
            ('has_pulse', 'has_pulse'),
            ('power_threshold', 'power_threshold'),
            ('cluster_centers', 'cluster_centers'),
            ('_timing', '_timing_Pulse Detection')
        ]
    )
    
    # Pulse timing analysis
    graph.add(
        wrap_with_timing(
            nodes.pulse_timing,
            "Pulse Timing Analysis",
            "Estimate pulse start times and PRIs",
            metadata
        ),
        label="Pulse Timing Analysis",
        inputs=[
            ('pulses_2d', 'pulses_2d'),
            ('pulse_start_times_s', 'pulse_start_times_s'),
            ('sample_rate_hz', 'sample_rate_hz'),
            ('num_pulses', 'num_pulses'),
            ('shortest_pri_samples', 'shortest_pri_samples'),
            ('pulse_window_indices', 'pulse_window_indices')
        ],
        outputs=[
            ('real_start_times_us', 'real_start_times_us'),
            ('pris_us', 'pris_us'),
            ('pulse_positions_samples', 'pulse_positions_samples'),
            ('intra_window_offsets', 'intra_window_offsets'),
            ('intra_window_offsets_us', 'intra_window_offsets_us'),
            ('_timing', '_timing_Pulse Timing Analysis')
        ]
    )
    
    # Auto-detect PRFs from PRI distribution
    graph.add(
        wrap_with_timing(
            nodes.detect_prfs, 
            "Detect PRFs (Cluster & Snap)", 
            "Estimate PRF modes and limits from PRI data",
            metadata
        ),
        label="Detect PRFs (Cluster & Snap)",
        inputs=[
            ('pris_us', 'pris_us'),
            ('auto_detect_prf', 'auto_detect_prf'),
            ('use_fixed_prfs', 'use_fixed_prfs'),
            ('fixed_prfs_hz', 'fixed_prfs_hz'),
            ('min_prf_hz', 'min_prf_hz'),
            ('max_prf_hz', 'max_prf_hz')
        ],
        outputs=[
            ('detected_min_prf_hz', 'detected_min_prf_hz'),
            ('detected_max_prf_hz', 'detected_max_prf_hz'),
            ('detected_prfs_hz', 'detected_prfs_hz'),
            ('pri_clusters', 'pri_clusters'),
            ('_timing', '_timing_Detect PRFs (Cluster & Snap)')
        ]
    )
    
    # Filter pulses by PRF bounds
    graph.add(
        wrap_with_timing(
            nodes.filter_pulses_by_prf_bounds,
            "Filter Pulses by PRF Bounds",
            "Reject pulses outside allowed PRI/PRF bounds",
            metadata
        ),
        label="Filter Pulses by PRF Bounds",
        inputs=[
            ('detected_min_prf_hz', 'detected_min_prf_hz'),
            ('detected_max_prf_hz', 'detected_max_prf_hz'),
            ('pulse_positions_samples', 'pulse_positions_samples'),
            ('pris_us', 'pris_us'),
            ('num_pulses', 'num_pulses')
        ],
        outputs=[
            ('pulse_positions_samples', 'pulse_positions_samples_filtered'),
            ('pris_us', 'pris_us_filtered'),
            ('num_pulses', 'num_pulses_filtered'),
            ('_timing', '_timing_Filter Pulses by PRF Bounds')
        ]
    )
    
    # PRI-based extraction
    graph.add(
        wrap_with_timing(
            nodes.pri_based_extraction,
            "PRI-Based Extraction",
            "Extract pulse windows using detected pulse positions",
            metadata
        ),
        label="PRI-Based Extraction",
        inputs=[
            ('mf_output', 'mf_output'),
            ('pulse_positions_samples_filtered', 'pulse_positions_samples'),
            ('pris_us_filtered', 'pris_us'),
            ('sample_rate_hz', 'sample_rate_hz'),
            ('num_pulses_filtered', 'num_pulses'),
            ('shortest_pri_samples', 'shortest_pri_samples')
        ],
        outputs=[
            ('pulses_extracted', 'pulses_extracted'),
            ('pulses_extracted_db', 'pulses_extracted_db'),
            ('extraction_window_samples', 'extraction_window_samples'),
            ('_timing', '_timing_PRI-Based Extraction')
        ]
    )
    
    # NUFFT Doppler compression - wrap to provide make_window function
    def nufft_with_make_window(inputs):
        return nodes.nufft_doppler(inputs, make_window)
    
    graph.add(
        wrap_with_timing(
            nufft_with_make_window,
            "NUFFT Doppler",
            "Perform Doppler compression on nonuniform PRI pulses",
            metadata
        ),
        label="NUFFT Doppler",
        inputs=[
            ('pulses_extracted', 'pulses_extracted'),
            ('pulse_positions_samples_filtered', 'pulse_positions_samples'),
            ('pris_us_filtered', 'pris_us'),
            ('sample_rate_hz', 'sample_rate_hz'),
            ('num_pulses_filtered', 'num_pulses'),
            ('extraction_window_samples', 'extraction_window_samples'),
            ('window_type', 'window_type'),
            ('nufft_kernel', 'nufft_kernel')
        ],
        outputs=[
            ('range_doppler', 'range_doppler'),
            ('range_doppler_db', 'range_doppler_db'),
            ('doppler_freqs_hz', 'doppler_freqs_hz'),
            ('doppler_resolution_hz', 'doppler_resolution_hz'),
            ('num_range_bins', 'num_range_bins'),
            ('is_uniform_prf', 'is_uniform_prf'),
            ('_timing', '_timing_NUFFT Doppler')
        ]
    )
    
    # Determine data type (pulsed vs continuous)
    graph.add(
        plots.determine_data_type,
        label="Determine Data Type",
        inputs=[
            ('num_pulses', 'num_pulses'),
            ('num_windows', 'num_windows')
        ],
        outputs=[
            ('is_pulsed', 'is_pulsed')
        ]
    )
    
    # Build processing summary table
    graph.add(
        summary.build_processing_summary,
        label="Build Processing Summary",
        inputs=[
            ('sample_rate_hz', 'sample_rate_hz'),
            ('window_type', 'window_type'),
            ('min_prf_hz', 'min_prf_hz'),
            ('max_prf_hz', 'max_prf_hz'),
            ('auto_detect_prf', 'auto_detect_prf'),
            ('total_samples', 'total_samples'),
            ('shortest_pri_samples', 'shortest_pri_samples'),
            ('detected_min_prf_hz', 'detected_min_prf_hz'),
            ('detected_max_prf_hz', 'detected_max_prf_hz'),
            ('detected_prfs_hz', 'detected_prfs_hz'),
            ('num_pulses', 'num_pulses'),
            ('num_windows', 'num_windows'),
            ('is_pulsed', 'is_pulsed'),
            ('doppler_resolution_hz', 'doppler_resolution_hz'),
            ('file_header_kvps', 'file_header_kvps')
        ],
        outputs=[
            ('summary_table', 'summary_table')
        ]
    )
    
    # Plot 1: Fixed-PRF windows
    graph.add(
        safe_plot_wrapper(plots.plot_fixed_prf_windows, "Plot Fixed-PRF Windows", ['fig_windows']),
        label="Plot Fixed-PRF Windows",
        inputs=[
            ('windows_2d_db', 'windows_2d_db'),
            ('sample_rate_hz', 'sample_rate_hz'),
            ('max_prf_hz', 'max_prf_hz'),
            ('is_pulsed', 'is_pulsed')
        ],
        outputs=[
            ('fig_windows', 'fig_windows')
        ]
    )
    
    # Plot 2: Pulse detection
    graph.add(
        safe_plot_wrapper(plots.plot_pulse_detection, "Plot Pulse Detection", ['fig_detection']),
        label="Plot Pulse Detection",
        inputs=[
            ('window_powers', 'window_powers'),
            ('has_pulse', 'has_pulse'),
            ('power_threshold', 'power_threshold')
        ],
        outputs=[
            ('fig_detection', 'fig_detection')
        ]
    )
    
    # Plot 3: Detected pulses
    graph.add(
        safe_plot_wrapper(plots.plot_detected_pulses, "Plot Detected Pulses", ['fig_pulses_heatmap', 'fig_amplitude', 'fig_phase']),
        label="Plot Detected Pulses",
        inputs=[
            ('pulses_2d_db', 'pulses_2d_db'),
            ('pulses_2d', 'pulses_2d'),
            ('sample_rate_hz', 'sample_rate_hz'),
            ('is_pulsed', 'is_pulsed')
        ],
        outputs=[
            ('fig_pulses_heatmap', 'fig_pulses_heatmap'),
            ('fig_amplitude', 'fig_amplitude'),
            ('fig_phase', 'fig_phase')
        ]
    )
    
    # Plot 4: PRI sequence
    graph.add(
        safe_plot_wrapper(plots.plot_pri_sequence, "Plot PRI Sequence", ['fig_pri']),
        label="Plot PRI Sequence",
        inputs=[
            ('pris_us_filtered', 'pris_us_filtered'),
            ('min_prf_hz', 'min_prf_hz'),
            ('max_prf_hz', 'max_prf_hz'),
            ('detected_min_prf_hz', 'detected_min_prf_hz'),
            ('detected_max_prf_hz', 'detected_max_prf_hz'),
            ('auto_detect_prf', 'auto_detect_prf'),
            ('is_pulsed', 'is_pulsed')
        ],
        outputs=[
            ('fig_pri', 'fig_pri')
        ]
    )
    
    # Plot 5: PRF clusters
    graph.add(
        safe_plot_wrapper(plots.plot_prf_clusters, "Plot PRF Clusters", ['fig_prf_clusters']),
        label="Plot PRF Clusters",
        inputs=[
            ('pris_us', 'pris_us'),
            ('detected_prfs_hz', 'detected_prfs_hz'),
            ('pri_clusters', 'pri_clusters'),
            ('auto_detect_prf', 'auto_detect_prf'),
            ('use_fixed_prfs', 'use_fixed_prfs'),
            ('is_pulsed', 'is_pulsed')
        ],
        outputs=[
            ('fig_prf_clusters', 'fig_prf_clusters')
        ]
    )
    
    # Plot 6: Extracted pulses
    graph.add(
        safe_plot_wrapper(plots.plot_extracted_pulses, "Plot Extracted Pulses", ['fig_extracted']),
        label="Plot Extracted Pulses",
        inputs=[
            ('pulses_extracted_db', 'pulses_extracted_db'),
            ('sample_rate_hz', 'sample_rate_hz'),
            ('num_pulses_filtered', 'num_pulses'),
            ('is_pulsed', 'is_pulsed')
        ],
        outputs=[
            ('fig_extracted', 'fig_extracted')
        ]
    )
    
    # Plot 7: Range-Doppler map
    graph.add(
        safe_plot_wrapper(plots.plot_range_doppler, "Plot Range-Doppler", ['fig_range_doppler']),
        label="Plot Range-Doppler",
        inputs=[
            ('range_doppler_db', 'range_doppler_db'),
            ('doppler_freqs_hz', 'doppler_freqs_hz'),
            ('sample_rate_hz', 'sample_rate_hz'),
            ('num_pulses_filtered', 'num_pulses'),
            ('is_uniform_prf', 'is_uniform_prf'),
            ('is_pulsed', 'is_pulsed')
        ],
        outputs=[
            ('fig_range_doppler', 'fig_range_doppler')
        ]
    )
    
    return graph


def _format_results(context, metadata):
    """Format and add all results to workflow"""
    # Extract all results from context
    results = {}
    for key, value in context.items():
        if isinstance(key, tuple):
            output_name, variant_idx = key
            results[output_name] = value
        else:
            results[key] = value
    
    workflow.add_text(f"**Filter Type:** Matched Filter (TX waveform)")
    
    # Check for NUFFT errors
    if 'nufft_error' in results:
        workflow.add_text(f"⚠️ **NUFFT Doppler Error:** {results['nufft_error']}")
        if 'nufft_traceback' in results:
            workflow.add_text(f"Traceback: {results['nufft_traceback']}")
    
    # Add summary table (generated by graph node)
    summary_table = results.get('summary_table')
    if summary_table is not None:
        workflow.add_table("Processing Parameters", summary_table)
    
    # Add plots in order (retrieved from graph execution context)
    plot_names = [
        'fig_windows',
        'fig_detection',
        'fig_pulses_heatmap',
        'fig_amplitude',
        'fig_phase',
        'fig_pri',
        'fig_prf_clusters',
        'fig_extracted',
        'fig_range_doppler'
    ]
    
    for fig_name in plot_names:
        fig = results.get(fig_name)
        if fig is not None:
            workflow.add_plot(fig)
    
    # Check for any plot errors
    plot_errors = {}
    for key in results.keys():
        if isinstance(key, str) and key.startswith('_plot_error') and not key.endswith('_traceback'):
            plot_errors[key] = results[key]
    
    if plot_errors:
        workflow.add_text("**⚠️ Plot Generation Errors:**")
        for key, err in plot_errors.items():
            workflow.add_text(f"- {err}")
    
    # Check for any node execution errors (stored as tuples with node output keys)
    node_errors = {}
    for key, value in results.items():
        if isinstance(key, str) and key == '_node_error':
            node_errors[key] = value
        elif isinstance(key, tuple) and len(key) == 2:
            # Check if this tuple output contains node error
            if key[0] == '_node_error':
                node_errors[key] = value
    
    if node_errors:
        workflow.add_text("**❌ Node Execution Errors:**")
        for key, err in node_errors.items():
            workflow.add_text(f"- {err}")
        workflow.add_text("**Note:** Errors in processing nodes will cause downstream plots to be missing.")

