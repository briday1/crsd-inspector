"""
Signal Analysis Workflow
Amplitude-based signal analysis with PRF selection and pulse stacking
Generates amplitude/phase heatmaps, histograms, and statistics
"""
import numpy as np
from dagex import Graph
import plotly.graph_objects as go
from crsd_inspector.workflows.workflow import Workflow
from crsd_inspector.workflows.src.proc import signal_analysis_nodes as nodes
from crsd_inspector.workflows.src.plot import signal_analysis_plots as plots
from crsd_inspector.workflows.src.summary import signal_analysis_summary as summary
from crsd_inspector.workflows.src.util.wrappers import safe_plot_wrapper, wrap_with_timing

# Create workflow instance
workflow = Workflow(
    name="Signal Analysis",
    description="Amplitude-based analysis with PRF selection and pulse stacking"
)


# Workflow parameters
workflow.params = {
    'prf_hz': {
        'type': 'number',
        'label': 'PRF (Hz)',
        'default': 1000,
        'min': 1,
        'max': 100000,
        'help': 'Pulse Repetition Frequency for pulse stacking (will use from file if available)'
    },
    'num_pulses_to_stack': {
        'type': 'number',
        'label': 'Number of Pulses to Stack',
        'default': -1,
        'min': -1,
        'max': 100000,
        'step': 1,
        'help': 'Number of pulses to extract and stack for analysis (-1 for all pulses)'
    },
    'downsample_range_factor': {
        'type': 'number',
        'label': 'Range Downsample Factor',
        'default': 1,
        'min': 1,
        'max': 100,
        'step': 1,
        'help': 'Downsample range dimension by this factor (1=no downsampling, 10=every 10th sample)'
    }
}


def run_workflow(signal_data, metadata=None, **kwargs):
    """Run the signal analysis workflow and return formatted results"""
    workflow.clear()  # Clear any previous results
    
    # Merge params into metadata
    if metadata is None:
        metadata = {}
    metadata.update(kwargs)
    
    # Create and execute graph
    graph = _create_graph(signal_data, metadata)
    dag = graph.build()

    try:
        # Run single-threaded (parallel=False) so progress callbacks work from main thread
        result = dag.execute(False, 1)
    except Exception as e:
        import traceback
        workflow.add_text(f"❌ **Error executing workflow graph:** {str(e)}")
        workflow.add_text("**Stack trace:**")
        workflow.add_text(traceback.format_exc())
        return workflow.build()

    # Normalize execution result to a dict context.
    if isinstance(result, dict):
        context = result
    elif hasattr(result, "context"):
        context = result.context
    elif hasattr(result, "results"):
        context = result.results
    else:
        context = dict(result) if hasattr(result, "__dict__") else {}

    if not isinstance(context, dict):
        context = {}

    # Format and return results
    try:
        if not context:
            workflow.add_text("⚠️ No workflow outputs were produced.")
            return workflow.build()
        _format_results(context, metadata)
    except Exception as e:
        import traceback
        workflow.add_text(f"❌ **Error formatting results:** {str(e)}")
        workflow.add_text("**Stack trace:**")
        workflow.add_text(traceback.format_exc())
    return workflow.build()


def _create_graph(signal_data, metadata):
    """Create signal analysis workflow graph"""
    graph = Graph()
    
    # Extract parameters
    prf_hz = float(metadata.get('prf_hz', 1000))
    sample_rate_hz = float(metadata.get('sample_rate_hz', 100e6))
    num_pulses_to_stack = int(metadata.get('num_pulses_to_stack', 100))
    downsample_factor = int(metadata.get('downsample_range_factor', 1))
    
    # Try to get PRF from metadata first
    prf_from_file = nodes.extract_prf_from_metadata(metadata)
    if prf_from_file is not None:
        prf_hz = prf_from_file
    
    # Calculate PRI in samples
    pri_samples = int(sample_rate_hz / prf_hz)
    
    if signal_data.ndim == 1:
        signal_data = signal_data[None, :]
    
    total_samples = signal_data.shape[1]
    
    # Provide data node
    graph.add(
        lambda inputs: {
            'signal_data': signal_data,
            'prf_hz': prf_hz,
            'sample_rate_hz': sample_rate_hz,
            'pri_samples': pri_samples,
            'num_pulses_to_stack': num_pulses_to_stack,
            'total_samples': total_samples,
            'downsample_factor': downsample_factor
        },
        label="Provide Data",
        inputs=[],
        outputs=[
            ('signal_data', 'signal_data'),
            ('prf_hz', 'prf_hz'),
            ('sample_rate_hz', 'sample_rate_hz'),
            ('pri_samples', 'pri_samples'),
            ('num_pulses_to_stack', 'num_pulses_to_stack'),
            ('total_samples', 'total_samples'),
            ('downsample_factor', 'downsample_factor')
        ]
    )
    
    # Pulse extraction node
    graph.add(
        wrap_with_timing(
            nodes.extract_pulses,
            "Extract Pulses",
            "Extract pulse windows using PRI-based segmentation",
            metadata
        ),
        label="Extract Pulses",
        inputs=[
            ('signal_data', 'signal_data'),
            ('pri_samples', 'pri_samples'),
            ('num_pulses_to_stack', 'num_pulses_to_stack'),
            ('total_samples', 'total_samples')
        ],
        outputs=[
            ('pulses', 'pulses'),
            ('actual_num_pulses', 'actual_num_pulses'),
            ('_timing', '_timing_Extract Pulses')
        ]
    )
    
    # Compute statistics node
    graph.add(
        wrap_with_timing(
            nodes.compute_statistics,
            "Compute Statistics",
            "Calculate amplitude, phase, and quality metrics",
            metadata
        ),
        label="Compute Statistics",
        inputs=[('pulses', 'pulses')],
        outputs=[
            ('amplitude', 'amplitude'),
            ('phase', 'phase'),
            ('amplitude_stats', 'amplitude_stats'),
            ('phase_stats', 'phase_stats'),
            ('quality_metrics', 'quality_metrics'),
            ('iq_stats', 'iq_stats'),
            ('amp_flat', 'amp_flat'),
            ('phase_flat', 'phase_flat'),
            ('pulses', 'pulses'),  # Pass through for PSD
            ('_timing', '_timing_Compute Statistics')
        ]
    )
    
    # Build analysis parameters summary table
    graph.add(
        summary.build_analysis_summary,
        label="Build Analysis Summary",
        inputs=[
            ('prf_hz', 'prf_hz'),
            ('sample_rate_hz', 'sample_rate_hz'),
            ('pri_samples', 'pri_samples'),
            ('actual_num_pulses', 'actual_num_pulses'),
            ('downsample_factor', 'downsample_factor')
        ],
        outputs=[
            ('params_table', 'params_table')
        ]
    )
    
    # Plotting nodes
    graph.add(
        safe_plot_wrapper(plots.plot_psd, "Plot PSD", ['fig_psd']),
        label="Plot PSD",
        inputs=[
            ('pulses', 'pulses'),
            ('sample_rate_hz', 'sample_rate_hz')
        ],
        outputs=[
            ('fig_psd', 'fig_psd')
        ]
    )
    
    graph.add(
        safe_plot_wrapper(plots.plot_amplitude_heatmap, "Plot Amplitude Heatmap", ['fig_amplitude_heatmap']),
        label="Plot Amplitude Heatmap",
        inputs=[
            ('amplitude', 'amplitude'),
            ('downsample_factor', 'downsample_factor')
        ],
        outputs=[
            ('fig_amplitude_heatmap', 'fig_amplitude_heatmap')
        ]
    )
    
    graph.add(
        safe_plot_wrapper(plots.plot_phase_heatmap, "Plot Phase Heatmap", ['fig_phase_heatmap']),
        label="Plot Phase Heatmap",
        inputs=[
            ('phase', 'phase'),
            ('downsample_factor', 'downsample_factor')
        ],
        outputs=[
            ('fig_phase_heatmap', 'fig_phase_heatmap')
        ]
    )
    
    graph.add(
        safe_plot_wrapper(plots.plot_amplitude_histogram, "Plot Amplitude Histogram", ['fig_amplitude_histogram']),
        label="Plot Amplitude Histogram",
        inputs=[
            ('amp_flat', 'amp_flat')
        ],
        outputs=[
            ('fig_amplitude_histogram', 'fig_amplitude_histogram')
        ]
    )
    
    graph.add(
        safe_plot_wrapper(plots.plot_phase_histogram, "Plot Phase Histogram", ['fig_phase_histogram']),
        label="Plot Phase Histogram",
        inputs=[
            ('phase_flat', 'phase_flat')
        ],
        outputs=[
            ('fig_phase_histogram', 'fig_phase_histogram')
        ]
    )
    
    return graph


def _format_results(context, metadata):
    """Format workflow results for display"""
    
    # Add analysis parameters table (generated by graph node)
    params_table = context.get('params_table')
    if params_table:
        workflow.add_table("Analysis Parameters", params_table)
    
    # Add plots in order (retrieved from graph execution context)
    plot_names = [
        'fig_psd',
        'fig_amplitude_heatmap',
        'fig_phase_heatmap',
        'fig_amplitude_histogram',
        'fig_phase_histogram'
    ]
    
    for fig_name in plot_names:
        fig = context.get(fig_name)
        if fig is not None:
            workflow.add_plot(fig)
    
    # Check for errors
    plot_errors = []
    node_errors = []
    for key, value in context.items():
        if isinstance(key, str):
            if key == '_plot_error':
                plot_errors.append(value)
            elif key == '_node_error':
                node_errors.append(value)
    
    if plot_errors:
        workflow.add_text("**⚠️ Plot Generation Errors:**")
        for err in plot_errors:
            workflow.add_text(f"- {err}")
    
    if node_errors:
        workflow.add_text("**❌ Node Execution Errors:**")
        for err in node_errors:
            workflow.add_text(f"- {err}")
    
    # Add statistics tables (generated by compute_statistics node)
    amplitude_stats = context.get('amplitude_stats')
    if amplitude_stats:
        workflow.add_table("Amplitude Statistics", amplitude_stats)
    
    phase_stats = context.get('phase_stats')
    if phase_stats:
        workflow.add_table("Phase Statistics", phase_stats)
    
    quality_metrics = context.get('quality_metrics')
    if quality_metrics:
        workflow.add_table("Quality Metrics", quality_metrics)
    
    iq_stats = context.get('iq_stats')
    if iq_stats:
        workflow.add_table("I/Q Statistics", iq_stats)
