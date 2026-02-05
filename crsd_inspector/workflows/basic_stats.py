"""
Basic Statistics Workflow
Simple workflow that computes basic signal statistics
"""
import numpy as np
from dagex import Graph
import plotly.graph_objects as go


WORKFLOW_NAME = "Basic Statistics"
WORKFLOW_DESCRIPTION = "Compute basic amplitude and phase statistics"


def run_workflow(signal_data, metadata=None):
    """Run the basic statistics workflow and return formatted results"""
    # Create and execute graph
    graph = _create_workflow(signal_data)
    dag = graph.build()
    context = dag.execute(True, 4)
    
    # Format and return results
    return _format_results(context)


def _create_workflow(signal_data=None):
    """Create basic statistics workflow"""
    graph = Graph()
    
    def provide_data(_inputs):
        return {"signal_data": signal_data}
    
    def compute_basic_stats(inputs):
        signal = inputs.get("signal_data")
        if signal is None:
            return {}
        
        amplitude = np.abs(signal)
        phase = np.angle(signal)
        
        # Comprehensive statistics
        amplitude_stats = {
            "min": float(np.min(amplitude)),
            "max": float(np.max(amplitude)),
            "mean": float(np.mean(amplitude)),
            "std": float(np.std(amplitude)),
            "median": float(np.median(amplitude)),
        }
        
        phase_stats = {
            "min": float(np.min(phase)),
            "max": float(np.max(phase)),
            "mean": float(np.mean(phase)),
            "std": float(np.std(phase)),
        }
        
        quality_metrics = {
            "snr_estimate_db": float(20 * np.log10(np.mean(amplitude) / (np.std(amplitude) + 1e-10))),
            "dynamic_range_db": float(20 * np.log10(np.max(amplitude) / (np.min(amplitude) + 1e-10))),
        }
        
        return {
            "amplitude_stats": amplitude_stats,
            "phase_stats": phase_stats,
            "quality_metrics": quality_metrics,
            "amplitude": amplitude,
            "phase": phase
        }
    
    graph.add(
        provide_data,
        label="Load Data",
        inputs=None,
        outputs=[("signal_data", "signal")]
    )
    
    graph.add(
        compute_basic_stats,
        label="Basic Statistics",
        inputs=[("signal", "signal_data")],
        outputs=[
            ("amplitude_stats", "amplitude_stats"),
            ("phase_stats", "phase_stats"),
            ("quality_metrics", "quality_metrics"),
            ("amplitude", "amp"),
            ("phase", "phase")
        ]
    )
    
    return graph


def _format_results(context):
    """Format workflow results for display"""
    results = {
        "tables": [],
        "plots": [],
        "text": []
    }
    
    # File metadata (from context metadata if available)
    # This will be populated by the workflow execution
    
    # Amplitude statistics table
    amplitude_stats = context.get("amplitude_stats")
    if amplitude_stats:
        results["tables"].append({
            "title": "Amplitude Statistics",
            "data": amplitude_stats
        })
    
    # Phase statistics table
    phase_stats = context.get("phase_stats")
    if phase_stats:
        results["tables"].append({
            "title": "Phase Statistics",
            "data": phase_stats
        })
    
    # Quality metrics table
    quality_metrics = context.get("quality_metrics")
    if quality_metrics:
        results["tables"].append({
            "title": "Quality Metrics",
            "data": quality_metrics
        })
    
    # Add amplitude plot
    amplitude = context.get("amp")
    if amplitude is not None:
        # Convert to dB
        amp_db = 20 * np.log10(np.abs(amplitude) + 1e-10)
        
        fig = go.Figure(data=go.Heatmap(
            z=amp_db,
            colorscale='Gray',
            colorbar=dict(title="Amplitude (dB)")
        ))
        fig.update_layout(
            title="Signal Amplitude (dB)",
            xaxis_title="Range Sample",
            yaxis_title="Azimuth Vector",
            height=500
        )
        results["plots"].append(fig)
    
    # Add phase plot
    phase = context.get("phase")
    if phase is not None:
        fig = go.Figure(data=go.Heatmap(
            z=phase,
            colorscale='HSV',
            colorbar=dict(title="Phase (rad)"),
            zmid=0
        ))
        fig.update_layout(
            title="Signal Phase",
            xaxis_title="Range Sample",
            yaxis_title="Azimuth Vector",
            height=500
        )
        results["plots"].append(fig)
    
    return results
