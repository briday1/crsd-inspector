"""
Basic Statistics Workflow
Simple workflow that computes basic signal statistics
"""
import numpy as np
from dagex import Graph


def create_workflow(signal_data=None):
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
        
        stats = {
            "amplitude_mean": float(np.mean(amplitude)),
            "amplitude_std": float(np.std(amplitude)),
            "amplitude_min": float(np.min(amplitude)),
            "amplitude_max": float(np.max(amplitude)),
            "phase_mean": float(np.mean(phase)),
            "phase_std": float(np.std(phase)),
        }
        
        return {"stats": stats, "amplitude": amplitude, "phase": phase}
    
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
        outputs=[("stats", "statistics"), ("amplitude", "amp"), ("phase", "phase")]
    )
    
    return graph


def format_results(context):
    """Format workflow results for display"""
    results = {
        "tables": [],
        "plots": [],
        "text": []
    }
    
    stats = context.get("statistics")
    if stats:
        results["tables"].append({
            "title": "Basic Statistics",
            "data": stats
        })
    
    return results


# Workflow metadata
WORKFLOW_NAME = "Basic Statistics"
WORKFLOW_DESCRIPTION = "Compute basic amplitude and phase statistics"
