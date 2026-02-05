"""
Range-Doppler Processing Workflow
Performs FFT processing to generate range-doppler map
"""
import numpy as np
from dagex import Graph
import plotly.graph_objects as go


WORKFLOW_NAME = "Range-Doppler Processing"
WORKFLOW_DESCRIPTION = "2D FFT processing to generate range-doppler map"


def run_workflow(signal_data, metadata=None):
    """Run the range-doppler workflow and return formatted results"""
    # Create and execute graph
    graph = _create_workflow(signal_data)
    dag = graph.build()
    context = dag.execute(True, 4)
    
    # Format and return results
    return _format_results(context)


def _create_workflow(signal_data=None):
    """Create range-doppler processing workflow"""
    graph = Graph()
    
    def provide_data(_inputs):
        return {"signal_data": signal_data}
    
    def compute_range_doppler(inputs):
        signal = inputs.get("signal_data")
        if signal is None:
            return {}
        
        # Perform 2D FFT for range-doppler processing
        # Assume signal is already in range-time format
        range_doppler = np.fft.fft2(signal)
        range_doppler = np.fft.fftshift(range_doppler)
        
        # Compute magnitude
        rd_magnitude = np.abs(range_doppler)
        
        # Compute in dB
        rd_db = 20 * np.log10(rd_magnitude + 1e-10)
        
        return {
            "range_doppler_complex": range_doppler,
            "range_doppler_db": rd_db,
            "range_doppler_magnitude": rd_magnitude
        }
    
    def compute_rd_statistics(inputs):
        rd_db = inputs.get("range_doppler_db")
        if rd_db is None:
            return {}
        
        stats = {
            "peak_value_db": float(np.max(rd_db)),
            "mean_value_db": float(np.mean(rd_db)),
            "dynamic_range_db": float(np.max(rd_db) - np.min(rd_db)),
        }
        
        # Find peak location
        peak_idx = np.unravel_index(np.argmax(rd_db), rd_db.shape)
        stats["peak_doppler_bin"] = int(peak_idx[0])
        stats["peak_range_bin"] = int(peak_idx[1])
        
        return {"rd_stats": stats}
    
    graph.add(
        provide_data,
        label="Provide Data",
        inputs=[],
        outputs=[("signal_data", "signal_data")]
    )
    
    graph.add(
        compute_range_doppler,
        label="Range-Doppler FFT",
        inputs=[("signal_data", "signal_data")],
        outputs=[
            ("range_doppler_db", "rd_db"),
            ("range_doppler_magnitude", "rd_mag"),
            ("range_doppler_complex", "rd_complex")
        ]
    )
    
    graph.add(
        compute_rd_statistics,
        label="RD Statistics",
        inputs=[("rd_db", "range_doppler_db")],
        outputs=[("rd_stats", "rd_statistics")]
    )
    
    return graph


def _format_results(context):
    """Format workflow results for display"""
    results = {
        "tables": [],
        "plots": [],
        "text": []
    }
    
    # Add RD statistics table
    rd_stats = context.get("rd_statistics")
    if rd_stats:
        results["tables"].append({
            "title": "Range-Doppler Statistics",
            "data": rd_stats
        })
    
    # Add range-doppler plot
    rd_db = context.get("rd_db")
    if rd_db is not None:
        fig = go.Figure(data=go.Heatmap(
            z=rd_db,
            colorscale='Viridis',
            colorbar=dict(title="Magnitude (dB)")
        ))
        fig.update_layout(
            title="Range-Doppler Map",
            xaxis_title="Range Bin",
            yaxis_title="Doppler Bin",
            height=500
        )
        results["plots"].append(fig)
    
    return results
