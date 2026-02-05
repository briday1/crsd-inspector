"""
Signal Quality Assessment Workflow
Comprehensive quality metrics for CRSD data
"""
import numpy as np
from dagex import Graph
import plotly.graph_objects as go


WORKFLOW_NAME = "Signal Quality Assessment"
WORKFLOW_DESCRIPTION = "Comprehensive quality metrics including SNR, dynamic range, and clipping detection"


def run_workflow(signal_data, metadata=None):
    """Run the quality assessment workflow and return formatted results"""
    # Create and execute graph
    graph = _create_workflow(signal_data)
    dag = graph.build()
    context = dag.execute(True, 4)
    
    # Format and return results
    return _format_results(context)


def _create_workflow(signal_data=None):
    """Create signal quality assessment workflow"""
    graph = Graph()
    
    def provide_data(_inputs):
        return {"signal_data": signal_data}
    
    def compute_quality_metrics(inputs):
        signal = inputs.get("signal_data")
        if signal is None:
            return {}
        
        amplitude = np.abs(signal)
        phase = np.angle(signal)
        
        # SNR estimation
        signal_power = np.mean(amplitude**2)
        noise_floor = np.percentile(amplitude, 10)**2
        snr_db = 10 * np.log10(signal_power / (noise_floor + 1e-10))
        
        # Dynamic range
        dynamic_range_db = 20 * np.log10(np.max(amplitude) / (np.min(amplitude) + 1e-10))
        
        # Phase coherence (measure of phase stability)
        phase_diff = np.diff(phase, axis=0)
        phase_coherence = 1.0 - (np.std(phase_diff) / np.pi)
        
        # Clipping detection
        max_amp = np.max(amplitude)
        clipping_threshold = 0.99 * max_amp
        clipped_samples = np.sum(amplitude > clipping_threshold)
        clipping_percentage = 100.0 * clipped_samples / amplitude.size
        
        quality_metrics = {
            "snr_db": float(snr_db),
            "dynamic_range_db": float(dynamic_range_db),
            "phase_coherence": float(phase_coherence),
            "clipping_percentage": float(clipping_percentage),
            "total_samples": int(amplitude.size),
            "clipped_samples": int(clipped_samples)
        }
        
        return {"quality": quality_metrics, "amplitude": amplitude}
    
    def generate_quality_report(inputs):
        quality = inputs.get("quality")
        if quality is None:
            return {}
        
        # Generate quality assessment
        issues = []
        if quality["snr_db"] < 10:
            issues.append("Low SNR detected")
        if quality["dynamic_range_db"] < 30:
            issues.append("Limited dynamic range")
        if quality["phase_coherence"] < 0.7:
            issues.append("Poor phase coherence")
        if quality["clipping_percentage"] > 1.0:
            issues.append("Significant clipping detected")
        
        if not issues:
            assessment = "GOOD"
        elif len(issues) == 1:
            assessment = "FAIR"
        else:
            assessment = "POOR"
        
        report = {
            "overall_assessment": assessment,
            "issues": issues,
            "num_issues": len(issues)
        }
        
        return {"report": report}
    
    graph.add(
        provide_data,
        label="Provide Data",
        inputs=[],
        outputs=[("signal_data", "signal_data")]
    )
    
    graph.add(
        compute_quality_metrics,
        label="Quality Metrics",
        inputs=[("signal_data", "signal_data")],
        outputs=[("quality", "quality_metrics"), ("amplitude", "amp")]
    )
    
    graph.add(
        generate_quality_report,
        label="Quality Report",
        inputs=[("quality_metrics", "quality")],
        outputs=[("report", "quality_report")]
    )
    
    return graph


def _format_results(context):
    """Format workflow results for display"""
    results = {
        "tables": [],
        "plots": [],
        "text": []
    }
    
    # Quality metrics table
    quality = context.get("quality_metrics")
    if quality:
        results["tables"].append({
            "title": "Quality Metrics",
            "data": quality
        })
    
    # Quality report
    report = context.get("quality_report")
    if report:
        results["tables"].append({
            "title": "Quality Assessment",
            "data": report
        })
    
    # Add amplitude histogram
    amplitude = context.get("amp")
    if amplitude is not None:
        # Create histogram data
        hist, edges = np.histogram(amplitude.flatten(), bins=100)
        bin_centers = (edges[:-1] + edges[1:]) / 2
        
        fig = go.Figure(data=go.Bar(
            x=bin_centers,
            y=hist
        ))
        fig.update_layout(
            title="Amplitude Distribution",
            xaxis_title="Amplitude",
            yaxis_title="Count",
            height=500
        )
        results["plots"].append(fig)
    
    return results

