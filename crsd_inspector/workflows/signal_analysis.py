"""
Signal Analysis Workflow
Comprehensive signal visualization and statistics
Combines amplitude, phase, and I/Q component analysis
"""
import numpy as np
from dagex import Graph
import plotly.graph_objects as go
from crsd_inspector.workflows.workflow import Workflow


# Create workflow instance
workflow = Workflow(
    name="Signal Analysis",
    description="Comprehensive signal visualization with amplitude, phase, and I/Q components"
)


def run_workflow(signal_data, metadata=None, **kwargs):
    """Run the signal analysis workflow and return formatted results"""
    workflow.clear()  # Clear any previous results
    
    # Create and execute graph
    graph = _create_graph(signal_data)
    dag = graph.build()
    context = dag.execute(True, 4)
    
    # Format and return results
    _format_results(context)
    return workflow.build()


def _create_graph(signal_data):
    """Create signal analysis workflow"""
    graph = Graph()
    
    def provide_data(_inputs):
        return {"signal_data": signal_data}
    
    def compute_statistics(inputs):
        signal = inputs.get("signal_data")
        if signal is None:
            return {}
        
        amplitude = np.abs(signal)
        phase = np.angle(signal)
        
        # Flatten for I/Q components
        if len(signal.shape) == 2:
            samples_flat = signal.flatten()
        else:
            samples_flat = signal
        
        # Comprehensive statistics
        amplitude_stats = {
            "Metric": ["Min", "Max", "Mean", "Std Dev", "Median"],
            "Value": [
                f"{np.min(amplitude):.4f}",
                f"{np.max(amplitude):.4f}",
                f"{np.mean(amplitude):.4f}",
                f"{np.std(amplitude):.4f}",
                f"{np.median(amplitude):.4f}"
            ]
        }
        
        phase_stats = {
            "Metric": ["Min (rad)", "Max (rad)", "Mean (rad)", "Std Dev (rad)"],
            "Value": [
                f"{np.min(phase):.4f}",
                f"{np.max(phase):.4f}",
                f"{np.mean(phase):.4f}",
                f"{np.std(phase):.4f}"
            ]
        }
        
        # Enhanced quality metrics
        signal_power = np.mean(amplitude**2)
        noise_floor = np.percentile(amplitude, 10)**2
        snr_db = 10 * np.log10(signal_power / (noise_floor + 1e-10))
        dynamic_range_db = 20 * np.log10(np.max(amplitude) / (np.min(amplitude) + 1e-10))
        
        # Clipping detection
        max_amp = np.max(amplitude)
        clipping_threshold = 0.99 * max_amp
        clipped_samples = np.sum(amplitude > clipping_threshold)
        clipping_percentage = 100.0 * clipped_samples / amplitude.size
        
        quality_metrics = {
            "Metric": ["SNR", "Dynamic Range", "Clipping %", "Total Samples", "Clipped Samples"],
            "Value": [
                f"{snr_db:.2f} dB",
                f"{dynamic_range_db:.2f} dB",
                f"{clipping_percentage:.2f}%",
                f"{amplitude.size}",
                f"{clipped_samples}"
            ]
        }
        
        # Quality assessment
        issues = []
        if snr_db < 10:
            issues.append("Low SNR detected")
        if dynamic_range_db < 30:
            issues.append("Limited dynamic range")
        if clipping_percentage > 1.0:
            issues.append("Significant clipping detected")
        
        if not issues:
            assessment = "GOOD"
        elif len(issues) == 1:
            assessment = "FAIR"
        else:
            assessment = "POOR"
        
        quality_report = {
            "Assessment": [assessment],
            "Issues Found": [len(issues)],
            "Details": ["; ".join(issues) if issues else "No issues detected"]
        }
        
        iq_stats = {
            "Component": ["I (Real)", "Q (Imaginary)"],
            "Mean": [f"{np.mean(samples_flat.real):.4f}", f"{np.mean(samples_flat.imag):.4f}"],
            "Std Dev": [f"{np.std(samples_flat.real):.4f}", f"{np.std(samples_flat.imag):.4f}"]
        }
        
        return {
            "amplitude_stats": amplitude_stats,
            "phase_stats": phase_stats,
            "quality_metrics": quality_metrics,
            "quality_report": quality_report,
            "iq_stats": iq_stats,
            "amplitude": amplitude,
            "phase": phase,
            "samples_flat": samples_flat
        }
    
    graph.add(
        provide_data,
        label="Load Data",
        inputs=None,
        outputs=[("signal_data", "signal")]
    )
    
    graph.add(
        compute_statistics,
        label="Compute Statistics",
        inputs=[("signal", "signal_data")],
        outputs=[
            ("amplitude_stats", "amplitude_stats"),
            ("phase_stats", "phase_stats"),
            ("quality_report", "quality_report"),
            ("quality_metrics", "quality_metrics"),
            ("iq_stats", "iq_stats"),
            ("amplitude", "amplitude"),
            ("phase", "phase"),
            ("samples_flat", "samples_flat")
        ]
    )
    
    return graph


def _format_results(context):
    """Format workflow results for display"""
    # Text summary first
    workflow.add_text("Signal analysis completed successfully")
    
    # Add plots
    amplitude = context.get("amplitude")
    if amplitude is not None:
        # Convert to dB
        amp_db = 20 * np.log10(np.abs(amplitude) + 1e-10)
        
        # Calculate reasonable range for controls
        amp_min_default = float(np.percentile(amp_db, 1))
        amp_max_default = float(np.percentile(amp_db, 99))
        
        fig = go.Figure(data=go.Heatmap(
            z=amp_db,
            colorscale='Jet',
            zmin=amp_min_default,
            zmax=amp_max_default,
            colorbar=dict(
                title="Amplitude (dB)",
                x=1.15
            )
        ))
        
        # Add interactive colormap controls as sliders
        steps_min = []
        steps_max = []
        range_vals_min = np.linspace(np.min(amp_db), amp_max_default, 20)
        range_vals_max = np.linspace(amp_min_default, np.max(amp_db), 20)
        
        for val in range_vals_min:
            steps_min.append(dict(
                method="restyle",
                args=[{"zmin": val}],
                label=f"{val:.0f}"
            ))
        
        for val in range_vals_max:
            steps_max.append(dict(
                method="restyle",
                args=[{"zmax": val}],
                label=f"{val:.0f}"
            ))
        
        fig.update_layout(
            title="Signal Amplitude (dB)",
            xaxis_title="Range Sample",
            yaxis_title="Pulse Number",
            height=600,
            template='plotly_dark',
            sliders=[
                dict(
                    active=10,
                    yanchor="top",
                    y=-0.15,
                    xanchor="left",
                    currentvalue=dict(
                        prefix="Min: ",
                        visible=True,
                        xanchor="right"
                    ),
                    pad=dict(b=10, t=10),
                    len=0.42,
                    x=0.0,
                    steps=steps_min
                ),
                dict(
                    active=10,
                    yanchor="top",
                    y=-0.15,
                    xanchor="right",
                    currentvalue=dict(
                        prefix="Max: ",
                        visible=True,
                        xanchor="left"
                    ),
                    pad=dict(b=10, t=10),
                    len=0.42,
                    x=1.0,
                    steps=steps_max
                )
            ]
        )
        workflow.add_plot(fig)
    
    phase = context.get("phase")
    if phase is not None:
        # Calculate reasonable range for controls
        phase_min_default = float(np.percentile(phase, 1))
        phase_max_default = float(np.percentile(phase, 99))
        
        fig = go.Figure(data=go.Heatmap(
            z=phase,
            colorscale='HSV',
            zmin=phase_min_default,
            zmax=phase_max_default,
            colorbar=dict(
                title="Phase (rad)",
                x=1.15
            ),
            zmid=0
        ))
        
        # Add interactive colormap controls as sliders
        steps_min = []
        steps_max = []
        range_vals_min = np.linspace(-np.pi, 0, 15)
        range_vals_max = np.linspace(0, np.pi, 15)
        
        for val in range_vals_min:
            steps_min.append(dict(
                method="restyle",
                args=[{"zmin": val}],
                label=f"{val:.2f}"
            ))
        
        for val in range_vals_max:
            steps_max.append(dict(
                method="restyle",
                args=[{"zmax": val}],
                label=f"{val:.2f}"
            ))
        
        fig.update_layout(
            title="Signal Phase",
            xaxis_title="Range Sample",
            yaxis_title="Pulse Number",
            height=600,
            template='plotly_dark',
            sliders=[
                dict(
                    active=7,
                    yanchor="top",
                    y=-0.15,
                    xanchor="left",
                    currentvalue=dict(
                        prefix="Min: ",
                        visible=True,
                        xanchor="right"
                    ),
                    pad=dict(b=10, t=10),
                    len=0.42,
                    x=0.0,
                    steps=steps_min
                ),
                dict(
                    active=7,
                    yanchor="top",
                    y=-0.15,
                    xanchor="right",
                    currentvalue=dict(
                        prefix="Max: ",
                        visible=True,
                        xanchor="left"
                    ),
                    pad=dict(b=10, t=10),
                    len=0.42,
                    x=1.0,
                    steps=steps_max
                )
            ]
        )
        workflow.add_plot(fig)
    
    # Power Spectral Density
    samples_flat = context.get("samples_flat")
    if samples_flat is not None:
        # Compute PSD using FFT
        n_samples = len(samples_flat)
        fft_data = np.fft.fft(samples_flat)
        psd = np.abs(fft_data)**2 / n_samples
        psd_db = 10 * np.log10(psd + 1e-10)
        
        # Frequency axis (normalized)
        freq = np.fft.fftfreq(n_samples)
        
        # Shift to center zero frequency
        freq_shifted = np.fft.fftshift(freq)
        psd_db_shifted = np.fft.fftshift(psd_db)
        
        # Estimate noise floor (use 25th percentile)
        noise_floor_db = np.percentile(psd_db_shifted, 25)
        
        fig_psd = go.Figure()
        fig_psd.add_trace(go.Scatter(
            x=freq_shifted,
            y=psd_db_shifted,
            mode='lines',
            name='PSD',
            line=dict(color='cyan', width=1)
        ))
        
        # Add estimated noise floor line
        fig_psd.add_trace(go.Scatter(
            x=[freq_shifted[0], freq_shifted[-1]],
            y=[noise_floor_db, noise_floor_db],
            mode='lines',
            name=f'Noise Floor Est. ({noise_floor_db:.1f} dB)',
            line=dict(color='magenta', width=2, dash='dash')
        ))
        
        fig_psd.update_layout(
            title="Power Spectral Density",
            xaxis_title="Normalized Frequency",
            yaxis_title="Power (dB)",
            height=500,
            hovermode='x',
            showlegend=True,
            template='plotly_dark'
        )
        workflow.add_plot(fig_psd)
    
    # Amplitude histogram
    if amplitude is not None:
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
            height=400,
            template='plotly_dark'
        )
        workflow.add_plot(fig)
    
    quality_report = context.get("quality_report")
    if quality_report:
        workflow.add_table("Quality Assessment", quality_report)
    
    # Tables at the end
    amplitude_stats = context.get("amplitude_stats")
    if amplitude_stats:
        workflow.add_table("Amplitude Statistics", amplitude_stats)
    
    phase_stats = context.get("phase_stats")
    if phase_stats:
        workflow.add_table("Phase Statistics", phase_stats)
    
    iq_stats = context.get("iq_stats")
    if iq_stats:
        workflow.add_table("I/Q Statistics", iq_stats)
    
    quality_metrics = context.get("quality_metrics")
    if quality_metrics:
        workflow.add_table("Quality Metrics", quality_metrics)
