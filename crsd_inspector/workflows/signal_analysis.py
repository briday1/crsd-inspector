"""
Signal Analysis Workflow
Amplitude-based signal analysis with PRF selection and pulse stacking
Generates amplitude/phase heatmaps, histograms, and statistics
"""
import numpy as np
from dagex import Graph
import plotly.graph_objects as go
from crsd_inspector.workflows.workflow import Workflow


# Workflow parameters
PARAMS = {
    'prf_hz': {
        'type': 'number',
        'label': 'PRF (Hz)',
        'default': 1000,
        'min': 1,
        'max': 100000,
        'help': 'Pulse Repetition Frequency for pulse stacking (will use from file if available)'
    },
    'tx_crsd_file': {
        'type': 'text',
        'label': 'TX CRSD File (optional)',
        'default': '',
        'help': 'Path to transmit waveform CRSD file (optional, will extract PRF if provided)'
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


# Create workflow instance
workflow = Workflow(
    name="Signal Analysis",
    description="Amplitude-based analysis with PRF selection and pulse stacking"
)


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
    context = dag.execute(True, 4)
    
    # Format and return results
    _format_results(context, metadata)
    return workflow.build()


def _extract_prf_from_metadata(metadata):
    """Extract PRF from CRSD file metadata if available"""
    prf_hz = None
    
    # Check file header KVPs
    file_header_kvps = metadata.get('file_header_kvps', {})
    if 'PRF_HZ' in file_header_kvps:
        try:
            prf_hz = float(file_header_kvps['PRF_HZ'])
        except:
            pass
    
    # Check if TX file provided
    tx_crsd_file = metadata.get('tx_crsd_file', '').strip()
    if tx_crsd_file and prf_hz is None:
        # Would need to load TX file and extract PRF
        # For now, just note it's available
        pass
    
    return prf_hz


def _create_graph(signal_data, metadata):
    """Create signal analysis workflow graph"""
    graph = Graph()
    
    # Extract parameters
    prf_hz = float(metadata.get('prf_hz', 1000))
    sample_rate_hz = float(metadata.get('sample_rate_hz', 100e6))
    num_pulses_to_stack = int(metadata.get('num_pulses_to_stack', 100))
    
    # Try to get PRF from metadata first
    prf_from_file = _extract_prf_from_metadata(metadata)
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
            'total_samples': total_samples
        },
        label="Provide Data",
        inputs=[],
        outputs=[
            ('signal_data', 'signal_data'),
            ('prf_hz', 'prf_hz'),
            ('sample_rate_hz', 'sample_rate_hz'),
            ('pri_samples', 'pri_samples'),
            ('num_pulses_to_stack', 'num_pulses_to_stack'),
            ('total_samples', 'total_samples')
        ]
    )
    
    # Pulse extraction node
    def extract_pulses(inputs):
        signal = inputs['signal_data']
        pri_samples = inputs['pri_samples']
        num_pulses = inputs['num_pulses_to_stack']
        total_samples = inputs['total_samples']
        
        # Calculate how many pulses we can extract
        max_pulses = total_samples // pri_samples
        # If num_pulses is -1, use all available pulses
        if num_pulses <= 0:
            actual_num_pulses = max_pulses
        else:
            actual_num_pulses = min(num_pulses, max_pulses)
        
        # Extract pulses
        pulses = np.zeros((actual_num_pulses, pri_samples), dtype=signal.dtype)
        for i in range(actual_num_pulses):
            start_idx = i * pri_samples
            end_idx = start_idx + pri_samples
            if end_idx <= total_samples:
                pulses[i, :] = signal[0, start_idx:end_idx]
        
        return {
            'pulses': pulses,
            'actual_num_pulses': actual_num_pulses
        }
    
    graph.add(
        extract_pulses,
        label="Extract Pulses",
        inputs=[
            ('signal_data', 'signal_data'),
            ('pri_samples', 'pri_samples'),
            ('num_pulses_to_stack', 'num_pulses_to_stack'),
            ('total_samples', 'total_samples')
        ],
        outputs=[
            ('pulses', 'pulses'),
            ('actual_num_pulses', 'actual_num_pulses')
        ]
    )
    
    # Compute statistics node
    def compute_statistics(inputs):
        pulses = inputs['pulses']
        
        amplitude = np.abs(pulses)
        phase = np.angle(pulses)
        
        # Flatten for overall statistics
        amp_flat = amplitude.flatten()
        phase_flat = phase.flatten()
        
        # Amplitude statistics
        amplitude_stats = {
            "Metric": ["Min", "Max", "Mean", "Std Dev", "Median"],
            "Value": [
                f"{np.min(amp_flat):.4f}",
                f"{np.max(amp_flat):.4f}",
                f"{np.mean(amp_flat):.4f}",
                f"{np.std(amp_flat):.4f}",
                f"{np.median(amp_flat):.4f}"
            ]
        }
        
        # Phase statistics
        phase_stats = {
            "Metric": ["Min (rad)", "Max (rad)", "Mean (rad)", "Std Dev (rad)"],
            "Value": [
                f"{np.min(phase_flat):.4f}",
                f"{np.max(phase_flat):.4f}",
                f"{np.mean(phase_flat):.4f}",
                f"{np.std(phase_flat):.4f}"
            ]
        }
        
        # Quality metrics
        signal_power = np.mean(amp_flat**2)
        noise_floor = np.percentile(amp_flat, 10)**2
        snr_db = 10 * np.log10(signal_power / (noise_floor + 1e-10))
        dynamic_range_db = 20 * np.log10(np.max(amp_flat) / (np.min(amp_flat) + 1e-10))
        
        # Clipping detection
        max_amp = np.max(amp_flat)
        clipping_threshold = 0.99 * max_amp
        clipped_samples = np.sum(amp_flat > clipping_threshold)
        clipping_percentage = 100.0 * clipped_samples / amp_flat.size
        
        quality_metrics = {
            "Metric": ["SNR", "Dynamic Range", "Clipping %", "Total Samples", "Clipped Samples"],
            "Value": [
                f"{snr_db:.2f} dB",
                f"{dynamic_range_db:.2f} dB",
                f"{clipping_percentage:.2f}%",
                f"{amp_flat.size}",
                f"{clipped_samples}"
            ]
        }
        
        # I/Q statistics
        iq_stats = {
            "Component": ["I (Real)", "Q (Imaginary)"],
            "Mean": [f"{np.mean(pulses.real):.4f}", f"{np.mean(pulses.imag):.4f}"],
            "Std Dev": [f"{np.std(pulses.real):.4f}", f"{np.std(pulses.imag):.4f}"]
        }
        
        return {
            'amplitude': amplitude,
            'phase': phase,
            'amplitude_stats': amplitude_stats,
            'phase_stats': phase_stats,
            'quality_metrics': quality_metrics,
            'iq_stats': iq_stats,
            'amp_flat': amp_flat,
            'phase_flat': phase_flat,
            'pulses': pulses  # Pass through for PSD calculation
        }
    
    graph.add(
        compute_statistics,
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
            ('pulses', 'pulses')  # Pass through for PSD
        ]
    )
    
    return graph


def _format_results(context, metadata):
    """Format workflow results for display"""
    
    # Get parameters
    prf_hz = context.get('prf_hz', metadata.get('prf_hz', 1000))
    sample_rate_hz = context.get('sample_rate_hz', metadata.get('sample_rate_hz', 100e6))
    actual_num_pulses = context.get('actual_num_pulses', 0)
    pri_samples = context.get('pri_samples', 0)
    downsample_factor = int(metadata.get('downsample_range_factor', 1))
    
    # Parameters table
    params_table = {
        "Parameter": ["PRF", "Sample Rate", "PRI (samples)", "Pulses Extracted", "Range Downsample"],
        "Value": [
            f"{prf_hz:.1f} Hz",
            f"{sample_rate_hz/1e6:.1f} MHz",
            f"{pri_samples}",
            f"{actual_num_pulses}",
            f"{downsample_factor}x"
        ]
    }
    workflow.add_table("Analysis Parameters", params_table)
    
    # Amplitude heatmap
    amplitude = context.get('amplitude')
    if amplitude is not None:
        # Downsample in range dimension if requested
        if downsample_factor > 1:
            amplitude_plot = amplitude[:, ::downsample_factor]
        else:
            amplitude_plot = amplitude
        
        amp_db = 20 * np.log10(amplitude_plot + 1e-10)
        
        amp_min_default = float(np.percentile(amp_db, 1))
        amp_max_default = float(np.percentile(amp_db, 99))
        
        fig = go.Figure(data=go.Heatmap(
            z=amp_db,
            colorscale='Jet',
            zmin=amp_min_default,
            zmax=amp_max_default,
            colorbar=dict(title="Amplitude (dB)", x=1.15)
        ))
        
        # Add sliders for colormap control
        steps_min = []
        steps_max = []
        range_vals_min = np.linspace(np.min(amp_db), amp_max_default, 20)
        range_vals_max = np.linspace(amp_min_default, np.max(amp_db), 20)
        
        for val in range_vals_min:
            steps_min.append(dict(method="restyle", args=[{"zmin": val}], label=f"{val:.0f}"))
        
        for val in range_vals_max:
            steps_max.append(dict(method="restyle", args=[{"zmax": val}], label=f"{val:.0f}"))
        
        fig.update_layout(
            title="Amplitude Heatmap (Pulse-Stacked)",
            xaxis_title="Range Sample",
            yaxis_title="Pulse Number",
            height=600,
            template='plotly_dark',
            sliders=[
                dict(active=10, yanchor="top", y=-0.15, xanchor="left",
                     currentvalue=dict(prefix="Min: ", visible=True, xanchor="right"),
                     pad=dict(b=10, t=10), len=0.42, x=0.0, steps=steps_min),
                dict(active=10, yanchor="top", y=-0.15, xanchor="right",
                     currentvalue=dict(prefix="Max: ", visible=True, xanchor="left"),
                     pad=dict(b=10, t=10), len=0.42, x=1.0, steps=steps_max)
            ]
        )
        workflow.add_plot(fig)
    
    # Phase heatmap
    phase = context.get('phase')
    if phase is not None:
        # Downsample in range dimension if requested
        if downsample_factor > 1:
            phase_plot = phase[:, ::downsample_factor]
        else:
            phase_plot = phase
        
        phase_min_default = float(np.percentile(phase_plot, 1))
        phase_max_default = float(np.percentile(phase_plot, 99))
        
        fig = go.Figure(data=go.Heatmap(
            z=phase_plot,
            colorscale='HSV',
            zmin=phase_min_default,
            zmax=phase_max_default,
            colorbar=dict(title="Phase (rad)", x=1.15),
            zmid=0
        ))
        
        steps_min = []
        steps_max = []
        range_vals_min = np.linspace(-np.pi, 0, 15)
        range_vals_max = np.linspace(0, np.pi, 15)
        
        for val in range_vals_min:
            steps_min.append(dict(method="restyle", args=[{"zmin": val}], label=f"{val:.2f}"))
        
        for val in range_vals_max:
            steps_max.append(dict(method="restyle", args=[{"zmax": val}], label=f"{val:.2f}"))
        
        fig.update_layout(
            title="Phase Heatmap (Pulse-Stacked)",
            xaxis_title="Range Sample",
            yaxis_title="Pulse Number",
            height=600,
            template='plotly_dark',
            sliders=[
                dict(active=7, yanchor="top", y=-0.15, xanchor="left",
                     currentvalue=dict(prefix="Min: ", visible=True, xanchor="right"),
                     pad=dict(b=10, t=10), len=0.42, x=0.0, steps=steps_min),
                dict(active=7, yanchor="top", y=-0.15, xanchor="right",
                     currentvalue=dict(prefix="Max: ", visible=True, xanchor="left"),
                     pad=dict(b=10, t=10), len=0.42, x=1.0, steps=steps_max)
            ]
        )
        workflow.add_plot(fig)
    
    # Amplitude histogram
    amp_flat = context.get('amp_flat')
    if amp_flat is not None:
        hist, edges = np.histogram(amp_flat, bins=100)
        bin_centers = (edges[:-1] + edges[1:]) / 2
        
        fig = go.Figure(data=go.Bar(x=bin_centers, y=hist, marker_color='cyan'))
        fig.update_layout(
            title="Amplitude Distribution",
            xaxis_title="Amplitude",
            yaxis_title="Count (log scale)",
            yaxis_type="log",
            height=400,
            template='plotly_dark'
        )
        workflow.add_plot(fig)
    
    # Phase histogram
    phase_flat = context.get('phase_flat')
    if phase_flat is not None:
        hist, edges = np.histogram(phase_flat, bins=100)
        bin_centers = (edges[:-1] + edges[1:]) / 2
        
        fig = go.Figure(data=go.Bar(x=bin_centers, y=hist, marker_color='magenta'))
        fig.update_layout(
            title="Phase Distribution",
            xaxis_title="Phase (radians)",
            yaxis_title="Count",
            height=400,
            template='plotly_dark'
        )
        workflow.add_plot(fig)
    
    # Tables
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
    
    # Power Spectral Density
    pulses = context.get('pulses')
    if pulses is not None:
        # Compute PSD along slow-time (Doppler) for each range bin
        # Average over selected range bins for visualization
        num_pulses, num_range = pulses.shape
        
        # Select middle range bins for PSD calculation
        mid_start = num_range // 4
        mid_end = 3 * num_range // 4
        pulses_mid = pulses[:, mid_start:mid_end]
        
        # Compute FFT along slow-time dimension (Doppler)
        fft_data = np.fft.fftshift(np.fft.fft(pulses_mid, axis=0), axes=0)
        psd = np.mean(np.abs(fft_data)**2, axis=1)  # Average over range bins
        psd_db = 10 * np.log10(psd + 1e-10)
        
        # Frequency axis (Doppler bins)
        doppler_freqs = np.fft.fftshift(np.fft.fftfreq(num_pulses, d=1/prf_hz))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=doppler_freqs,
            y=psd_db,
            mode='lines',
            line=dict(color='cyan', width=2),
            name='PSD'
        ))
        
        fig.update_layout(
            title="Power Spectral Density (Doppler Domain)",
            xaxis_title="Doppler Frequency (Hz)",
            yaxis_title="Power (dB)",
            height=400,
            template='plotly_dark',
            showlegend=False
        )
        workflow.add_plot(fig)
