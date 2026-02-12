"""
Signal Analysis Plotting Functions
Creates all plots for signal analysis workflow as graph nodes
"""
import numpy as np
import plotly.graph_objects as go


def plot_psd(inputs):
    """
    Compute and plot Power Spectral Density in fast-time frequency domain
    
    Args:
        inputs: dict with:
            - pulses: 2D array of extracted pulses (num_pulses, num_range_samples)
            - sample_rate_hz: Sample rate in Hz
        
    Returns:
        dict: {'fig_psd': plotly figure}
    """
    pulses = inputs.get('pulses')
    sample_rate_hz = inputs.get('sample_rate_hz')
    
    if pulses is None:
        return {'fig_psd': None}
    
    num_pulses, num_range = pulses.shape
    
    # Compute FFT along fast-time dimension (range)
    fft_data = np.fft.fftshift(np.fft.fft(pulses, axis=1), axes=1)
    psd = np.mean(np.abs(fft_data)**2, axis=0)  # Average over pulses
    psd_db = 10 * np.log10(psd + 1e-10)
    
    # Frequency axis (range frequency bins)
    range_freqs = np.fft.fftshift(np.fft.fftfreq(num_range, d=1/sample_rate_hz))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=range_freqs / 1e6,  # Convert to MHz
        y=psd_db,
        mode='lines',
        line=dict(color='cyan', width=2),
        name='PSD'
    ))
    
    fig.update_layout(
        title="Power Spectral Density (Fast-Time Frequency Domain)",
        xaxis_title="Frequency (MHz)",
        yaxis_title="Power (dB)",
        height=400,
        template='plotly_dark',
        showlegend=False
    )
    
    return {'fig_psd': fig}


def plot_amplitude_heatmap(inputs):
    """
    Create amplitude heatmap with interactive colormap sliders
    
    Args:
        inputs: dict with:
            - amplitude: 2D amplitude array (num_pulses, num_range_samples)
            - downsample_factor: Range dimension downsampling factor
        
    Returns:
        dict: {'fig_amplitude_heatmap': plotly figure}
    """
    amplitude = inputs.get('amplitude')
    downsample_factor = inputs.get('downsample_factor', 1)
    
    if amplitude is None:
        return {'fig_amplitude_heatmap': None}
    
    # Downsample in range dimension if requested
    amplitude_plot = amplitude[:, ::downsample_factor] if downsample_factor > 1 else amplitude
    
    # Auto-downsample pulse dimension for very large heatmaps
    # Target: ~500k data points max to prevent JSON serialization errors
    max_heatmap_points = 500000
    total_points = amplitude_plot.shape[0] * amplitude_plot.shape[1]
    if total_points > max_heatmap_points:
        downsample_pulses = int(np.ceil(np.sqrt(total_points / max_heatmap_points)))
        amplitude_plot = amplitude_plot[::downsample_pulses, :]
    
    amp_db = 20 * np.log10(amplitude_plot + 1e-10)
    
    amp_min_default = float(np.percentile(amp_db, 1))
    amp_max_default = float(np.percentile(amp_db, 99))
    
    fig = go.Figure(data=go.Heatmap(
        z=amp_db,
        colorscale='HSV',
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
    
    return {'fig_amplitude_heatmap': fig}


def plot_phase_heatmap(inputs):
    """
    Create phase heatmap with interactive colormap sliders
    
    Args:
        inputs: dict with:
            - phase: 2D phase array (num_pulses, num_range_samples) in radians
            - downsample_factor: Range dimension downsampling factor
        
    Returns:
        dict: {'fig_phase_heatmap': plotly figure}
    """
    phase = inputs.get('phase')
    downsample_factor = inputs.get('downsample_factor', 1)
    
    if phase is None:
        return {'fig_phase_heatmap': None}
    
    # Downsample in range dimension if requested
    phase_plot = phase[:, ::downsample_factor] if downsample_factor > 1 else phase
    
    # Auto-downsample pulse dimension for very large heatmaps
    # Target: ~500k data points max to prevent JSON serialization errors
    max_heatmap_points = 500000
    total_points = phase_plot.shape[0] * phase_plot.shape[1]
    if total_points > max_heatmap_points:
        downsample_pulses = int(np.ceil(np.sqrt(total_points / max_heatmap_points)))
        phase_plot = phase_plot[::downsample_pulses, :]
    
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
    
    return {'fig_phase_heatmap': fig}


def plot_amplitude_histogram(inputs):
    """
    Create amplitude distribution histogram
    
    Args:
        inputs: dict with:
            - amp_flat: Flattened amplitude values
        
    Returns:
        dict: {'fig_amplitude_histogram': plotly figure}
    """
    amp_flat = inputs.get('amp_flat')
    
    if amp_flat is None:
        return {'fig_amplitude_histogram': None}
    
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
    
    return {'fig_amplitude_histogram': fig}


def plot_phase_histogram(inputs):
    """
    Create phase distribution histogram
    
    Args:
        inputs: dict with:
            - phase_flat: Flattened phase values in radians
        
    Returns:
        dict: {'fig_phase_histogram': plotly figure}
    """
    phase_flat = inputs.get('phase_flat')
    
    if phase_flat is None:
        return {'fig_phase_histogram': None}
    
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
    
    return {'fig_phase_histogram': fig}
