"""
Range Doppler Plotting Node Functions

This module contains plotting node functions for the range-doppler workflow.
Each function takes inputs from the graph context and returns plotly figures.
"""

import numpy as np
import plotly.graph_objects as go
from crsd_inspector.workflows.src.util.wrappers import downsample_heatmap


def determine_data_type(inputs):
    """
    Determine if data is pulsed or continuous based on pulse detection
    
    Parameters
    ----------
    inputs : dict
        - num_pulses: Number of detected pulses
        - num_windows: Number of windows processed
    
    Returns
    -------
    dict
        - is_pulsed: Boolean indicating if data is pulsed
    """
    num_pulses = inputs.get('num_pulses', 0)
    num_windows = inputs.get('num_windows', 0)
    is_pulsed = num_pulses > 2 and num_windows > 0 and (num_pulses / num_windows) < 0.9
    
    return {'is_pulsed': is_pulsed}


def plot_fixed_prf_windows(inputs):
    """
    Plot fixed-PRF windowed data (heatmap for pulsed, 1D average for continuous)
    
    Parameters
    ----------
    inputs : dict
        - windows_2d_db: Windowed power data in dB
        - sample_rate_hz: Sample rate
        - max_prf_hz: Maximum PRF used for windowing
        - is_pulsed: Boolean indicating data type
    
    Returns
    -------
    dict
        - fig_windows: Plotly figure or None
    """
    windows_2d_db = inputs.get('windows_2d_db')
    sample_rate_hz = inputs.get('sample_rate_hz')
    max_prf_hz = inputs.get('max_prf_hz')
    is_pulsed = inputs.get('is_pulsed', False)
    
    if windows_2d_db is None or sample_rate_hz is None:
        return {'fig_windows': None}
    
    if is_pulsed:
        # Heatmap for pulsed data
        heatmap_data, skip_x, skip_y = downsample_heatmap(windows_2d_db, max_width=2000, max_height=1000)
        fast_time_us = np.arange(heatmap_data.shape[1]) * skip_x / sample_rate_hz * 1e6
        window_numbers = np.arange(heatmap_data.shape[0]) * skip_y
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=fast_time_us,
            y=window_numbers,
            colorscale='HSV',
            colorbar=dict(title='Power (dB)'),
            zmin=heatmap_data.min(),
            zmax=heatmap_data.max(),
        ))
        fig.update_layout(
            title=f"Reshaped Pulse Array (Fixed {max_prf_hz} Hz Sampling)",
            xaxis_title="Fast Time (μs)",
            yaxis_title="Window Number",
            template='plotly_dark',
            height=700
        )
    else:
        # 1D average power profile for continuous data
        avg_power_db = np.mean(windows_2d_db, axis=0)
        fast_time_us = np.arange(len(avg_power_db)) / sample_rate_hz * 1e6
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fast_time_us, y=avg_power_db,
            mode='lines', name='Average Power',
            line=dict(color='cyan', width=1)
        ))
        fig.update_layout(
            title="Average Power Profile (Continuous Waveform)",
            xaxis_title="Fast Time (μs)",
            yaxis_title="Power (dB)",
            template='plotly_dark',
            height=600
        )
    
    return {'fig_windows': fig}


def plot_pulse_detection(inputs):
    """
    Plot pulse detection k-means clustering results
    
    Parameters
    ----------
    inputs : dict
        - window_powers: Power of each window
        - has_pulse: Boolean array indicating pulse presence
        - power_threshold: Detection threshold
    
    Returns
    -------
    dict
        - fig_detection: Plotly figure or None
    """
    window_powers = inputs.get('window_powers')
    has_pulse = inputs.get('has_pulse')
    power_threshold = inputs.get('power_threshold')
    
    if window_powers is None or has_pulse is None or power_threshold is None:
        return {'fig_detection': None}
    
    window_powers_db = 10 * np.log10(window_powers**2 + 1e-12)
    threshold_db = 10 * np.log10(power_threshold**2 + 1e-12)
    
    fig = go.Figure()
    
    rejected_indices = np.where(~has_pulse)[0]
    if len(rejected_indices) > 0:
        fig.add_trace(go.Scatter(
            x=rejected_indices,
            y=window_powers_db[rejected_indices],
            mode='markers',
            name='Rejected (Empty)',
            marker=dict(color='red', size=4, opacity=0.6)
        ))
    
    pulse_indices = np.where(has_pulse)[0]
    if len(pulse_indices) > 0:
        fig.add_trace(go.Scatter(
            x=pulse_indices,
            y=window_powers_db[pulse_indices],
            mode='markers',
            name='Detected (Pulse)',
            marker=dict(color='cyan', size=4)
        ))
    
    fig.add_trace(go.Scatter(
        x=[0, len(window_powers)-1],
        y=[threshold_db, threshold_db],
        mode='lines',
        name='Decision Boundary',
        line=dict(color='yellow', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Pulse Detection via K-Means Clustering",
        xaxis_title="Window Number",
        yaxis_title="Window Power (dB)",
        template='plotly_dark',
        height=600,
        showlegend=True
    )
    
    return {'fig_detection': fig}


def plot_detected_pulses(inputs):
    """
    Plot detected pulses (heatmap for pulsed, amplitude/phase for continuous)
    
    Parameters
    ----------
    inputs : dict
        - pulses_2d_db: Detected pulses in dB
        - pulses_2d: Complex pulse data
        - sample_rate_hz: Sample rate
        - is_pulsed: Boolean indicating data type
    
    Returns
    -------
    dict
        - fig_pulses_heatmap: Heatmap figure or None
        - fig_amplitude: Amplitude figure or None
        - fig_phase: Phase figure or None
    """
    pulses_2d_db = inputs.get('pulses_2d_db')
    pulses_2d = inputs.get('pulses_2d')
    sample_rate_hz = inputs.get('sample_rate_hz')
    is_pulsed = inputs.get('is_pulsed', False)
    
    result = {'fig_pulses_heatmap': None, 'fig_amplitude': None, 'fig_phase': None}
    
    if is_pulsed and pulses_2d_db is not None and sample_rate_hz is not None:
        # Heatmap for pulsed data
        heatmap_pulses, skip_x_p, skip_y_p = downsample_heatmap(pulses_2d_db, max_width=2000, max_height=1000)
        fast_time_pulses_us = np.arange(heatmap_pulses.shape[1]) * skip_x_p / sample_rate_hz * 1e6
        pulse_indices = np.arange(heatmap_pulses.shape[0]) * skip_y_p
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_pulses,
            x=fast_time_pulses_us,
            y=pulse_indices,
            colorscale='HSV',
            colorbar=dict(title='Power (dB)'),
            zmin=heatmap_pulses.min(),
            zmax=heatmap_pulses.max(),
        ))
        fig.update_layout(
            title=f"Detected Pulses Only (Empty Windows Removed)",
            xaxis_title="Fast Time (μs)",
            yaxis_title="Pulse Number",
            template='plotly_dark',
            height=700
        )
        result['fig_pulses_heatmap'] = fig
        
    elif not is_pulsed and pulses_2d is not None and len(pulses_2d) > 0 and sample_rate_hz is not None:
        # Amplitude and phase plots for continuous data
        avg_signal = np.mean(pulses_2d, axis=0)
        fast_time_us = np.arange(len(avg_signal)) / sample_rate_hz * 1e6
        
        # Amplitude plot
        fig_amp = go.Figure()
        fig_amp.add_trace(go.Scatter(
            x=fast_time_us, y=np.abs(avg_signal),
            mode='lines', name='Signal Amplitude',
            line=dict(color='cyan', width=1)
        ))
        fig_amp.update_layout(
            title="Signal Amplitude (Continuous Waveform)",
            xaxis_title="Fast Time (μs)",
            yaxis_title="Amplitude",
            template='plotly_dark',
            height=600
        )
        result['fig_amplitude'] = fig_amp
        
        # Phase plot
        fig_phase = go.Figure()
        fig_phase.add_trace(go.Scatter(
            x=fast_time_us, y=np.angle(avg_signal),
            mode='lines', name='Signal Phase',
            line=dict(color='magenta', width=1)
        ))
        fig_phase.update_layout(
            title="Signal Phase (Continuous Waveform)",
            xaxis_title="Fast Time (μs)",
            yaxis_title="Phase (radians)",
            template='plotly_dark',
            height=600
        )
        result['fig_phase'] = fig_phase
    
    return result


def plot_pri_sequence(inputs):
    """
    Plot PRI sequence with PRF bounds
    
    Parameters
    ----------
    inputs : dict
        - pris_us_filtered: Filtered PRI values in microseconds
        - min_prf_hz: Minimum PRF (original or detected)
        - max_prf_hz: Maximum PRF (original or detected)
        - detected_min_prf_hz: Auto-detected min PRF (optional)
        - detected_max_prf_hz: Auto-detected max PRF (optional)
        - auto_detect_prf: Boolean indicating if auto-detect was used
        - is_pulsed: Boolean indicating data type
    
    Returns
    -------
    dict
        - fig_pri: Plotly figure or None
    """
    pris_us = inputs.get('pris_us_filtered')
    min_prf_hz = inputs.get('min_prf_hz')
    max_prf_hz = inputs.get('max_prf_hz')
    detected_min_prf_hz = inputs.get('detected_min_prf_hz')
    detected_max_prf_hz = inputs.get('detected_max_prf_hz')
    auto_detect_prf = inputs.get('auto_detect_prf', False)
    is_pulsed = inputs.get('is_pulsed', False)
    
    if not is_pulsed or pris_us is None or len(pris_us) == 0:
        return {'fig_pri': None}
    
    # Use detected PRFs if auto-detect is enabled
    prf_min_display = detected_min_prf_hz if auto_detect_prf and detected_min_prf_hz else min_prf_hz
    prf_max_display = detected_max_prf_hz if auto_detect_prf and detected_max_prf_hz else max_prf_hz
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(pris_us)),
        y=pris_us,
        mode='markers+lines',
        name='PRI',
        marker=dict(color='cyan', size=4),
        line=dict(color='cyan', width=1)
    ))
    
    min_pri_us = 1e6 / prf_max_display
    max_pri_us = 1e6 / prf_min_display
    fig.add_trace(go.Scatter(
        x=[0, len(pris_us)-1],
        y=[min_pri_us, min_pri_us],
        mode='lines',
        name=f'Min PRI ({min_pri_us:.1f} μs, {prf_max_display:.0f} Hz)',
        line=dict(color='yellow', width=1, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=[0, len(pris_us)-1],
        y=[max_pri_us, max_pri_us],
        mode='lines',
        name=f'Max PRI ({max_pri_us:.1f} μs, {prf_min_display:.0f} Hz)',
        line=dict(color='orange', width=1, dash='dash')
    ))
    fig.update_layout(
        title="Pulse Repetition Intervals (PRI) Between Consecutive Pulses",
        xaxis_title="Pulse Pair Number",
        yaxis_title="PRI (μs)",
        template='plotly_dark',
        height=600,
        showlegend=True
    )
    
    return {'fig_pri': fig}


def plot_prf_clusters(inputs):
    """
    Plot PRF clustering results
    
    Parameters
    ----------
    inputs : dict
        - pris_us: Original PRI values (before filtering)
        - detected_prfs_hz: Detected PRF centers
        - pri_clusters: Cluster assignments
        - auto_detect_prf: Boolean indicating if auto-detect was used
        - use_fixed_prfs: Boolean indicating if fixed PRFs were used
        - is_pulsed: Boolean indicating data type
    
    Returns
    -------
    dict
        - fig_prf_clusters: Plotly figure or None
    """
    pris_us = inputs.get('pris_us')
    detected_prfs_hz = inputs.get('detected_prfs_hz')
    pri_clusters = inputs.get('pri_clusters')
    auto_detect_prf = inputs.get('auto_detect_prf', False)
    use_fixed_prfs = inputs.get('use_fixed_prfs', False)
    is_pulsed = inputs.get('is_pulsed', False)
    
    if not is_pulsed or not (auto_detect_prf or use_fixed_prfs):
        return {'fig_prf_clusters': None}
    
    if detected_prfs_hz is None or pris_us is None or len(pris_us) == 0:
        return {'fig_prf_clusters': None}
    
    # Convert PRIs to PRFs
    prfs_hz = 1e6 / pris_us
    
    fig = go.Figure()
    
    # Plot PRF values colored by cluster
    if pri_clusters is not None:
        for cluster_id in range(len(detected_prfs_hz)):
            cluster_mask = pri_clusters == cluster_id
            if np.any(cluster_mask):
                fig.add_trace(go.Scatter(
                    x=np.where(cluster_mask)[0],
                    y=prfs_hz[cluster_mask],
                    mode='markers',
                    name=f'Cluster {cluster_id+1}: {detected_prfs_hz[cluster_id]:.0f} Hz',
                    marker=dict(size=6, opacity=0.7)
                ))
    else:
        fig.add_trace(go.Scatter(
            x=np.arange(len(prfs_hz)),
            y=prfs_hz,
            mode='markers',
            name='PRF Values',
            marker=dict(color='cyan', size=4)
        ))
    
    # Add horizontal lines for detected PRF centers
    for i, prf_hz in enumerate(detected_prfs_hz):
        fig.add_trace(go.Scatter(
            x=[0, len(prfs_hz)-1],
            y=[prf_hz, prf_hz],
            mode='lines',
            name=f'PRF {i+1}: {prf_hz:.0f} Hz',
            line=dict(width=2, dash='dash'),
            showlegend=False
        ))
    
    prf_title = "Fixed PRF Snapping" if use_fixed_prfs else "Auto-Detected PRF Clusters (Cluster & Snap Method)"
    fig.update_layout(
        title=prf_title,
        xaxis_title="Pulse Pair Number",
        yaxis_title="PRF (Hz)",
        template='plotly_dark',
        height=600,
        showlegend=True
    )
    
    return {'fig_prf_clusters': fig}


def plot_extracted_pulses(inputs):
    """
    Plot PRI-based extracted pulses
    
    Parameters
    ----------
    inputs : dict
        - pulses_extracted_db: Extracted pulses in dB
        - sample_rate_hz: Sample rate
        - num_pulses: Number of pulses
        - is_pulsed: Boolean indicating data type
    
    Returns
    -------
    dict
        - fig_extracted: Plotly figure or None
    """
    pulses_extracted_db = inputs.get('pulses_extracted_db')
    sample_rate_hz = inputs.get('sample_rate_hz')
    num_pulses = inputs.get('num_pulses')
    is_pulsed = inputs.get('is_pulsed', False)
    
    if not is_pulsed or pulses_extracted_db is None or sample_rate_hz is None:
        return {'fig_extracted': None}
    
    heatmap_extracted, skip_x_e, skip_y_e = downsample_heatmap(pulses_extracted_db, max_width=2000, max_height=1000)
    fast_time_extracted_us = np.arange(heatmap_extracted.shape[1]) * skip_x_e / sample_rate_hz * 1e6
    pulse_indices_extracted = np.arange(heatmap_extracted.shape[0]) * skip_y_e
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_extracted,
        x=fast_time_extracted_us,
        y=pulse_indices_extracted,
        colorscale='HSV',
        colorbar=dict(title='Power (dB)'),
        zmin=heatmap_extracted.min(),
        zmax=heatmap_extracted.max(),
    ))
    fig.update_layout(
        title=f"PRI-Based Extracted Pulses (n={num_pulses}, Pre-Aligned)",
        xaxis_title="Fast Time (μs)",
        yaxis_title="Pulse Number",
        template='plotly_dark',
        height=700
    )
    
    return {'fig_extracted': fig}


def plot_range_doppler(inputs):
    """
    Plot range-doppler map with interactive controls
    
    Parameters
    ----------
    inputs : dict
        - range_doppler_db: Range-doppler map in dB
        - doppler_freqs_hz: Doppler frequency axis
        - sample_rate_hz: Sample rate
        - num_pulses: Number of pulses
        - is_uniform_prf: Boolean indicating uniform PRF
        - is_pulsed: Boolean indicating data type
    
    Returns
    -------
    dict
        - fig_range_doppler: Plotly figure or None
    """
    range_doppler_db = inputs.get('range_doppler_db')
    doppler_freqs_hz = inputs.get('doppler_freqs_hz')
    sample_rate_hz = inputs.get('sample_rate_hz')
    num_pulses = inputs.get('num_pulses')
    is_uniform_prf = inputs.get('is_uniform_prf', False)
    is_pulsed = inputs.get('is_pulsed', False)
    
    if not is_pulsed or range_doppler_db is None or doppler_freqs_hz is None or sample_rate_hz is None:
        return {'fig_range_doppler': None}
    
    heatmap_rd, skip_x_rd, skip_y_rd = downsample_heatmap(range_doppler_db, max_width=2000, max_height=1000)
    fast_time_rd_us = np.arange(heatmap_rd.shape[1]) * skip_x_rd / sample_rate_hz * 1e6
    doppler_freqs_display = doppler_freqs_hz[::skip_y_rd] if skip_y_rd < len(doppler_freqs_hz) else doppler_freqs_hz
    
    # Calculate default dynamic range
    rd_peak = np.max(heatmap_rd)
    rd_min_default = rd_peak - 60
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_rd,
        x=fast_time_rd_us,
        y=doppler_freqs_display[:heatmap_rd.shape[0]],
        colorscale='viridis',
        colorbar=dict(title='Power (dB)'),
        zmin=rd_min_default,
        zmax=rd_peak,
    ))
    
    # Create interactive colormap sliders
    rd_steps_min = []
    rd_steps_max = []
    rd_vals_min = np.linspace(np.min(heatmap_rd), rd_peak - 10, 20)
    rd_vals_max = np.linspace(rd_peak - 70, rd_peak, 20)
    
    for val in rd_vals_min:
        rd_steps_min.append(dict(
            method="restyle",
            args=[{"zmin": val}],
            label=f"{val:.0f}"
        ))
    
    for val in rd_vals_max:
        rd_steps_max.append(dict(
            method="restyle",
            args=[{"zmax": val}],
            label=f"{val:.0f}"
        ))
    
    # Update title based on whether FFT or NUFFT was used
    doppler_method = "FFT, Uniform PRF" if is_uniform_prf else "NUFFT, Non-Uniform Timing"
    fig.update_layout(
        title=f"Range-Doppler Map ({doppler_method}, {num_pulses} pulses)",
        xaxis_title='Range (μs)',
        yaxis_title='Doppler (Hz)',
        template='plotly_dark',
        height=800,
        sliders=[
            dict(
                active=10,
                yanchor="top",
                y=-0.08,
                xanchor="left",
                currentvalue=dict(
                    prefix="Min: ",
                    visible=True,
                    xanchor="right"
                ),
                pad=dict(b=10, t=10),
                len=0.42,
                x=0.0,
                steps=rd_steps_min
            ),
            dict(
                active=10,
                yanchor="top",
                y=-0.08,
                xanchor="right",
                currentvalue=dict(
                    prefix="Max: ",
                    visible=True,
                    xanchor="left"
                ),
                pad=dict(b=10, t=10),
                len=0.42,
                x=1.0,
                steps=rd_steps_max
            )
        ]
    )
    
    return {'fig_range_doppler': fig}
