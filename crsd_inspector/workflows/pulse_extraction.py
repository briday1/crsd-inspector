"""
Pulse Extraction Workflow

Simple matched filter processing for continuous radar data.
"""

import numpy as np
import plotly.graph_objects as go
from crsd_inspector.workflows.workflow import Workflow
from scipy.signal import correlate
from sklearn.cluster import KMeans
import finufft


# Create workflow instance
workflow = Workflow(
    name="Pulse Extraction",
    description="Matched filter processing for continuous radar data"
)

# Workflow parameters
PARAMS = {
    'window_type': {
        'label': 'Range Window',
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
        'default': 1200,
    },
}


def _make_window(n, window_type):
    if (window_type is None) or (window_type == 'none'):
        return np.ones(n)
    if window_type == 'hamming':
        return np.hamming(n)
    if window_type == 'hanning':
        return np.hanning(n)
    if window_type == 'blackman':
        return np.blackman(n)
    return np.ones(n)


def detect_pulse_toas(mf_output, sample_rate_hz, min_prf_hz, threshold_db=10):
    """
    Detect pulse time-of-arrivals (TOAs) using leading edge detection.
    
    Args:
        mf_output: 1D matched filter output (complex)
        sample_rate_hz: Sample rate in Hz
        min_prf_hz: Minimum PRF (slowest pulse rate) - used to set minimum spacing
        threshold_db: Detection threshold in dB above noise floor
    
    Returns:
        toa_indices: Array of TOA sample indices
    """
    # Compute magnitude in dB
    mf_mag = np.abs(mf_output)
    mf_db = 10 * np.log10(mf_mag ** 2 + 1e-12)
    
    # Estimate noise floor from lower percentile
    noise_floor = np.percentile(mf_db, 10)
    detection_threshold = noise_floor + threshold_db
    
    # Find samples above threshold
    above_threshold = mf_db > detection_threshold
    
    # Find leading edges (transitions from below to above threshold)
    edges = np.diff(above_threshold.astype(int))
    leading_edges = np.where(edges == 1)[0] + 1  # +1 because diff shifts by 1
    
    # Enforce minimum spacing (based on max PRF to avoid double detections)
    min_spacing = int(sample_rate_hz / min_prf_hz * 0.8)  # 80% of min PRI
    
    toa_indices = []
    last_toa = -min_spacing
    for edge_idx in leading_edges:
        if edge_idx - last_toa >= min_spacing:
            toa_indices.append(edge_idx)
            last_toa = edge_idx
    
    return np.array(toa_indices, dtype=int)


def extract_toa_windowed_2d(mf_output, toa_indices, min_prf_hz, max_prf_hz, sample_rate_hz):
    """
    Extract fixed-length pulse windows at each TOA to create 2D pulse array.
    
    Each window captures just the pulse itself (not the full PRI), so all
    pulses are aligned at the start of their respective rows.
    
    Args:
        mf_output: 1D matched filter output
        toa_indices: Array of TOA sample indices
        min_prf_hz: Minimum PRF
        max_prf_hz: Maximum PRF (determines window length - one pulse period)
        sample_rate_hz: Sample rate in Hz
    
    Returns:
        windows_2d: 2D array (num_pulses x window_length)
        window_length: Length of each window in samples
        toa_indices: Filtered TOA indices (ones that fit valid windows)
    """
    # Window length = one period at fastest PRF (captures single pulse)
    window_length = int(sample_rate_hz / max_prf_hz)
    
    # Filter TOAs to ensure valid windows
    valid_toas = []
    for toa in toa_indices:
        if toa + window_length <= len(mf_output):
            valid_toas.append(toa)
    
    valid_toas = np.array(valid_toas, dtype=int)
    num_pulses = len(valid_toas)
    
    # Extract windows - all starting at TOA so pulses are aligned
    windows_2d = np.zeros((num_pulses, window_length), dtype=mf_output.dtype)
    for i, toa in enumerate(valid_toas):
        windows_2d[i, :] = mf_output[toa:toa + window_length]
    
    return windows_2d, window_length, valid_toas


def compute_ppc_correlations(windows_2d):
    """
    Compute correlations between consecutive windows for PPC analysis.
    
    Correlates each window with the next window to extract the full
    correlation function, revealing pulse interval patterns.
    
    Args:
        windows_2d: 2D array (num_windows x window_length) from windowed data
    
    Returns:
        correlations: 2D array (num_pairs x correlation_length) of correlation magnitudes
        peak_positions: Array of peak positions (sample index within window)
        peak_values: Array of peak magnitudes
    """
    num_windows = windows_2d.shape[0]
    num_pairs = num_windows - 1
    window_length = windows_2d.shape[1]
    
    # Store all correlations
    correlations = np.zeros((num_pairs, window_length), dtype=float)
    peak_positions = np.zeros(num_windows, dtype=int)
    peak_values = np.zeros(num_windows)
    
    # First, get peak position in each window
    for i in range(num_windows):
        window_mag = np.abs(windows_2d[i, :])
        peak_idx = np.argmax(window_mag)
        peak_positions[i] = peak_idx
        peak_values[i] = window_mag[peak_idx]
    
    # Compute correlations between consecutive windows
    for i in range(num_pairs):
        corr = correlate(windows_2d[i+1, :], np.conj(windows_2d[i, :]), mode='same', method='fft')
        correlations[i, :] = np.abs(corr)
    
    return correlations, peak_positions, peak_values


def downsample_heatmap(data_2d, max_width=2000, max_height=1000):
    """
    Downsample 2D heatmap data for efficient rendering.
    
    Args:
        data_2d: 2D array (rows x cols)
        max_width: Maximum number of columns to keep
        max_height: Maximum number of rows to keep
    
    Returns:
        downsampled: Downsampled 2D array
        skip_x: Downsampling factor for columns
        skip_y: Downsampling factor for rows
    """
    rows, cols = data_2d.shape
    
    # Calculate downsampling factors
    skip_x = max(1, cols // max_width)
    skip_y = max(1, rows // max_height)
    
    # Downsample by taking every Nth sample
    downsampled = data_2d[::skip_y, ::skip_x]
    
    return downsampled, skip_x, skip_y


def apply_matched_filter(signal_data, reference_waveform, window_type='hamming'):
    """
    Apply matched filter to continuous signal data.

    Args:
        signal_data: Continuous signal (1 x num_samples) or (num_samples,)
        reference_waveform: Reference pulse for matched filtering
        window_type: Window function for range processing ('none', 'hamming', 'hanning', 'blackman')

    Returns:
        mf_output: Matched filter output (complex, same length as signal_data)
    """
    if signal_data.ndim > 1:
        signal_data = signal_data.ravel()

    ref_wfm = reference_waveform.copy()
    w = _make_window(len(ref_wfm), window_type)
    ref_wfm = ref_wfm * w

    mf_output = correlate(signal_data, np.conj(ref_wfm), mode='same', method='fft')
    return mf_output


def run_workflow(signal_data, metadata=None, **kwargs):
    workflow.clear()
    if metadata is None:
        metadata = {}

    sample_rate_hz = float(metadata.get('sample_rate_hz', 100e6))
    window_type = metadata.get('window_type', 'hamming')
    min_prf_hz = float(metadata.get('min_prf_hz', 800))
    max_prf_hz = float(metadata.get('max_prf_hz', 1200))
    tx_wfm = metadata.get('tx_wfm', None)

    if signal_data.ndim == 1:
        signal_data = signal_data[None, :]

    total_samples = int(signal_data.shape[1])
    total_time_ms = total_samples / sample_rate_hz * 1000
    shortest_pri_samples = int(sample_rate_hz / max_prf_hz)
    shortest_pri_us = shortest_pri_samples / sample_rate_hz * 1e6
    
    # Create summary table
    workflow.add_text("## Processing Parameters")
    
    file_header_kvps = metadata.get('file_header_kvps', {})
    
    # Build summary table data
    summary_rows = [
        ["Total Samples", f"{total_samples:,}"],
        ["Sample Rate", f"{sample_rate_hz/1e6:.1f} MHz"],
        ["Total Duration", f"{total_time_ms:.2f} ms"],
        ["Range Window", window_type],
        ["PRF Search Range", f"{min_prf_hz:.0f} - {max_prf_hz:.0f} Hz"],
        ["Shortest PRI", f"{shortest_pri_us:.2f} μs ({shortest_pri_samples} samples)"],
    ]
    
    if file_header_kvps:
        stagger_pattern = file_header_kvps.get('STAGGER_PATTERN', 'unknown')
        num_pulses_truth = int(file_header_kvps.get('NUM_PULSES', 0))
        num_targets = int(file_header_kvps.get('NUM_TARGETS', 0))
        summary_rows.extend([
            ["", ""],
            ["**Ground Truth**", ""],
            ["Stagger Pattern", stagger_pattern],
            ["Pulses (Truth)", str(num_pulses_truth)],
            ["Targets", str(num_targets)],
        ])
    
    workflow.add_table(["Parameter", "Value"], summary_rows)

    try:
        if tx_wfm is None:
            workflow.add_text("\nError: No reference waveform provided")
            return workflow.build()

        workflow.add_text("\n## Matched Filter Output")

        mf_output = apply_matched_filter(signal_data, tx_wfm, window_type=window_type)
        mf_output_db = 10 * np.log10(np.abs(mf_output) ** 2 + 1e-12)

        # Plot matched filter output
        time_axis_ms = np.arange(len(mf_output_db)) / sample_rate_hz * 1000.0
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=time_axis_ms, y=mf_output_db,
            mode='lines', name='Matched Filter Output',
            line=dict(color='cyan', width=1)
        ))
        fig1.update_layout(
            title="Matched Filter Output",
            xaxis_title="Time (ms)",
            yaxis_title="Power (dB)",
            template='plotly_dark',
            height=600
        )
        workflow.add_plot(fig1)

        # Reshape data according to shortest PRI
        workflow.add_text("\n## Fixed-PRF Pulse Array")
        
        # Reshape: extract windows every shortest_pri samples
        num_windows = (len(mf_output) - shortest_pri_samples) // shortest_pri_samples
        windows_2d = np.zeros((num_windows, shortest_pri_samples), dtype=mf_output.dtype)
        
        for i in range(num_windows):
            start_idx = i * shortest_pri_samples
            end_idx = start_idx + shortest_pri_samples
            windows_2d[i, :] = mf_output[start_idx:end_idx]
        
        # Create 2D heatmap
        windows_2d_db = 10 * np.log10(np.abs(windows_2d) ** 2 + 1e-12)
        
        # Downsample for rendering
        heatmap_data, skip_x, skip_y = downsample_heatmap(windows_2d_db, max_width=2000, max_height=1000)
        
        # Time axes for heatmap
        fast_time_us = np.arange(heatmap_data.shape[1]) * skip_x / sample_rate_hz * 1e6
        window_numbers = np.arange(heatmap_data.shape[0]) * skip_y
        
        fig2 = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=fast_time_us,
            y=window_numbers,
            colorscale='HSV',
            colorbar=dict(title='Power (dB)'),
            zmin=heatmap_data.min(),
            zmax=heatmap_data.max(),
        ))
        fig2.update_layout(
            title=f"Reshaped Pulse Array (Fixed {max_prf_hz} Hz Sampling)",
            xaxis_title="Fast Time (μs)",
            yaxis_title="Window Number",
            template='plotly_dark',
            height=700,
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(label="Auto", method="relayout", args=[{"coloraxis.cmin": None, "coloraxis.cmax": None}]),
                        dict(label="Full", method="relayout", args=[{"coloraxis.cmin": heatmap_data.min(), "coloraxis.cmax": heatmap_data.max()}]),
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                ),
            ],
            sliders=[
                dict(
                    active=100,
                    yanchor="top",
                    y=-0.1,
                    xanchor="left",
                    currentvalue=dict(prefix="Color Max: ", visible=True, xanchor="right"),
                    pad=dict(b=10, t=50),
                    len=0.45,
                    x=0.0,
                    steps=[
                        dict(method="relayout", args=[{"coloraxis.cmax": heatmap_data.min() + i * (heatmap_data.max() - heatmap_data.min()) / 100}], label=f"{heatmap_data.min() + i * (heatmap_data.max() - heatmap_data.min()) / 100:.1f}")
                        for i in range(101)
                    ]
                ),
                dict(
                    active=0,
                    yanchor="top",
                    y=-0.1,
                    xanchor="right",
                    currentvalue=dict(prefix="Color Min: ", visible=True, xanchor="left"),
                    pad=dict(b=10, t=50),
                    len=0.45,
                    x=1.0,
                    steps=[
                        dict(method="relayout", args=[{"coloraxis.cmin": heatmap_data.min() + i * (heatmap_data.max() - heatmap_data.min()) / 100}], label=f"{heatmap_data.min() + i * (heatmap_data.max() - heatmap_data.min()) / 100:.1f}")
                        for i in range(101)
                    ]
                ),
            ]
        )
        workflow.add_plot(fig2)

        # Pulse detection using k-means clustering
        workflow.add_text("\n## Pulse Detection (K-Means Clustering)")
        
        # Use k-means clustering to separate pulses from null/empty windows
        window_powers = np.mean(np.abs(windows_2d), axis=1)
        
        # Fit k-means with 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(window_powers.reshape(-1, 1))
        cluster_centers = kmeans.cluster_centers_.flatten()
        
        # Determine which cluster is the null/empty pulses (lower power)
        null_cluster = np.argmin(cluster_centers)
        pulse_cluster = 1 - null_cluster
        
        # Mark windows in the larger power cluster as having pulses
        has_pulse = (cluster_labels == pulse_cluster)
        
        # Track the original window number and start time for each pulse
        pulse_window_indices = np.where(has_pulse)[0]
        num_pulses = len(pulse_window_indices)
        
        # Start time of each window in seconds
        window_start_times_s = np.arange(num_windows) * shortest_pri_samples / sample_rate_hz
        pulse_start_times_s = window_start_times_s[pulse_window_indices]
        
        # Compute cluster boundary as threshold for display
        power_threshold = (cluster_centers[null_cluster] + cluster_centers[pulse_cluster]) / 2
        
        workflow.add_text(f"Detected **{num_pulses}** pulses (rejected {num_windows - num_pulses} empty windows)")
        
        # Plot window power statistic vs. k-means clusters
        window_powers_db = 10 * np.log10(window_powers**2 + 1e-12)
        threshold_db = 10 * np.log10(power_threshold**2 + 1e-12)
        
        fig2b = go.Figure()
        
        # Plot rejected windows (below threshold)
        rejected_indices = np.where(~has_pulse)[0]
        if len(rejected_indices) > 0:
            fig2b.add_trace(go.Scatter(
                x=rejected_indices,
                y=window_powers_db[rejected_indices],
                mode='markers',
                name='Rejected (Empty)',
                marker=dict(color='red', size=4, opacity=0.6)
            ))
        
        # Plot accepted windows (above threshold)
        accepted_indices = np.where(has_pulse)[0]
        if len(accepted_indices) > 0:
            fig2b.add_trace(go.Scatter(
                x=accepted_indices,
                y=window_powers_db[accepted_indices],
                mode='markers',
                name='Accepted (Pulse)',
                marker=dict(color='cyan', size=4)
            ))
        
        # Add decision boundary line
        fig2b.add_trace(go.Scatter(
            x=[0, num_windows-1],
            y=[threshold_db, threshold_db],
            mode='lines',
            name=f'Decision Boundary ({threshold_db:.1f} dB)',
            line=dict(color='yellow', width=2, dash='dash')
        ))
        
        fig2b.update_layout(
            title="Window Power Statistics (K-Means Clustering, k=2)",
            xaxis_title="Window Number",
            yaxis_title="Mean Power (dB)",
            template='plotly_dark',
            height=600,
            showlegend=True
        )
        workflow.add_plot(fig2b)
        
        # Extract only the windows with pulses
        pulses_2d = windows_2d[pulse_window_indices, :]
        
        # Create heatmap of accepted pulses only (compact, no empty windows)
        pulses_2d_db = 10 * np.log10(np.abs(pulses_2d) ** 2 + 1e-12)
        
        # Downsample for rendering
        heatmap_pulses, skip_x_p, skip_y_p = downsample_heatmap(pulses_2d_db, max_width=2000, max_height=1000)
        workflow.add_text(f"Pulses-only heatmap: {pulses_2d_db.shape} → {heatmap_pulses.shape} (downsampled {skip_x_p}x{skip_y_p})")
        
        # Time axes for pulses heatmap
        fast_time_pulses_us = np.arange(heatmap_pulses.shape[1]) * skip_x_p / sample_rate_hz * 1e6
        pulse_indices = np.arange(heatmap_pulses.shape[0]) * skip_y_p
        
        fig2c = go.Figure(data=go.Heatmap(
            z=heatmap_pulses,
            x=fast_time_pulses_us,
            y=pulse_indices,
            colorscale='HSV',
            colorbar=dict(title='Power (dB)'),
            zmin=heatmap_pulses.min(),
            zmax=heatmap_pulses.max(),
        ))
        fig2c.update_layout(
            title=f"Accepted Pulses Only (Empty Windows Removed, n={num_pulses})",
            xaxis_title="Fast Time (μs)",
            yaxis_title="Pulse Number (Compact)",
            template='plotly_dark',
            height=700,
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(label="Auto", method="relayout", args=[{"coloraxis.cmin": None, "coloraxis.cmax": None}]),
                        dict(label="Full", method="relayout", args=[{"coloraxis.cmin": heatmap_pulses.min(), "coloraxis.cmax": heatmap_pulses.max()}]),
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                ),
            ],
            sliders=[
                dict(
                    active=100,
                    yanchor="top",
                    y=-0.1,
                    xanchor="left",
                    currentvalue=dict(prefix="Color Max: ", visible=True, xanchor="right"),
                    pad=dict(b=10, t=50),
                    len=0.45,
                    x=0.0,
                    steps=[
                        dict(method="relayout", args=[{"coloraxis.cmax": heatmap_pulses.min() + i * (heatmap_pulses.max() - heatmap_pulses.min()) / 100}], label=f"{heatmap_pulses.min() + i * (heatmap_pulses.max() - heatmap_pulses.min()) / 100:.1f}")
                        for i in range(101)
                    ]
                ),
                dict(
                    active=0,
                    yanchor="top",
                    y=-0.1,
                    xanchor="right",
                    currentvalue=dict(prefix="Color Min: ", visible=True, xanchor="left"),
                    pad=dict(b=10, t=50),
                    len=0.45,
                    x=1.0,
                    steps=[
                        dict(method="relayout", args=[{"coloraxis.cmin": heatmap_pulses.min() + i * (heatmap_pulses.max() - heatmap_pulses.min()) / 100}], label=f"{heatmap_pulses.min() + i * (heatmap_pulses.max() - heatmap_pulses.min()) / 100:.1f}")
                        for i in range(101)
                    ]
                ),
            ]
        )
        workflow.add_plot(fig2c)
        
        workflow.add_text("\n## Pulse Pair Correlation")
        
        # Now do PPC on consecutive pulses (in the filtered list)
        num_pulse_pairs = num_pulses - 1
        peak_lags = np.zeros(num_pulse_pairs, dtype=int)
        peak_values = np.zeros(num_pulse_pairs)
        
        for i in range(num_pulse_pairs):
            # Correlate consecutive pulses
            corr = correlate(pulses_2d[i+1, :], np.conj(pulses_2d[i, :]), mode='same', method='fft')
            corr_mag = np.abs(corr)
            
            # Find peak lag
            peak_idx = np.argmax(corr_mag)
            lag = peak_idx - len(corr_mag) // 2  # Lag relative to center
            
            # Unwrap: convert to modulo window length, keep in [0, shortest_pri_samples)
            lag_unwrapped = lag % shortest_pri_samples
            
            peak_lags[i] = lag_unwrapped
            peak_values[i] = corr_mag[peak_idx]
        
        # Convert lags to time
        peak_lags_us = peak_lags / sample_rate_hz * 1e6
        
        # Plot PPC offsets
        if num_pulse_pairs > 0:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=np.arange(num_pulse_pairs),
                y=peak_lags_us,
                mode='markers',
                name='PPC Offsets',
                marker=dict(color='cyan', size=4),
                text=[f"Windows {pulse_window_indices[i]} → {pulse_window_indices[i+1]}" 
                      for i in range(num_pulse_pairs)],
                hovertemplate='Pulse Pair %{x}<br>Lag: %{y:.2f} μs<br>%{text}<extra></extra>'
            ))
            fig3.update_layout(
                title="PPC Peak Offsets Between Consecutive Pulses (Empty Windows Filtered)",
                xaxis_title="Pulse Pair Number",
                yaxis_title="Offset (μs)",
                template='plotly_dark',
                height=600
            )
            workflow.add_plot(fig3)

        # Pulse timing analysis
        workflow.add_text("\n## Pulse Timing & PRI Analysis")
        
        # For each pulse, detect where the pulse peak is within its window
        # This gives us the intra-window offset (where the pulse actually starts)
        intra_window_offsets = np.zeros(num_pulses, dtype=int)
        for i in range(num_pulses):
            pulse_mag = np.abs(pulses_2d[i, :])
            peak_idx = np.argmax(pulse_mag)
            intra_window_offsets[i] = peak_idx
        
        # Raw start times are based on fixed PRF sampling (window start)
        raw_start_times_us = pulse_start_times_s * 1e6  # Convert to microseconds
        
        # Real start times = window start + intra-window offset
        intra_window_offsets_us = intra_window_offsets / sample_rate_hz * 1e6
        real_start_times_us = raw_start_times_us + intra_window_offsets_us
        
        # Compute differences (actual position within window)
        time_correction_us = intra_window_offsets_us
        
        # Compute PRIs (time between consecutive pulses)
        if num_pulses > 1:
            pris_us = np.diff(real_start_times_us)
            workflow.add_text(f"**PRI Range:** {pris_us.min():.2f} - {pris_us.max():.2f} μs  |  **PRF Range:** {1e6/pris_us.max():.1f} - {1e6/pris_us.min():.1f} Hz")
        
        # Plot raw vs real start times
        fig3b = go.Figure()
        fig3b.add_trace(go.Scatter(
            x=np.arange(num_pulses),
            y=raw_start_times_us,
            mode='markers',
            name='Raw Start Times (Window Start)',
            marker=dict(color='red', size=4, opacity=0.6)
        ))
        fig3b.add_trace(go.Scatter(
            x=np.arange(num_pulses),
            y=real_start_times_us,
            mode='markers',
            name='Real Start Times (Peak Position)',
            marker=dict(color='cyan', size=4)
        ))
        fig3b.update_layout(
            title="Pulse Start Times: Window Start vs. Actual Peak Position",
            xaxis_title="Pulse Number",
            yaxis_title="Start Time (μs)",
            template='plotly_dark',
            height=600,
            showlegend=True
        )
        workflow.add_plot(fig3b)
        
        # Plot intra-window offsets
        fig3c = go.Figure()
        fig3c.add_trace(go.Scatter(
            x=np.arange(num_pulses),
            y=intra_window_offsets_us,
            mode='markers',
            name='Intra-Window Offset',
            marker=dict(color='magenta', size=4)
        ))
        fig3c.update_layout(
            title="Pulse Peak Position Within Each Window",
            xaxis_title="Pulse Number",
            yaxis_title="Offset from Window Start (μs)",
            template='plotly_dark',
            height=600
        )
        workflow.add_plot(fig3c)
        
        # Plot PRIs
        if num_pulses > 1:
            fig3d = go.Figure()
            fig3d.add_trace(go.Scatter(
                x=np.arange(num_pulses - 1),
                y=pris_us,
                mode='markers+lines',
                name='PRI',
                marker=dict(color='cyan', size=4),
                line=dict(color='cyan', width=1)
            ))
            # Add reference lines for min/max PRF
            min_pri_us = 1e6 / max_prf_hz
            max_pri_us = 1e6 / min_prf_hz
            fig3d.add_trace(go.Scatter(
                x=[0, num_pulses-2],
                y=[min_pri_us, min_pri_us],
                mode='lines',
                name=f'Min PRI ({min_pri_us:.1f} μs, {max_prf_hz} Hz)',
                line=dict(color='yellow', width=1, dash='dash')
            ))
            fig3d.add_trace(go.Scatter(
                x=[0, num_pulses-2],
                y=[max_pri_us, max_pri_us],
                mode='lines',
                name=f'Max PRI ({max_pri_us:.1f} μs, {min_prf_hz} Hz)',
                line=dict(color='orange', width=1, dash='dash')
            ))
            fig3d.update_layout(
                title="Pulse Repetition Intervals (PRI) Between Consecutive Pulses",
                xaxis_title="Pulse Pair Number",
                yaxis_title="PRI (μs)",
                template='plotly_dark',
                height=600,
                showlegend=True
            )
            workflow.add_plot(fig3d)
        
        workflow.add_text("\n## PRI-Based Pulse Re-Extraction")
        
        # Use the estimated PRIs to re-extract pulses from the MF output at correct positions
        # Start with the first detected pulse position
        first_pulse_sample = int(pulse_window_indices[0] * shortest_pri_samples + intra_window_offsets[0])
        
        # Build pulse extraction positions based on PRIs
        pulse_positions_samples = [first_pulse_sample]
        for i in range(num_pulses - 1):
            pri_samples = int(pris_us[i] * sample_rate_hz / 1e6)
            next_position = pulse_positions_samples[-1] + pri_samples
            pulse_positions_samples.append(next_position)
        
        pulse_positions_samples = np.array(pulse_positions_samples)
        
        # Define extraction window size (use longest PRI as window size)
        extraction_window_samples = int(np.max(pris_us) * sample_rate_hz / 1e6) if num_pulses > 1 else shortest_pri_samples
        
        # Extract pulses from MF output at the computed positions
        pulses_extracted = np.zeros((num_pulses, extraction_window_samples), dtype=complex)
        for i in range(num_pulses):
            start_idx = pulse_positions_samples[i]
            end_idx = start_idx + extraction_window_samples
            
            # Handle edge cases
            if end_idx <= len(mf_output):
                pulses_extracted[i, :] = mf_output[start_idx:end_idx]
            else:
                # Pulse extends beyond data, pad with zeros
                available = len(mf_output) - start_idx
                if available > 0:
                    pulses_extracted[i, :available] = mf_output[start_idx:]
        
        # Create heatmap of PRI-extracted pulses
        pulses_extracted_db = 10 * np.log10(np.abs(pulses_extracted) ** 2 + 1e-12)
        
        # Downsample for rendering
        heatmap_extracted, skip_x_e, skip_y_e = downsample_heatmap(pulses_extracted_db, max_width=2000, max_height=1000)
        
        # Time axes for extracted pulses heatmap
        fast_time_extracted_us = np.arange(heatmap_extracted.shape[1]) * skip_x_e / sample_rate_hz * 1e6
        pulse_indices_extracted = np.arange(heatmap_extracted.shape[0]) * skip_y_e
        
        fig4 = go.Figure(data=go.Heatmap(
            z=heatmap_extracted,
            x=fast_time_extracted_us,
            y=pulse_indices_extracted,
            colorscale='HSV',
            colorbar=dict(title='Power (dB)'),
            zmin=heatmap_extracted.min(),
            zmax=heatmap_extracted.max(),
        ))
        fig4.update_layout(
            title=f"PRI-Based Extracted Pulses (n={num_pulses}, Pre-Aligned)",
            xaxis_title="Fast Time (μs)",
            yaxis_title="Pulse Number",
            template='plotly_dark',
            height=700,
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(label="Auto", method="relayout", args=[{"coloraxis.cmin": None, "coloraxis.cmax": None}]),
                        dict(label="Full", method="relayout", args=[{"coloraxis.cmin": heatmap_extracted.min(), "coloraxis.cmax": heatmap_extracted.max()}]),
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                ),
            ],
            sliders=[
                dict(
                    active=100,
                    yanchor="top",
                    y=-0.1,
                    xanchor="left",
                    currentvalue=dict(prefix="Color Max: ", visible=True, xanchor="right"),
                    pad=dict(b=10, t=50),
                    len=0.45,
                    x=0.0,
                    steps=[
                        dict(method="relayout", args=[{"coloraxis.cmax": heatmap_extracted.min() + i * (heatmap_extracted.max() - heatmap_extracted.min()) / 100}], label=f"{heatmap_extracted.min() + i * (heatmap_extracted.max() - heatmap_extracted.min()) / 100:.1f}")
                        for i in range(101)
                    ]
                ),
                dict(
                    active=0,
                    yanchor="top",
                    y=-0.1,
                    xanchor="right",
                    currentvalue=dict(prefix="Color Min: ", visible=True, xanchor="left"),
                    pad=dict(b=10, t=50),
                    len=0.45,
                    x=1.0,
                    steps=[
                        dict(method="relayout", args=[{"coloraxis.cmin": heatmap_extracted.min() + i * (heatmap_extracted.max() - heatmap_extracted.min()) / 100}], label=f"{heatmap_extracted.min() + i * (heatmap_extracted.max() - heatmap_extracted.min()) / 100:.1f}")
                        for i in range(101)
                    ]
                ),
            ]
        )
        workflow.add_plot(fig4)

        # Motion compensation - align peaks in range before Doppler compression
        workflow.add_text("\n=== Motion Compensation (Range Cell Migration Correction) ===")
        
        # Find peak in each pulse (search in first 20 μs where targets are)
        search_range = min(2000, extraction_window_samples)  # 20 μs at 100 MHz
        peak_positions = np.zeros(num_pulses, dtype=int)
        peak_values = np.zeros(num_pulses)
        
        for i in range(num_pulses):
            pulse_mag = np.abs(pulses_extracted[i, :search_range])
            peak_idx = np.argmax(pulse_mag)
            peak_positions[i] = peak_idx
            peak_values[i] = pulse_mag[peak_idx]
        
        # Use median peak position as reference
        ref_peak_pos = int(np.median(peak_positions))
        
        workflow.add_text([
            f"Peak positions in pulses:",
            f"  Range: {peak_positions.min()} - {peak_positions.max()} samples",
            f"  Median: {ref_peak_pos} samples ({ref_peak_pos/sample_rate_hz*1e6:.2f} μs)",
            f"  Std dev: {np.std(peak_positions):.2f} samples ({np.std(peak_positions)/sample_rate_hz*1e6:.3f} μs)",
            f"  Migration span: {(peak_positions.max() - peak_positions.min())/sample_rate_hz*1e6:.3f} μs",
        ])
        
        # Align pulses by shifting to common peak position
        pulses_aligned = np.zeros_like(pulses_extracted, dtype=complex)
        for i in range(num_pulses):
            shift = ref_peak_pos - peak_positions[i]
            if shift >= 0:
                # Shift right
                end = min(shift + extraction_window_samples, extraction_window_samples)
                copy_len = end - shift
                pulses_aligned[i, shift:end] = pulses_extracted[i, :copy_len]
            else:
                # Shift left
                start = -shift
                copy_len = min(extraction_window_samples + shift, extraction_window_samples)
                pulses_aligned[i, :copy_len] = pulses_extracted[i, start:start+copy_len]
        
        workflow.add_text("✓ Pulses aligned to median peak position")

        # Doppler compression on motion-compensated pulses using NUFFT
        workflow.add_text("\n=== Doppler Compression (NUFFT for Non-Uniform Timing) ===")
        
        # Use the actual pulse positions (in samples) from PRI-based extraction
        pulse_times_s = pulse_positions_samples / sample_rate_hz
        
        # Make times relative to first pulse
        pulse_times_relative_s = pulse_times_s - pulse_times_s[0]
        t_span = pulse_times_relative_s[-1]
        
        # Apply window function across slow time (Kaiser has better sidelobe suppression)
        slow_time_window = np.kaiser(num_pulses, beta=8.6)  # beta=8.6 gives ~-60dB sidelobes
        
        # Compute Doppler parameters
        num_doppler_bins = num_pulses
        avg_pri_s = np.mean(pris_us) / 1e6 if num_pulses > 1 else 1.0
        doppler_resolution_hz = 1.0 / t_span if t_span > 0 else 1.0
        
        # For non-uniform sampling, the unambiguous Doppler range is approximately
        # determined by the average PRF, but with stagger we get spectral artifacts
        avg_prf = 1.0 / avg_pri_s
        nyquist_doppler = avg_prf / 2.0
        
        # The output frequency grid
        doppler_freqs_hz = np.fft.fftfreq(num_doppler_bins, t_span / num_doppler_bins)
        
        workflow.add_text([
            f"Number of pulses: {num_pulses}",
            f"Observation time span: {t_span*1e6:.2f} μs ({t_span*1e3:.2f} ms)",
            f"Average PRI: {avg_pri_s*1e6:.2f} μs",
            f"Average PRF: {avg_prf:.1f} Hz",
            f"Nyquist Doppler (avg PRF): ±{nyquist_doppler:.1f} Hz",
            f"Doppler resolution: {doppler_resolution_hz:.2f} Hz",
            f"Using pulse positions from PRI-based extraction",
            f"Window: Kaiser (β=8.6, ~-60dB sidelobes)",
        ])
        
        # Use more range bins - up to 10000 or available
        max_range_bins = min(10000, extraction_window_samples)
        pulses_trimmed = pulses_aligned[:, :max_range_bins]
        num_range_bins = pulses_trimmed.shape[1]
        
        workflow.add_text([
            f"Processing {num_range_bins} range bins ({num_range_bins/sample_rate_hz*1e6:.2f} μs)",
        ])
        
        # Apply NUFFT using direct computation (more reliable)
        range_doppler = np.zeros((num_doppler_bins, num_range_bins), dtype=complex)
        
        # Compute NUFFT: F[k] = sum_j c_j * exp(-2πi * f_k * t_j)
        for range_bin in range(num_range_bins):
            # Get signal across all pulses for this range bin
            signal = pulses_trimmed[:, range_bin] * slow_time_window
            
            # Direct NUFFT computation: F[k] = sum_j signal[j] * exp(-2πi * freq[k] * time[j])
            for k, freq in enumerate(doppler_freqs_hz):
                range_doppler[k, range_bin] = np.sum(signal * np.exp(-2j * np.pi * freq * pulse_times_relative_s))
        
        # Shift zero frequency to center
        range_doppler = np.fft.fftshift(range_doppler, axes=0)
        doppler_freqs_hz = np.fft.fftshift(doppler_freqs_hz)
        
        # Convert to dB
        range_doppler_db = 10 * np.log10(np.abs(range_doppler) ** 2 + 1e-12)
        
        workflow.add_text(f"\n## Range-Doppler Map (NUFFT)")
        workflow.add_text(f"**Doppler Resolution:** {doppler_resolution_hz:.2f} Hz  |  **Doppler Range:** {doppler_freqs_hz[0]:.1f} - {doppler_freqs_hz[-1]:.1f} Hz")
        
        # Downsample for rendering
        heatmap_rd, skip_x_rd, skip_y_rd = downsample_heatmap(range_doppler_db, max_width=2000, max_height=1000)
        
        # Time and frequency axes
        fast_time_rd_us = np.arange(heatmap_rd.shape[1]) * skip_x_rd / sample_rate_hz * 1e6
        doppler_freqs_display = doppler_freqs_hz[::skip_y_rd] if skip_y_rd < len(doppler_freqs_hz) else doppler_freqs_hz
        
        fig5 = go.Figure(data=go.Heatmap(
            z=heatmap_rd,
            x=fast_time_rd_us,
            y=doppler_freqs_display[:heatmap_rd.shape[0]],
            colorscale='Jet',
            colorbar=dict(title='Power (dB)'),
        ))
        
        # Create slider steps for zmin and zmax
        zmin_val = heatmap_rd.min()
        zmax_val = heatmap_rd.max()
        z_range = zmax_val - zmin_val
        
        fig5.update_layout(
            title=f"Range-Doppler Map (NUFFT, {num_pulses} pulses, Non-Uniform Timing)",
            xaxis_title='Range (μs)',
            yaxis_title='Doppler (Hz)',
            template='plotly_dark',
            height=800,
            sliders=[
                dict(
                    active=100,
                    yanchor="top",
                    y=-0.05,
                    xanchor="left",
                    currentvalue=dict(prefix="Color Max: ", visible=True, xanchor="right"),
                    pad=dict(b=10, t=50),
                    len=0.45,
                    x=0.0,
                    steps=[
                        dict(
                            method="restyle",
                            args=[{"zmax": zmin_val + i * z_range / 100}],
                            label=f"{zmin_val + i * z_range / 100:.1f}"
                        )
                        for i in range(101)
                    ]
                ),
                dict(
                    active=0,
                    yanchor="top",
                    y=-0.05,
                    xanchor="right",
                    currentvalue=dict(prefix="Color Min: ", visible=True, xanchor="left"),
                    pad=dict(b=10, t=50),
                    len=0.45,
                    x=1.0,
                    steps=[
                        dict(
                            method="restyle",
                            args=[{"zmin": zmin_val + i * z_range / 100}],
                            label=f"{zmin_val + i * z_range / 100:.1f}"
                        )
                        for i in range(101)
                    ]
                ),
            ]
        )
        workflow.add_plot(fig5)

        return workflow.build()

        # Detect pulse TOAs
        workflow.add_text("\n=== Staggered PRF Analysis ===")
        windows_2d, window_length, valid_toas = extract_toa_windowed_2d(mf_output, toa_indices, min_prf_hz, max_prf_hz, sample_rate_hz)
        workflow.add_text([
            f"Min PRF: {min_prf_hz} Hz (period: {1000/min_prf_hz:.3f} ms)",
            f"Max PRF: {max_prf_hz} Hz (period: {1000/max_prf_hz:.3f} ms)",
            f"Window length: {window_length} samples ({window_length/sample_rate_hz*1000:.3f} ms)",
            f"Number of pulses: {windows_2d.shape[0]}",
        ])

        # Create 2D heatmap
        windows_2d_db = 10 * np.log10(np.abs(windows_2d) ** 2 + 1e-12)
        
        # Downsample for rendering
        heatmap_data, skip_x, skip_y = downsample_heatmap(windows_2d_db, max_width=2000, max_height=1000)
        workflow.add_text(f"Heatmap shape: {windows_2d_db.shape} → {heatmap_data.shape} (downsampled {skip_x}x{skip_y})")
        
        # Time axes for heatmap (adjusted for downsampling)
        fast_time_us = np.arange(heatmap_data.shape[1]) * skip_x / sample_rate_hz * 1e6  # microseconds
        pulse_numbers = np.arange(heatmap_data.shape[0]) * skip_y  # pulse index
        
        fig2 = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=fast_time_us,
            y=pulse_numbers,
            colorscale='HSV',
            colorbar=dict(title='Power (dB)'),
        ))
        fig2.update_layout(
            title=f"Staggered PRF Heatmap ({min_prf_hz}-{max_prf_hz} Hz)",
            xaxis_title="Fast Time (μs)",
            yaxis_title="Pulse Number",
            template='plotly_dark',
            height=700
        )
        workflow.add_plot(fig2)

        # Compute Pulse Pair Correlation (PPC)
        workflow.add_text("\n=== Pulse Pair Correlation ===")
        correlations, peak_positions, peak_values = compute_ppc_correlations(windows_2d)
        
        workflow.add_text([
            f"Number of correlation pairs: {correlations.shape[0]}",
            f"Correlation length: {correlations.shape[1]} samples",
        ])
        
        # Create heatmap of correlations
        correlations_db = 10 * np.log10(correlations + 1e-12)
        
        # Downsample correlations for rendering
        corr_downsampled, skip_x_corr, skip_y_corr = downsample_heatmap(correlations_db, max_width=2000, max_height=1000)
        workflow.add_text(f"Correlation heatmap: {correlations_db.shape} → {corr_downsampled.shape} (downsampled {skip_x_corr}x{skip_y_corr})")
        
        # Time axes for correlation heatmap
        corr_lag_us = np.arange(corr_downsampled.shape[1]) * skip_x_corr / sample_rate_hz * 1e6
        corr_pulse_pairs = np.arange(corr_downsampled.shape[0]) * skip_y_corr
        
        fig3 = go.Figure(data=go.Heatmap(
            z=corr_downsampled,
            x=corr_lag_us,
            y=corr_pulse_pairs,
            colorscale='HSV',
            colorbar=dict(title='Correlation (dB)'),
        ))
        fig3.update_layout(
            title="PPC Correlation Heatmap",
            xaxis_title="Lag (μs)",
            yaxis_title="Pulse Pair Number",
            template='plotly_dark',
            height=700
        )
        workflow.add_plot(fig3)

        # Compute actual time between consecutive pulses (measured PRIs)
        measured_pri_samples = np.diff(valid_toas)
        measured_pri_us = measured_pri_samples / sample_rate_hz * 1e6
        pulse_pair_numbers = np.arange(len(measured_pri_us))
        
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=pulse_pair_numbers,
            y=measured_pri_us,
            mode='markers',
            name='Measured PRI',
            marker=dict(
                color='magenta',
                size=3,
            )
        ))
        fig4.update_layout(
            title="Measured Pulse Repetition Intervals (PRI)",
            xaxis_title="Pulse Pair Number",
            yaxis_title="PRI (μs)",
            template='plotly_dark',
            height=600
        )
        workflow.add_plot(fig4)

        return workflow.build()

    except Exception as e:
        import traceback
        traceback.print_exc()
        workflow.add_text(f"Error: {str(e)}")
        return workflow.build()