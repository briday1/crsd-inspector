"""
Staggered PRF Range-Doppler Processing Workflow

Performs:
1. Range compression via matched filtering
2. PRF stagger estimation/determination
3. Pulse extraction
4. NUFFT-based Doppler compression for non-uniform pulse timing

This workflow handles signals with unknown staggered PRF patterns.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dagex import Graph
from crsd_inspector.workflows.workflow import Workflow
import traceback


# Create workflow instance
workflow = Workflow(
    name="Staggered PRF Range-Doppler",
    description="Range-Doppler processing with PRF stagger estimation and NUFFT"
)


def estimate_prf_stagger(pulse_times: np.ndarray) -> dict:
    """
    Estimate PRF stagger pattern from pulse timing data
    
    Args:
        pulse_times: Array of pulse transmission/reception times (seconds)
        
    Returns:
        Dictionary with:
            - pri: Pulse repetition intervals (seconds)
            - avg_prf: Average PRF (Hz)
            - prf_std: Standard deviation of PRF (Hz)
            - stagger_detected: Boolean indicating if stagger is detected
            - stagger_pattern: Estimated pattern ('uniform', '2-step', '3-step', 'random')
            - unique_pris: Unique PRI values (seconds)
    """
    # Compute PRIs
    pri = np.diff(pulse_times)
    
    # Basic statistics
    avg_pri = np.mean(pri)
    std_pri = np.std(pri)
    avg_prf = 1.0 / avg_pri
    prf_std = std_pri * avg_prf**2  # Approximate std of PRF from std of PRI
    
    # Detect stagger by checking variance
    cv = std_pri / avg_pri  # Coefficient of variation
    stagger_detected = cv > 0.01  # More than 1% variation
    
    # Estimate pattern
    if not stagger_detected:
        stagger_pattern = 'uniform'
        unique_pris = np.array([avg_pri])
    else:
        # Find unique PRIs (cluster PRIs within 1% tolerance)
        sorted_pri = np.sort(pri)
        unique_pris = []
        tol = 0.01 * avg_pri
        
        current_cluster = [sorted_pri[0]]
        for p in sorted_pri[1:]:
            if p - current_cluster[-1] < tol:
                current_cluster.append(p)
            else:
                unique_pris.append(np.mean(current_cluster))
                current_cluster = [p]
        unique_pris.append(np.mean(current_cluster))
        unique_pris = np.array(unique_pris)
        
        # Classify pattern
        num_unique = len(unique_pris)
        if num_unique == 2:
            stagger_pattern = '2-step'
        elif num_unique == 3:
            stagger_pattern = '3-step'
        else:
            stagger_pattern = 'random'
    
    return {
        'pri': pri,
        'avg_prf': float(avg_prf),
        'prf_std': float(prf_std),
        'stagger_detected': bool(stagger_detected),
        'stagger_pattern': stagger_pattern,
        'unique_pris': unique_pris,
        'min_prf': float(1.0 / np.max(pri)),
        'max_prf': float(1.0 / np.min(pri)),
    }


def nufft_1d(x: np.ndarray, t: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    1D Non-uniform FFT using direct summation (type-1 NUFFT)
    
    Computes X(f_k) = sum_j x_j * exp(-2πi f_k t_j)
    
    Args:
        x: Signal values at non-uniform times (length N)
        t: Non-uniform time samples (length N)
        f: Uniform frequency samples where to evaluate (length M)
        
    Returns:
        X: NUFFT values at frequency samples (length M)
    """
    # Direct summation - not optimized but correct
    # For production, use finufft or cufinufft
    N = len(x)
    M = len(f)
    X = np.zeros(M, dtype=np.complex128)
    
    for k in range(M):
        X[k] = np.sum(x * np.exp(-2j * np.pi * f[k] * t))
    
    return X


def nufft_doppler_compression(range_compressed: np.ndarray, pulse_times: np.ndarray, 
                               doppler_bins: int = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Doppler compression using NUFFT for non-uniform pulse times
    
    Args:
        range_compressed: Range-compressed signal (num_pulses x num_range_bins)
        pulse_times: Non-uniform pulse times (seconds)
        doppler_bins: Number of Doppler frequency bins (default: num_pulses)
        
    Returns:
        range_doppler: Range-Doppler map (doppler_bins x num_range_bins)
        doppler_freqs: Doppler frequency axis (Hz)
    """
    num_pulses, num_range_bins = range_compressed.shape
    
    if doppler_bins is None:
        doppler_bins = num_pulses
    
    # Normalize time to start at 0
    t_norm = pulse_times - pulse_times[0]
    T_total = t_norm[-1]
    
    # Average PRF for frequency axis
    avg_prf = (num_pulses - 1) / T_total
    
    # Create uniform Doppler frequency grid
    # Use avg PRF to set frequency range
    doppler_freqs = np.fft.fftshift(np.fft.fftfreq(doppler_bins, 1.0 / avg_prf))
    
    # Apply NUFFT to each range bin
    range_doppler = np.zeros((doppler_bins, num_range_bins), dtype=np.complex128)
    
    for r in range(num_range_bins):
        signal = range_compressed[:, r]
        range_doppler[:, r] = nufft_1d(signal, t_norm, doppler_freqs)
    
    return range_doppler, doppler_freqs


def apply_range_window_staggered(signal_data, tx_wfm, sample_rate_hz, range_window='none'):
    """
    Apply range windowing and prepare for matched filtering
    
    Args:
        signal_data: Input signal (num_pulses x num_samples)
        tx_wfm: Transmit waveform
        sample_rate_hz: Sample rate
        range_window: Window type
        
    Returns:
        Dictionary with windowed data
    """
    num_pulses, num_samples = signal_data.shape
    
    # Prepare matched filter
    matched_filter_time = np.conj(tx_wfm[::-1])
    matched_filter_freq = np.fft.fft(matched_filter_time, n=num_samples)
    
    # Apply range windowing to matched filter
    if range_window != 'none':
        if range_window == 'hamming':
            window = np.hamming(num_samples)
        elif range_window == 'hann':
            window = np.hanning(num_samples)
        elif range_window == 'blackman':
            window = np.blackman(num_samples)
        elif range_window == 'kaiser':
            window = np.kaiser(num_samples, beta=8.6)
        else:
            window = np.ones(num_samples)
        
        matched_filter_freq = matched_filter_freq * window
    
    return {
        'signal_data': signal_data,
        'matched_filter_freq': matched_filter_freq,
        'num_pulses': num_pulses,
        'num_samples': num_samples,
    }


def range_compression_staggered(windowed_data):
    """
    Perform range compression via frequency-domain matched filtering
    
    Args:
        windowed_data: Output from apply_range_window_staggered
        
    Returns:
        Dictionary with range-compressed signal
    """
    signal_data = windowed_data['signal_data']
    matched_filter_freq = windowed_data['matched_filter_freq']
    
    # FFT along fast-time (range) dimension
    signal_freq = np.fft.fft(signal_data, axis=1)
    
    # Multiply in frequency domain
    compressed_freq = signal_freq * matched_filter_freq[None, :]
    
    # Transform back to time domain
    compressed = np.fft.ifft(compressed_freq, axis=1)
    
    return {
        'range_compressed': compressed,
        'num_pulses': windowed_data['num_pulses'],
        'num_samples': windowed_data['num_samples'],
    }


def estimate_stagger_and_extract_pulses(compressed_data, pulse_times):
    """
    Estimate PRF stagger pattern and extract pulse timing information
    
    Args:
        compressed_data: Range-compressed signal data
        pulse_times: Pulse timing from metadata (TxTime or RcvTime from PVP/PPP)
        
    Returns:
        Dictionary with stagger information and extracted data
    """
    range_compressed = compressed_data['range_compressed']
    
    # Estimate stagger
    stagger_info = estimate_prf_stagger(pulse_times)
    
    return {
        'range_compressed': range_compressed,
        'pulse_times': pulse_times,
        'stagger_info': stagger_info,
    }


def nufft_doppler_processing(pulse_data, doppler_bins=None):
    """
    Perform NUFFT-based Doppler compression
    
    Args:
        pulse_data: Output from estimate_stagger_and_extract_pulses
        doppler_bins: Number of Doppler bins (default: num_pulses)
        
    Returns:
        Dictionary with range-Doppler map and profiles
    """
    range_compressed = pulse_data['range_compressed']
    pulse_times = pulse_data['pulse_times']
    stagger_info = pulse_data['stagger_info']
    
    # Perform NUFFT Doppler compression
    range_doppler, doppler_freqs = nufft_doppler_compression(
        range_compressed, pulse_times, doppler_bins
    )
    
    # Compute magnitude and dB
    rd_mag = np.abs(range_doppler)
    rd_db = 20 * np.log10(rd_mag + 1e-10)
    
    return {
        'range_doppler': range_doppler,
        'rd_mag': rd_mag,
        'rd_db': rd_db,
        'doppler_freqs': doppler_freqs,
        'stagger_info': stagger_info,
    }


def compute_profiles_and_stats_staggered(doppler_data):
    """
    Compute range and Doppler profiles plus statistics
    
    Args:
        doppler_data: Output from nufft_doppler_processing
        
    Returns:
        Individual output fields for graph structure
    """
    rd_mag = doppler_data['rd_mag']
    rd_db = doppler_data['rd_db']
    doppler_freqs = doppler_data['doppler_freqs']
    stagger_info = doppler_data['stagger_info']
    
    # Extract profiles
    range_profile = np.mean(rd_mag, axis=0)
    range_profile_db = 20 * np.log10(range_profile + 1e-10)
    
    doppler_profile = np.mean(rd_mag, axis=1)
    doppler_profile_db = 20 * np.log10(doppler_profile + 1e-10)
    
    # Compute statistics
    peak_val = np.max(rd_db)
    noise_floor = np.percentile(rd_db, 5)
    dynamic_range = peak_val - noise_floor
    
    peak_idx = np.unravel_index(np.argmax(rd_mag), rd_mag.shape)
    peak_doppler_idx, peak_range_idx = peak_idx
    peak_doppler_freq = doppler_freqs[peak_doppler_idx]
    
    stats = {
        'peak_magnitude_db': float(peak_val),
        'noise_floor_db': float(noise_floor),
        'dynamic_range_db': float(dynamic_range),
        'peak_doppler_hz': float(peak_doppler_freq),
        'peak_range_bin': int(peak_range_idx),
        'avg_prf_hz': stagger_info['avg_prf'],
        'prf_std_hz': stagger_info['prf_std'],
        'stagger_pattern': stagger_info['stagger_pattern'],
        'stagger_detected': stagger_info['stagger_detected'],
    }
    
    return {
        'rd_db': rd_db,
        'range_profile_db': range_profile_db,
        'doppler_profile_db': doppler_profile_db,
        'doppler_freqs': doppler_freqs,
        'stats': stats,
        'stagger_info': stagger_info,
    }


def generate_plots_staggered(profile_data):
    """
    Generate range-Doppler map and profile plots
    
    Args:
        profile_data: Output from compute_profiles_and_stats_staggered
        
    Returns:
        Dictionary with Plotly figure JSON
    """
    rd_db = profile_data['rd_db']
    range_profile_db = profile_data['range_profile_db']
    doppler_profile_db = profile_data['doppler_profile_db']
    doppler_freqs = profile_data['doppler_freqs']
    stats = profile_data['stats']
    stagger_info = profile_data['stagger_info']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Range-Doppler Map',
            'Range Profile',
            'Doppler Profile',
            'Stagger Info'
        ),
        specs=[
            [{'type': 'heatmap'}, {'type': 'scatter'}],
            [{'type': 'scatter'}, {'type': 'table'}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )
    
    # Range-Doppler map
    num_doppler, num_range = rd_db.shape
    range_bins = np.arange(num_range)
    
    fig.add_trace(
        go.Heatmap(
            z=rd_db,
            x=range_bins,
            y=doppler_freqs,
            colorscale='Viridis',
            colorbar=dict(title='dB', x=0.46),
            name='R-D Map'
        ),
        row=1, col=1
    )
    
    # Range profile
    fig.add_trace(
        go.Scatter(
            x=range_bins,
            y=range_profile_db,
            mode='lines',
            name='Range',
            line=dict(color='blue')
        ),
        row=1, col=2
    )
    
    # Doppler profile
    fig.add_trace(
        go.Scatter(
            x=doppler_freqs,
            y=doppler_profile_db,
            mode='lines',
            name='Doppler',
            line=dict(color='red')
        ),
        row=2, col=1
    )
    
    # Stagger info table
    stagger_table_data = [
        ['Pattern', stagger_info['stagger_pattern']],
        ['Detected', 'Yes' if stagger_info['stagger_detected'] else 'No'],
        ['Avg PRF', f"{stagger_info['avg_prf']:.2f} Hz"],
        ['PRF Std', f"{stagger_info['prf_std']:.2f} Hz"],
        ['Min PRF', f"{stagger_info['min_prf']:.2f} Hz"],
        ['Max PRF', f"{stagger_info['max_prf']:.2f} Hz"],
        ['Peak (dB)', f"{stats['peak_magnitude_db']:.1f}"],
        ['Dynamic Range', f"{stats['dynamic_range_db']:.1f} dB"],
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Parameter', 'Value'],
                fill_color='lightgray',
                align='left'
            ),
            cells=dict(
                values=list(zip(*stagger_table_data)),
                fill_color='white',
                align='left'
            )
        ),
        row=2, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="Range Bin", row=1, col=1)
    fig.update_yaxes(title_text="Doppler (Hz)", row=1, col=1)
    fig.update_xaxes(title_text="Range Bin", row=1, col=2)
    fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=2)
    fig.update_xaxes(title_text="Doppler (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="Magnitude (dB)", row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title_text="Staggered PRF Range-Doppler Processing Results",
        showlegend=False,
        height=800
    )
    
    return {
        'figure_json': fig.to_json()
    }


def run_workflow(signal_data, metadata=None, **kwargs):
    """
    Execute staggered PRF range-Doppler processing workflow
    
    Args:
        signal_data: Complex signal array (num_pulses x num_samples)
        metadata: Dictionary containing:
            - tx_wfm: Transmit waveform
            - sample_rate_hz: Sample rate in Hz
            - pulse_times: Pulse timing array (from TxTime or RcvTime)
            - range_window: Range window type (optional, default 'hamming')
            - doppler_bins: Number of Doppler bins (optional, default num_pulses)
    """
    workflow.clear()  # Clear any previous results
    
    if metadata is None:
        metadata = {}
    
    try:
        # Validate inputs
        if signal_data is None:
            workflow.add_text("Error: No signal data provided")
            return workflow.build()
        
        tx_wfm = metadata.get('tx_wfm')
        if tx_wfm is None:
            workflow.add_text("Error: No TX waveform found in CRSD file. Cannot perform matched filtering.")
            return workflow.build()
        
        pulse_times = metadata.get('pulse_times')
        if pulse_times is None:
            workflow.add_text("Warning: No pulse timing data found. Assuming uniform PRF.")
            prf_hz = metadata.get('prf_hz', 1000.0)
            num_pulses = signal_data.shape[0]
            pulse_times = np.arange(num_pulses) / prf_hz
        
        # Extract parameters
        sample_rate_hz = metadata.get('sample_rate_hz', 100e6)
        range_window = metadata.get('range_window', 'hamming')
        doppler_bins = metadata.get('doppler_bins', None)
        
        # Execute processing steps directly (simpler than building complex graph)
        try:
            # Step 1: Range windowing and compression
            windowed = apply_range_window_staggered(signal_data, tx_wfm, sample_rate_hz, range_window)
            compressed = range_compression_staggered(windowed)
            
            # Step 2: Estimate stagger and extract pulses
            extracted = estimate_stagger_and_extract_pulses(compressed, pulse_times)
            
            # Step 3: NUFFT Doppler compression
            doppler = nufft_doppler_processing(extracted, doppler_bins)
            
            # Step 4: Compute profiles and statistics
            profile_data = compute_profiles_and_stats_staggered(doppler)
            
            # Step 5: Generate plots
            plot_result = generate_plots_staggered(profile_data)
            profile_data['figure_json'] = plot_result['figure_json']
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            workflow.add_text(f"Error during processing: {str(e)}")
            return workflow.build()
        
        # Format results for display
        workflow.add_text(f"Staggered PRF Processing Complete")
        
        # Add stagger info
        stagger_info = profile_data['stagger_info']
        workflow.add_text([
            f"PRF Pattern: {stagger_info['stagger_pattern']}",
            f"Stagger Detected: {'Yes' if stagger_info['stagger_detected'] else 'No'}",
            f"Average PRF: {stagger_info['avg_prf']:.2f} Hz",
            f"PRF Range: {stagger_info['min_prf']:.2f} - {stagger_info['max_prf']:.2f} Hz",
            f"PRF Std Dev: {stagger_info['prf_std']:.2f} Hz",
        ])
        
        # Add statistics table
        stats = profile_data['stats']
        stats_table = {
            'Metric': [
                'Peak Magnitude (dB)',
                'Noise Floor (dB)',
                'Dynamic Range (dB)',
                'Peak Doppler (Hz)',
                'Peak Range Bin',
            ],
            'Value': [
                f"{stats['peak_magnitude_db']:.2f}",
                f"{stats['noise_floor_db']:.2f}",
                f"{stats['dynamic_range_db']:.2f}",
                f"{stats['peak_doppler_hz']:.2f}",
                f"{stats['peak_range_bin']}",
            ]
        }
        workflow.add_table("Processing Statistics", stats_table)
        
        # Add plot
        import json
        import plotly.graph_objects as go
        fig = go.Figure(json.loads(profile_data['figure_json']))
        workflow.add_plot(fig)
        
        return workflow.build()
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        workflow.add_text(f"Error: {str(e)}")
        return workflow.build()
