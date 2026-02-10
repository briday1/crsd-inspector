"""
Pulse Extraction Workflow (Dagex version)

Modular matched filter processing for continuous staggered-PRF radar data.
"""

import numpy as np
import plotly.graph_objects as go
from crsd_inspector.workflows.workflow import Workflow
from scipy.signal import correlate
from sklearn.cluster import KMeans
from dagex import Graph


# Create workflow instance
workflow = Workflow(
    name="Pulse Extraction",
    description="Matched filter processing for continuous radar data with pulse extraction"
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
    """Create window function"""
    if (window_type is None) or (window_type == 'none'):
        return np.ones(n)
    if window_type == 'hamming':
        return np.hamming(n)
    if window_type == 'hanning':
        return np.hanning(n)
    if window_type == 'blackman':
        return np.blackman(n)
    return np.ones(n)


def downsample_heatmap(data_2d, max_width=2000, max_height=1000):
    """Downsample 2D array for efficient rendering"""
    rows, cols = data_2d.shape
    skip_x = max(1, cols // max_width)
    skip_y = max(1, rows // max_height)
    downsampled = data_2d[::skip_y, ::skip_x]
    return downsampled, skip_x, skip_y


# ============================================================================
# DAGEX NODE FUNCTIONS
# ============================================================================

def node_matched_filter(inputs):
    """Apply matched filter to continuous signal"""
    signal_data = inputs['signal_data']
    tx_wfm = inputs['tx_wfm']
    window_type = inputs['window_type']
    
    if signal_data.ndim > 1:
        signal_data = signal_data.ravel()
    
    ref_wfm = tx_wfm.copy()
    w = _make_window(len(ref_wfm), window_type)
    ref_wfm = ref_wfm * w
    
    mf_output = correlate(signal_data, np.conj(ref_wfm), mode='same', method='fft')
    mf_output_db = 10 * np.log10(np.abs(mf_output) ** 2 + 1e-12)
    
    return {
        'mf_output': mf_output,
        'mf_output_db': mf_output_db
    }


def node_fixed_prf_windows(inputs):
    """Reshape MF output into fixed-PRF windows"""
    mf_output = inputs['mf_output']
    shortest_pri_samples = inputs['shortest_pri_samples']
    
    num_windows = (len(mf_output) - shortest_pri_samples) // shortest_pri_samples
    windows_2d = np.zeros((num_windows, shortest_pri_samples), dtype=mf_output.dtype)
    
    for i in range(num_windows):
        start_idx = i * shortest_pri_samples
        end_idx = start_idx + shortest_pri_samples
        windows_2d[i, :] = mf_output[start_idx:end_idx]
    
    windows_2d_db = 10 * np.log10(np.abs(windows_2d) ** 2 + 1e-12)
    
    return {
        'windows_2d': windows_2d,
        'windows_2d_db': windows_2d_db,
        'num_windows': num_windows
    }


def node_pulse_detection(inputs):
    """Detect pulses using K-means clustering"""
    windows_2d = inputs['windows_2d']
    shortest_pri_samples = inputs['shortest_pri_samples']
    sample_rate_hz = inputs['sample_rate_hz']
    
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
    num_windows = len(windows_2d)
    
    # Start time of each window in seconds
    window_start_times_s = np.arange(num_windows) * shortest_pri_samples / sample_rate_hz
    pulse_start_times_s = window_start_times_s[pulse_window_indices]
    
    # Compute cluster boundary as threshold for display
    power_threshold = (cluster_centers[null_cluster] + cluster_centers[pulse_cluster]) / 2
    
    # Extract only pulses (remove empty windows)
    pulses_2d = windows_2d[pulse_window_indices, :]
    pulses_2d_db = 10 * np.log10(np.abs(pulses_2d) ** 2 + 1e-12)
    
    return {
        'pulses_2d': pulses_2d,
        'pulses_2d_db': pulses_2d_db,
        'num_pulses': num_pulses,
        'pulse_window_indices': pulse_window_indices,
        'pulse_start_times_s': pulse_start_times_s,
        'window_powers': window_powers,
        'has_pulse': has_pulse,
        'power_threshold': power_threshold,
        'cluster_centers': cluster_centers
    }


def node_pulse_timing(inputs):
    """Analyze pulse timing and compute PRIs"""
    pulses_2d = inputs['pulses_2d']
    pulse_start_times_s = inputs['pulse_start_times_s']
    sample_rate_hz = inputs['sample_rate_hz']
    num_pulses = inputs['num_pulses']
    shortest_pri_samples = inputs['shortest_pri_samples']
    pulse_window_indices = inputs['pulse_window_indices']
    
    # For each pulse, detect where the pulse peak is within its window
    intra_window_offsets = np.zeros(num_pulses, dtype=int)
    for i in range(num_pulses):
        pulse_mag = np.abs(pulses_2d[i, :])
        peak_idx = np.argmax(pulse_mag)
        intra_window_offsets[i] = peak_idx
    
    # Raw start times are based on fixed PRF sampling (window start)
    raw_start_times_us = pulse_start_times_s * 1e6
    
    # Real start times = window start + intra-window offset
    intra_window_offsets_us = intra_window_offsets / sample_rate_hz * 1e6
    real_start_times_us = raw_start_times_us + intra_window_offsets_us
    
    # Compute PRIs (time between consecutive pulses)
    pris_us = np.diff(real_start_times_us) if num_pulses > 1 else np.array([1000.0])
    
    # Build pulse extraction positions based on PRIs
    first_pulse_sample = int(pulse_window_indices[0] * shortest_pri_samples + intra_window_offsets[0])
    pulse_positions_samples = [first_pulse_sample]
    
    for i in range(num_pulses - 1):
        pri_samples = int(pris_us[i] * sample_rate_hz / 1e6)
        next_position = pulse_positions_samples[-1] + pri_samples
        pulse_positions_samples.append(next_position)
    
    pulse_positions_samples = np.array(pulse_positions_samples)
    
    return {
        'real_start_times_us': real_start_times_us,
        'pris_us': pris_us,
        'pulse_positions_samples': pulse_positions_samples,
        'intra_window_offsets': intra_window_offsets,
        'intra_window_offsets_us': intra_window_offsets_us
    }


def node_pri_based_extraction(inputs):
    """Re-extract pulses using estimated PRIs"""
    mf_output = inputs['mf_output']
    pulse_positions_samples = inputs['pulse_positions_samples']
    pris_us = inputs['pris_us']
    sample_rate_hz = inputs['sample_rate_hz']
    num_pulses = inputs['num_pulses']
    
    # Define extraction window size (use longest PRI as window size)
    extraction_window_samples = int(np.max(pris_us) * sample_rate_hz / 1e6) if len(pris_us) > 0 else 1000
    
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
    
    pulses_extracted_db = 10 * np.log10(np.abs(pulses_extracted) ** 2 + 1e-12)
    
    return {
        'pulses_extracted': pulses_extracted,
        'pulses_extracted_db': pulses_extracted_db,
        'extraction_window_samples': extraction_window_samples
    }


def node_motion_compensation(inputs):
    """Align pulses by compensating for range cell migration"""
    pulses_extracted = inputs['pulses_extracted']
    extraction_window_samples = inputs['extraction_window_samples']
    num_pulses = inputs['num_pulses']
    
    # Find peak in each pulse (search in first 20 μs where targets are)
    search_range = min(2000, extraction_window_samples)
    peak_positions = np.zeros(num_pulses, dtype=int)
    
    for i in range(num_pulses):
        pulse_mag = np.abs(pulses_extracted[i, :search_range])
        peak_idx = np.argmax(pulse_mag)
        peak_positions[i] = peak_idx
    
    # Use median peak position as reference
    ref_peak_pos = int(np.median(peak_positions))
    
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
    
    return {
        'pulses_aligned': pulses_aligned,
        'peak_positions': peak_positions,
        'ref_peak_pos': ref_peak_pos
    }


def node_nufft_doppler(inputs):
    """Perform NUFFT Doppler compression on aligned pulses"""
    pulses_aligned = inputs['pulses_aligned']
    pulse_positions_samples = inputs['pulse_positions_samples']
    pris_us = inputs['pris_us']
    sample_rate_hz = inputs['sample_rate_hz']
    num_pulses = inputs['num_pulses']
    extraction_window_samples = inputs['extraction_window_samples']
    
    # Use the actual pulse positions (in samples) from PRI-based extraction
    pulse_times_s = pulse_positions_samples / sample_rate_hz
    
    # Make times relative to first pulse
    pulse_times_relative_s = pulse_times_s - pulse_times_s[0]
    t_span = pulse_times_relative_s[-1]
    
    # Apply window function across slow time
    slow_time_window = np.kaiser(num_pulses, beta=8.6)
    
    # Compute Doppler parameters
    avg_pri_s = np.mean(pris_us) / 1e6 if num_pulses > 1 else 1.0
    avg_prf = 1.0 / avg_pri_s
    nyquist_doppler = avg_prf / 2.0
    
    # For NUFFT, the frequency resolution is 1/t_span
    doppler_resolution_hz = 1.0 / t_span if t_span > 0 else 1.0
    num_doppler_bins = num_pulses
    
    # Create frequency grid centered at zero, spanning +/- Nyquist
    doppler_freqs_hz = np.linspace(-nyquist_doppler, nyquist_doppler, num_doppler_bins)
    
    # Use more range bins - up to 10000 or available
    max_range_bins = min(10000, extraction_window_samples)
    pulses_trimmed = pulses_aligned[:, :max_range_bins]
    num_range_bins = pulses_trimmed.shape[1]
    
    # Apply NUFFT using direct computation
    range_doppler = np.zeros((num_doppler_bins, num_range_bins), dtype=complex)
    
    # Compute NUFFT: F[k] = sum_j c_j * exp(-2πi * f_k * t_j)
    for range_bin in range(num_range_bins):
        # Get signal across all pulses for this range bin
        signal = pulses_trimmed[:, range_bin] * slow_time_window
        
        # Direct NUFFT computation: F[k] = sum_j signal[j] * exp(-2πi * freq[k] * time[j])
        for k, freq in enumerate(doppler_freqs_hz):
            range_doppler[k, range_bin] = np.sum(signal * np.exp(-2j * np.pi * freq * pulse_times_relative_s))
    
    # Convert to dB
    range_doppler_db = 10 * np.log10(np.abs(range_doppler) ** 2 + 1e-12)
    
    return {
        'range_doppler': range_doppler,
        'range_doppler_db': range_doppler_db,
        'doppler_freqs_hz': doppler_freqs_hz,
        'doppler_resolution_hz': doppler_resolution_hz,
        'num_range_bins': num_range_bins
    }


# ============================================================================
# WORKFLOW EXECUTION
# ============================================================================

def run_workflow(signal_data, metadata=None, **kwargs):
    """Execute pulse extraction workflow using dagex Graph"""
    workflow.clear()
    
    if metadata is None:
        metadata = {}
    
    tx_wfm = metadata.get('tx_wfm')
    if tx_wfm is None:
        workflow.add_text("No TX waveform found in CRSD file. Cannot perform matched filtering.")
        return workflow.build()
    
    # Create and execute graph
    graph = _create_graph(signal_data, metadata)
    
    try:
        dag = graph.build()
        result = dag.execute(True, 4)
        
        # Extract context
        if hasattr(result, 'context'):
            context = result.context
        elif hasattr(result, 'results'):
            context = result.results
        else:
            context = result
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        workflow.add_text(f"Error executing graph: {str(e)}")
        return workflow.build()
    
    # Format and return results
    _format_results(context, metadata)
    return workflow.build()


def _create_graph(signal_data, metadata):
    """Create pulse extraction workflow graph"""
    graph = Graph()
    
    # Extract parameters
    sample_rate_hz = float(metadata.get('sample_rate_hz', 100e6))
    window_type = metadata.get('window_type', 'hamming')
    min_prf_hz = float(metadata.get('min_prf_hz', 800))
    max_prf_hz = float(metadata.get('max_prf_hz', 1200))
    tx_wfm = metadata.get('tx_wfm')
    
    if signal_data.ndim == 1:
        signal_data = signal_data[None, :]
    
    total_samples = int(signal_data.shape[1])
    shortest_pri_samples = int(sample_rate_hz / max_prf_hz)
    
    # Provide initial data
    graph.add(
        lambda inputs: {
            'signal_data': signal_data,
            'tx_wfm': tx_wfm,
            'window_type': window_type,
            'sample_rate_hz': sample_rate_hz,
            'min_prf_hz': min_prf_hz,
            'max_prf_hz': max_prf_hz,
            'shortest_pri_samples': shortest_pri_samples,
            'total_samples': total_samples
        },
        label="Provide Data",
        inputs=[],
        outputs=[
            ('signal_data', 'signal_data'),
            ('tx_wfm', 'tx_wfm'),
            ('window_type', 'window_type'),
            ('sample_rate_hz', 'sample_rate_hz'),
            ('min_prf_hz', 'min_prf_hz'),
            ('max_prf_hz', 'max_prf_hz'),
            ('shortest_pri_samples', 'shortest_pri_samples'),
            ('total_samples', 'total_samples')
        ]
    )
    
    # Matched filter
    graph.add(
        node_matched_filter,
        label="Matched Filter",
        inputs=[
            ('signal_data', 'signal_data'),
            ('tx_wfm', 'tx_wfm'),
            ('window_type', 'window_type')
        ],
        outputs=[
            ('mf_output', 'mf_output'),
            ('mf_output_db', 'mf_output_db')
        ]
    )
    
    # Fixed-PRF windowing
    graph.add(
        node_fixed_prf_windows,
        label="Fixed-PRF Windows",
        inputs=[
            ('mf_output', 'mf_output'),
            ('shortest_pri_samples', 'shortest_pri_samples')
        ],
        outputs=[
            ('windows_2d', 'windows_2d'),
            ('windows_2d_db', 'windows_2d_db'),
            ('num_windows', 'num_windows')
        ]
    )
    
    # Pulse detection
    graph.add(
        node_pulse_detection,
        label="Pulse Detection",
        inputs=[
            ('windows_2d', 'windows_2d'),
            ('shortest_pri_samples', 'shortest_pri_samples'),
            ('sample_rate_hz', 'sample_rate_hz')
        ],
        outputs=[
            ('pulses_2d', 'pulses_2d'),
            ('pulses_2d_db', 'pulses_2d_db'),
            ('num_pulses', 'num_pulses'),
            ('pulse_window_indices', 'pulse_window_indices'),
            ('pulse_start_times_s', 'pulse_start_times_s'),
            ('window_powers', 'window_powers'),
            ('has_pulse', 'has_pulse'),
            ('power_threshold', 'power_threshold'),
            ('cluster_centers', 'cluster_centers')
        ]
    )
    
    # Pulse timing analysis
    graph.add(
        node_pulse_timing,
        label="Pulse Timing Analysis",
        inputs=[
            ('pulses_2d', 'pulses_2d'),
            ('pulse_start_times_s', 'pulse_start_times_s'),
            ('sample_rate_hz', 'sample_rate_hz'),
            ('num_pulses', 'num_pulses'),
            ('shortest_pri_samples', 'shortest_pri_samples'),
            ('pulse_window_indices', 'pulse_window_indices')
        ],
        outputs=[
            ('real_start_times_us', 'real_start_times_us'),
            ('pris_us', 'pris_us'),
            ('pulse_positions_samples', 'pulse_positions_samples'),
            ('intra_window_offsets', 'intra_window_offsets'),
            ('intra_window_offsets_us', 'intra_window_offsets_us')
        ]
    )
    
    # PRI-based extraction
    graph.add(
        node_pri_based_extraction,
        label="PRI-Based Extraction",
        inputs=[
            ('mf_output', 'mf_output'),
            ('pulse_positions_samples', 'pulse_positions_samples'),
            ('pris_us', 'pris_us'),
            ('sample_rate_hz', 'sample_rate_hz'),
            ('num_pulses', 'num_pulses')
        ],
        outputs=[
            ('pulses_extracted', 'pulses_extracted'),
            ('pulses_extracted_db', 'pulses_extracted_db'),
            ('extraction_window_samples', 'extraction_window_samples')
        ]
    )
    
    # Motion compensation
    graph.add(
        node_motion_compensation,
        label="Motion Compensation",
        inputs=[
            ('pulses_extracted', 'pulses_extracted'),
            ('extraction_window_samples', 'extraction_window_samples'),
            ('num_pulses', 'num_pulses')
        ],
        outputs=[
            ('pulses_aligned', 'pulses_aligned'),
            ('peak_positions', 'peak_positions'),
            ('ref_peak_pos', 'ref_peak_pos')
        ]
    )
    
    # NUFFT Doppler compression
    graph.add(
        node_nufft_doppler,
        label="NUFFT Doppler",
        inputs=[
            ('pulses_aligned', 'pulses_aligned'),
            ('pulse_positions_samples', 'pulse_positions_samples'),
            ('pris_us', 'pris_us'),
            ('sample_rate_hz', 'sample_rate_hz'),
            ('num_pulses', 'num_pulses'),
            ('extraction_window_samples', 'extraction_window_samples')
        ],
        outputs=[
            ('range_doppler', 'range_doppler'),
            ('range_doppler_db', 'range_doppler_db'),
            ('doppler_freqs_hz', 'doppler_freqs_hz'),
            ('doppler_resolution_hz', 'doppler_resolution_hz'),
            ('num_range_bins', 'num_range_bins')
        ]
    )
    
    return graph


def _format_results(context, metadata):
    """Format and add all results to workflow"""
    # Extract all results from context
    results = {}
    for key, value in context.items():
        if isinstance(key, tuple):
            output_name, variant_idx = key
            results[output_name] = value
        else:
            results[key] = value
    
    # Build summary table
    sample_rate_hz = float(metadata.get('sample_rate_hz', 100e6))
    window_type = metadata.get('window_type', 'hamming')
    min_prf_hz = float(metadata.get('min_prf_hz', 800))
    max_prf_hz = float(metadata.get('max_prf_hz', 1200))
    
    total_samples = results.get('total_samples', 0)
    total_time_ms = total_samples / sample_rate_hz * 1000
    shortest_pri_samples = results.get('shortest_pri_samples', 1000)
    shortest_pri_us = shortest_pri_samples / sample_rate_hz * 1e6
    
    file_header_kvps = metadata.get('file_header_kvps', {})
    
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
    
    workflow.add_table("Processing Parameters", {
        "Parameter": [row[0] for row in summary_rows],
        "Value": [row[1] for row in summary_rows]
    })
    
    # Generate all plots from results
    _add_plots(results, metadata)


def _add_plots(results, metadata):
    """Generate and add all plots to workflow"""
    sample_rate_hz = float(metadata.get('sample_rate_hz', 100e6))
    min_prf_hz = float(metadata.get('min_prf_hz', 800))
    max_prf_hz = float(metadata.get('max_prf_hz', 1200))
    shortest_pri_samples = results.get('shortest_pri_samples', 1000)
    
    # Plot 1: Matched filter output
    mf_output_db = results.get('mf_output_db')
    if mf_output_db is not None:
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
    
    # Plot 2: Fixed-PRF pulse array heatmap
    windows_2d_db = results.get('windows_2d_db')
    num_windows = results.get('num_windows', 0)
    if windows_2d_db is not None:
        heatmap_data, skip_x, skip_y = downsample_heatmap(windows_2d_db, max_width=2000, max_height=1000)
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
            height=700
        )
        workflow.add_plot(fig2)
    
    # Plot 3: Pulse detection (K-means clustering result)
    window_powers = results.get('window_powers')
    has_pulse = results.get('has_pulse')
    power_threshold = results.get('power_threshold')
    if window_powers is not None and has_pulse is not None:
        window_powers_db = 10 * np.log10(window_powers**2 + 1e-12)
        threshold_db = 10 * np.log10(power_threshold**2 + 1e-12)
        
        fig2b = go.Figure()
        
        rejected_indices = np.where(~has_pulse)[0]
        if len(rejected_indices) > 0:
            fig2b.add_trace(go.Scatter(
                x=rejected_indices,
                y=window_powers_db[rejected_indices],
                mode='markers',
                name='Rejected (Empty)',
                marker=dict(color='red', size=4, opacity=0.6)
            ))
        
        pulse_indices = np.where(has_pulse)[0]
        if len(pulse_indices) > 0:
            fig2b.add_trace(go.Scatter(
                x=pulse_indices,
                y=window_powers_db[pulse_indices],
                mode='markers',
                name='Detected (Pulse)',
                marker=dict(color='cyan', size=4)
            ))
        
        fig2b.add_trace(go.Scatter(
            x=[0, len(window_powers)-1],
            y=[threshold_db, threshold_db],
            mode='lines',
            name='Decision Boundary',
            line=dict(color='yellow', width=2, dash='dash')
        ))
        
        fig2b.update_layout(
            title="Pulse Detection via K-Means Clustering",
            xaxis_title="Window Number",
            yaxis_title="Window Power (dB)",
            template='plotly_dark',
            height=600,
            showlegend=True
        )
        workflow.add_plot(fig2b)
    
    # Plot 4: Pulses-only heatmap
    pulses_2d_db = results.get('pulses_2d_db')
    if pulses_2d_db is not None:
        heatmap_pulses, skip_x_p, skip_y_p = downsample_heatmap(pulses_2d_db, max_width=2000, max_height=1000)
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
            title=f"Detected Pulses Only (Empty Windows Removed)",
            xaxis_title="Fast Time (μs)",
            yaxis_title="Pulse Number",
            template='plotly_dark',
            height=700
        )
        workflow.add_plot(fig2c)
    
    # Plot 5: PRI plot
    pris_us = results.get('pris_us')
    num_pulses = results.get('num_pulses', 0)
    if pris_us is not None and len(pris_us) > 0:
        fig3d = go.Figure()
        fig3d.add_trace(go.Scatter(
            x=np.arange(len(pris_us)),
            y=pris_us,
            mode='markers+lines',
            name='PRI',
            marker=dict(color='cyan', size=4),
            line=dict(color='cyan', width=1)
        ))
        
        min_pri_us = 1e6 / max_prf_hz
        max_pri_us = 1e6 / min_prf_hz
        fig3d.add_trace(go.Scatter(
            x=[0, len(pris_us)-1],
            y=[min_pri_us, min_pri_us],
            mode='lines',
            name=f'Min PRI ({min_pri_us:.1f} μs, {max_prf_hz} Hz)',
            line=dict(color='yellow', width=1, dash='dash')
        ))
        fig3d.add_trace(go.Scatter(
            x=[0, len(pris_us)-1],
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
    
    # Plot 6: PRI-extracted pulses heatmap
    pulses_extracted_db = results.get('pulses_extracted_db')
    extraction_window_samples = results.get('extraction_window_samples', 1000)
    if pulses_extracted_db is not None:
        heatmap_extracted, skip_x_e, skip_y_e = downsample_heatmap(pulses_extracted_db, max_width=2000, max_height=1000)
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
            height=700
        )
        workflow.add_plot(fig4)
    
    # Plot 7: Range-Doppler map
    range_doppler_db = results.get('range_doppler_db')
    doppler_freqs_hz = results.get('doppler_freqs_hz')
    if range_doppler_db is not None and doppler_freqs_hz is not None:
        heatmap_rd, skip_x_rd, skip_y_rd = downsample_heatmap(range_doppler_db, max_width=2000, max_height=1000)
        fast_time_rd_us = np.arange(heatmap_rd.shape[1]) * skip_x_rd / sample_rate_hz * 1e6
        doppler_freqs_display = doppler_freqs_hz[::skip_y_rd] if skip_y_rd < len(doppler_freqs_hz) else doppler_freqs_hz
        
        fig5 = go.Figure(data=go.Heatmap(
            z=heatmap_rd,
            x=fast_time_rd_us,
            y=doppler_freqs_display[:heatmap_rd.shape[0]],
            colorscale='Jet',
            colorbar=dict(title='Power (dB)'),
        ))
        
        fig5.update_layout(
            title=f"Range-Doppler Map (NUFFT, {num_pulses} pulses, Non-Uniform Timing)",
            xaxis_title='Range (μs)',
            yaxis_title='Doppler (Hz)',
            template='plotly_dark',
            height=800
        )
        workflow.add_plot(fig5)
