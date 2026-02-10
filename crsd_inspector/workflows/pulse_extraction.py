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
        'label': 'Slow-Time Window',
        'type': 'dropdown',
        'default': 'hamming',
        'options': [
            {'label': 'None', 'value': 'none'},
            {'label': 'Hamming', 'value': 'hamming'},
            {'label': 'Hanning', 'value': 'hanning'},
            {'label': 'Blackman', 'value': 'blackman'},
        ]
    },
    'range_window_type': {
        'label': 'Fast-Time (Range) Window',
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
        'default': 2400,
    },
    'auto_detect_prf': {
        'label': 'Auto-detect PRFs (cluster & snap)',
        'type': 'checkbox',
        'default': False,
    },
    'use_fixed_prfs': {
        'label': 'Use Fixed PRFs',
        'type': 'checkbox',
        'default': False,
    },
    'fixed_prfs_hz': {
        'label': 'Fixed PRFs (Hz, comma-separated)',
        'type': 'text',
        'default': '1000,1200,1500',
    },
    'nufft_kernel': {
        'label': 'NUFFT Kernel',
        'type': 'dropdown',
        'default': 'direct',
        'options': [
            {'label': 'Direct Computation (Exact)', 'value': 'direct'},
            {'label': 'Kaiser-Bessel (Approx, Fast)', 'value': 'kaiser_bessel'},
            {'label': 'Gaussian (Approx, Fast)', 'value': 'gaussian'},
        ]
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
    range_window_type = inputs['range_window_type']
    
    if signal_data.ndim > 1:
        signal_data = signal_data.ravel()
    
    ref_wfm = tx_wfm.copy()
    w = _make_window(len(ref_wfm), range_window_type)
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
    
    # For each pulse, detect where the pulse starts within its window using leading edge detection
    intra_window_offsets = np.zeros(num_pulses, dtype=int)
    for i in range(num_pulses):
        pulse_mag = np.abs(pulses_2d[i, :])
        
        # Find the peak first
        peak_idx = np.argmax(pulse_mag)
        peak_val = pulse_mag[peak_idx]
        
        # Use a threshold at 20% of peak to find leading edge
        threshold = 0.2 * peak_val
        
        # Search backwards from peak to find first sample above threshold
        leading_edge_idx = peak_idx
        for j in range(peak_idx, -1, -1):
            if pulse_mag[j] < threshold:
                leading_edge_idx = j + 1  # First sample above threshold
                break
        
        intra_window_offsets[i] = leading_edge_idx
    
    # Raw start times are based on fixed PRF sampling (window start)
    raw_start_times_us = pulse_start_times_s * 1e6
    
    # Real start times = window start + intra-window offset
    intra_window_offsets_us = intra_window_offsets / sample_rate_hz * 1e6
    real_start_times_us = raw_start_times_us + intra_window_offsets_us
    
    # Compute PRIs (time between consecutive pulses)
    pris_us = np.diff(real_start_times_us) if num_pulses > 1 else np.array([1000.0])
    
    # Build pulse extraction positions directly from window indices + intra-window offsets
    # Don't accumulate PRIs as that causes drift from rounding errors
    pulse_positions_samples = pulse_window_indices * shortest_pri_samples + intra_window_offsets
    pulse_positions_samples = pulse_positions_samples.astype(int)
    
    return {
        'real_start_times_us': real_start_times_us,
        'pris_us': pris_us,
        'pulse_positions_samples': pulse_positions_samples,
        'intra_window_offsets': intra_window_offsets,
        'intra_window_offsets_us': intra_window_offsets_us
    }


def node_detect_prfs(inputs):
    """Auto-detect discrete PRF values from PRI distribution using clustering"""
    pris_us = inputs['pris_us']
    auto_detect = inputs['auto_detect_prf']
    use_fixed = inputs.get('use_fixed_prfs', False)
    fixed_prfs_str = inputs.get('fixed_prfs_hz', '1000,1200,1500')
    min_prf_hz_input = inputs['min_prf_hz']
    max_prf_hz_input = inputs['max_prf_hz']
    
    # Handle fixed PRFs mode
    if use_fixed:
        try:
            # Parse CSV of fixed PRF values
            fixed_prfs_hz = np.array([float(x.strip()) for x in fixed_prfs_str.split(',') if x.strip()])
            fixed_prfs_hz = np.sort(fixed_prfs_hz)
            
            # Convert PRIs to PRFs
            prfs_hz = 1e6 / pris_us
            
            # Snap each PRF to the nearest fixed PRF
            snapped_labels = np.zeros(len(prfs_hz), dtype=int)
            for i, prf in enumerate(prfs_hz):
                distances = np.abs(fixed_prfs_hz - prf)
                snapped_labels[i] = np.argmin(distances)
            
            return {
                'detected_min_prf_hz': float(fixed_prfs_hz.min()),
                'detected_max_prf_hz': float(fixed_prfs_hz.max()),
                'detected_prfs_hz': fixed_prfs_hz,
                'pri_clusters': snapped_labels
            }
        except Exception as e:
            # Fall back to input values if parsing fails
            return {
                'detected_min_prf_hz': min_prf_hz_input,
                'detected_max_prf_hz': max_prf_hz_input,
                'detected_prfs_hz': None,
                'pri_clusters': None
            }
    
    if not auto_detect or len(pris_us) < 5:
        # Use input values if auto-detect is disabled or insufficient data
        return {
            'detected_min_prf_hz': min_prf_hz_input,
            'detected_max_prf_hz': max_prf_hz_input,
            'detected_prfs_hz': None,
            'pri_clusters': None
        }
    
    # Convert PRIs to PRFs
    prfs_hz = 1e6 / pris_us
    
    # Determine number of clusters (2-5 depending on data spread)
    prf_range = prfs_hz.max() - prfs_hz.min()
    prf_std = np.std(prfs_hz)
    
    # Use 2 clusters if data is relatively tight, up to 5 if spread out
    if prf_std < 50:
        n_clusters = 2
    elif prf_std < 150:
        n_clusters = 3
    elif prf_std < 300:
        n_clusters = 4
    else:
        n_clusters = 5
    
    n_clusters = min(n_clusters, len(prfs_hz))
    
    # Cluster PRFs using K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(prfs_hz.reshape(-1, 1))
    cluster_centers = kmeans.cluster_centers_.flatten()
    cluster_centers = np.sort(cluster_centers)
    
    # Snap to detected PRF values
    detected_min_prf_hz = float(cluster_centers.min())
    detected_max_prf_hz = float(cluster_centers.max())
    
    return {
        'detected_min_prf_hz': detected_min_prf_hz,
        'detected_max_prf_hz': detected_max_prf_hz,
        'detected_prfs_hz': cluster_centers,
        'pri_clusters': cluster_labels
    }


def node_pri_based_extraction(inputs):
    """Re-extract pulses using estimated PRIs"""
    mf_output = inputs['mf_output']
    pulse_positions_samples = inputs['pulse_positions_samples']
    pris_us = inputs['pris_us']
    sample_rate_hz = inputs['sample_rate_hz']
    num_pulses = inputs['num_pulses']
    shortest_pri_samples = inputs['shortest_pri_samples']
    
    # After alignment, extract only what's needed for the pulse itself
    # Use ~20% of shortest PRI (much smaller than the full PRI period)
    extraction_window_samples = max(int(shortest_pri_samples * 0.2), 100)
    
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


def node_nufft_doppler(inputs):
    """Perform NUFFT Doppler compression on extracted pulses (or FFT if uniform PRF)"""
    pulses_extracted = inputs['pulses_extracted']
    pulse_positions_samples = inputs['pulse_positions_samples']
    pris_us = inputs['pris_us']
    sample_rate_hz = inputs['sample_rate_hz']
    num_pulses = inputs['num_pulses']
    extraction_window_samples = inputs['extraction_window_samples']
    window_type = inputs['window_type']
    nufft_kernel = inputs.get('nufft_kernel', 'direct')
    
    # Use the actual pulse positions (in samples) from PRI-based extraction
    pulse_times_s = pulse_positions_samples / sample_rate_hz
    
    # Make times relative to first pulse
    pulse_times_relative_s = pulse_times_s - pulse_times_s[0]
    t_span = pulse_times_relative_s[-1]
    
    # Apply window function across slow time (Doppler dimension)
    slow_time_window = _make_window(num_pulses, window_type)
    
    # Compute Doppler parameters
    # For staggered PRF, use the minimum PRF (longest PRI) to get unambiguous Nyquist
    max_pri_s = np.max(pris_us) / 1e6 if num_pulses > 1 else 1.0
    min_prf = 1.0 / max_pri_s
    nyquist_doppler = min_prf / 2.0
    
    # For NUFFT, the frequency resolution is 1/t_span
    doppler_resolution_hz = 1.0 / t_span if t_span > 0 else 1.0
    num_doppler_bins = num_pulses
    
    # Create frequency grid centered at zero, spanning +/- Nyquist
    doppler_freqs_hz = np.linspace(-nyquist_doppler, nyquist_doppler, num_doppler_bins)
    
    # Use more range bins - up to 10000 or available
    max_range_bins = min(10000, extraction_window_samples)
    pulses_trimmed = pulses_extracted[:, :max_range_bins]
    num_range_bins = pulses_trimmed.shape[1]
    
    # Check if PRF is uniform (single PRF) - if so, use FFT instead of NUFFT
    pri_variation = np.std(pris_us) / np.mean(pris_us) if len(pris_us) > 0 else 0
    is_uniform_prf = pri_variation < 0.01  # Less than 1% variation
    
    if is_uniform_prf and num_pulses > 1:
        # Uniform PRF detected - use regular FFT (much faster)
        avg_prf = 1e6 / np.mean(pris_us)
        nyquist_doppler = avg_prf / 2.0
        
        # Apply windowing and compute FFT for each range bin
        range_doppler = np.zeros((num_doppler_bins, num_range_bins), dtype=complex)
        
        for range_bin in range(num_range_bins):
            # Get signal across all pulses for this range bin
            signal = pulses_trimmed[:, range_bin] * slow_time_window
            
            # Apply FFT and fftshift to center zero frequency
            fft_result = np.fft.fft(signal)
            range_doppler[:, range_bin] = np.fft.fftshift(fft_result)
        
        # Update frequency grid for uniform sampling
        doppler_freqs_hz = np.fft.fftshift(np.fft.fftfreq(num_pulses, 1.0 / avg_prf))
        doppler_resolution_hz = avg_prf / num_pulses
        
    else:
        # Non-uniform PRF - use NUFFT with selected kernel
        range_doppler = np.zeros((num_doppler_bins, num_range_bins), dtype=complex)
        
        if nufft_kernel == 'direct':
            # Direct NUFFT computation: F[k] = sum_j c_j * exp(-2πi * f_k * t_j)
            # Normalize by sqrt(N) for proper energy scaling
            normalization = 1.0 / np.sqrt(num_pulses)
            for range_bin in range(num_range_bins):
                # Get signal across all pulses for this range bin (apply slow-time window here)
                signal = pulses_trimmed[:, range_bin] * slow_time_window
                
                # Direct computation: F[k] = sum_j signal[j] * exp(-2πi * freq[k]* time[j])
                for k, freq in enumerate(doppler_freqs_hz):
                    range_doppler[k, range_bin] = normalization * np.sum(signal * np.exp(-2j * np.pi * freq * pulse_times_relative_s))
        
        elif nufft_kernel == 'kaiser_bessel':
            # Kaiser-Bessel gridding approximation (faster but approximate)
            # Oversample factor for better accuracy
            oversample = 2
            grid_size = num_doppler_bins * oversample
            
            # Create uniform grid
            grid_times = np.linspace(0, t_span, grid_size)
            dt_grid = t_span / (grid_size - 1) if grid_size > 1 else 1.0
            
            # Kaiser-Bessel parameters
            beta = 8.0  # Shape parameter
            width = 4  # Kernel width in grid points
            
            for range_bin in range(num_range_bins):
                signal = pulses_trimmed[:, range_bin] * slow_time_window
                
                # Grid the non-uniform data using Kaiser-Bessel kernel
                grid_data = np.zeros(grid_size, dtype=complex)
                for j, t_j in enumerate(pulse_times_relative_s):
                    # Find nearest grid point
                    grid_idx = int(t_j / dt_grid)
                    
                    # Spread onto nearby grid points using Kaiser-Bessel kernel
                    for offset in range(-width, width + 1):
                        idx = grid_idx + offset
                        if 0 <= idx < grid_size:
                            distance = abs(t_j - grid_times[idx]) / dt_grid
                            if distance <= width:
                                # Kaiser-Bessel kernel
                                from scipy.special import i0
                                arg = beta * np.sqrt(1 - (distance / width)**2)
                                kernel_val = i0(arg) / i0(beta)
                                grid_data[idx] += signal[j] * kernel_val
                
                # FFT the gridded data
                fft_result = np.fft.fft(grid_data)
                fft_result = np.fft.fftshift(fft_result)
                
                # Resample to desired frequency grid
                from scipy.interpolate import interp1d
                fft_freqs = np.fft.fftshift(np.fft.fftfreq(grid_size, dt_grid))
                interp_func = interp1d(fft_freqs, fft_result, kind='linear', bounds_error=False, fill_value=0)
                range_doppler[:, range_bin] = interp_func(doppler_freqs_hz)
        
        elif nufft_kernel == 'gaussian':
            # Gaussian gridding approximation
            oversample = 2
            grid_size = num_doppler_bins * oversample
            
            grid_times = np.linspace(0, t_span, grid_size)
            dt_grid = t_span / (grid_size - 1) if grid_size > 1 else 1.0
            
            # Gaussian parameters
            sigma = 1.5  # Width parameter
            truncate = 3.0  # Truncate at 3 sigma
            
            for range_bin in range(num_range_bins):
                signal = pulses_trimmed[:, range_bin] * slow_time_window
                
                # Grid using Gaussian kernel
                grid_data = np.zeros(grid_size, dtype=complex)
                for j, t_j in enumerate(pulse_times_relative_s):
                    grid_idx = int(t_j / dt_grid)
                    width = int(truncate * sigma)
                    
                    for offset in range(-width, width + 1):
                        idx = grid_idx + offset
                        if 0 <= idx < grid_size:
                            distance = (t_j - grid_times[idx]) / dt_grid
                            kernel_val = np.exp(-distance**2 / (2 * sigma**2))
                            grid_data[idx] += signal[j] * kernel_val
                
                # FFT and resample
                fft_result = np.fft.fftshift(np.fft.fft(grid_data))
                fft_freqs = np.fft.fftshift(np.fft.fftfreq(grid_size, dt_grid))
                
                from scipy.interpolate import interp1d
                interp_func = interp1d(fft_freqs, fft_result, kind='linear', bounds_error=False, fill_value=0)
                range_doppler[:, range_bin] = interp_func(doppler_freqs_hz)
        
        else:
            # Fall back to direct if unknown kernel
            normalization = 1.0 / np.sqrt(num_pulses)
            for range_bin in range(num_range_bins):
                signal = pulses_trimmed[:, range_bin] * slow_time_window
                for k, freq in enumerate(doppler_freqs_hz):
                    range_doppler[k, range_bin] = normalization * np.sum(signal * np.exp(-2j * np.pi * freq * pulse_times_relative_s))
    
    # Convert to dB
    range_doppler_db = 10 * np.log10(np.abs(range_doppler) ** 2 + 1e-12)
    
    return {
        'range_doppler': range_doppler,
        'range_doppler_db': range_doppler_db,
        'doppler_freqs_hz': doppler_freqs_hz,
        'doppler_resolution_hz': doppler_resolution_hz,
        'num_range_bins': num_range_bins,
        'pulse_times_relative_s': pulse_times_relative_s,
        'nyquist_doppler': nyquist_doppler,
        'min_prf': min_prf,
        'is_uniform_prf': is_uniform_prf
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
    range_window_type = metadata.get('range_window_type', 'hamming')
    min_prf_hz = float(metadata.get('min_prf_hz', 800))
    max_prf_hz = float(metadata.get('max_prf_hz', 2400))
    auto_detect_prf = bool(metadata.get('auto_detect_prf', False))
    use_fixed_prfs = bool(metadata.get('use_fixed_prfs', False))
    fixed_prfs_hz = str(metadata.get('fixed_prfs_hz', '1000,1200,1500'))
    nufft_kernel = metadata.get('nufft_kernel', 'direct')
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
            'range_window_type': range_window_type,
            'sample_rate_hz': sample_rate_hz,
            'min_prf_hz': min_prf_hz,
            'max_prf_hz': max_prf_hz,
            'auto_detect_prf': auto_detect_prf,
            'use_fixed_prfs': use_fixed_prfs,
            'fixed_prfs_hz': fixed_prfs_hz,
            'nufft_kernel': nufft_kernel,
            'shortest_pri_samples': shortest_pri_samples,
            'total_samples': total_samples
        },
        label="Provide Data",
        inputs=[],
        outputs=[
            ('signal_data', 'signal_data'),
            ('tx_wfm', 'tx_wfm'),
            ('window_type', 'window_type'),
            ('range_window_type', 'range_window_type'),
            ('sample_rate_hz', 'sample_rate_hz'),
            ('min_prf_hz', 'min_prf_hz'),
            ('max_prf_hz', 'max_prf_hz'),
            ('auto_detect_prf', 'auto_detect_prf'),
            ('use_fixed_prfs', 'use_fixed_prfs'),
            ('fixed_prfs_hz', 'fixed_prfs_hz'),
            ('nufft_kernel', 'nufft_kernel'),
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
            ('range_window_type', 'range_window_type')
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
    
    # Auto-detect PRFs from PRI distribution
    graph.add(
        node_detect_prfs,
        label="Detect PRFs (Cluster & Snap)",
        inputs=[
            ('pris_us', 'pris_us'),
            ('auto_detect_prf', 'auto_detect_prf'),
            ('use_fixed_prfs', 'use_fixed_prfs'),
            ('fixed_prfs_hz', 'fixed_prfs_hz'),
            ('min_prf_hz', 'min_prf_hz'),
            ('max_prf_hz', 'max_prf_hz')
        ],
        outputs=[
            ('detected_min_prf_hz', 'detected_min_prf_hz'),
            ('detected_max_prf_hz', 'detected_max_prf_hz'),
            ('detected_prfs_hz', 'detected_prfs_hz'),
            ('pri_clusters', 'pri_clusters')
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
            ('num_pulses', 'num_pulses'),
            ('shortest_pri_samples', 'shortest_pri_samples')
        ],
        outputs=[
            ('pulses_extracted', 'pulses_extracted'),
            ('pulses_extracted_db', 'pulses_extracted_db'),
            ('extraction_window_samples', 'extraction_window_samples')
        ]
    )
    
    # NUFFT Doppler compression
    graph.add(
        node_nufft_doppler,
        label="NUFFT Doppler",
        inputs=[
            ('pulses_extracted', 'pulses_extracted'),
            ('pulse_positions_samples', 'pulse_positions_samples'),
            ('pris_us', 'pris_us'),
            ('sample_rate_hz', 'sample_rate_hz'),
            ('num_pulses', 'num_pulses'),
            ('extraction_window_samples', 'extraction_window_samples'),
            ('window_type', 'window_type'),
            ('nufft_kernel', 'nufft_kernel')
        ],
        outputs=[
            ('range_doppler', 'range_doppler'),
            ('range_doppler_db', 'range_doppler_db'),
            ('doppler_freqs_hz', 'doppler_freqs_hz'),
            ('doppler_resolution_hz', 'doppler_resolution_hz'),
            ('num_range_bins', 'num_range_bins'),
            ('is_uniform_prf', 'is_uniform_prf')
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
    max_prf_hz = float(metadata.get('max_prf_hz', 2500))
    auto_detect_prf = bool(metadata.get('auto_detect_prf', False))
    
    total_samples = results.get('total_samples', 0)
    total_time_ms = total_samples / sample_rate_hz * 1000
    shortest_pri_samples = results.get('shortest_pri_samples', 1000)
    shortest_pri_us = shortest_pri_samples / sample_rate_hz * 1e6
    
    file_header_kvps = metadata.get('file_header_kvps', {})
    
    # Check if PRFs were detected
    detected_min_prf_hz = results.get('detected_min_prf_hz', min_prf_hz)
    detected_max_prf_hz = results.get('detected_max_prf_hz', max_prf_hz)
    detected_prfs_hz = results.get('detected_prfs_hz')
    
    summary_rows = [
        ["Total Samples", f"{total_samples:,}"],
        ["Sample Rate", f"{sample_rate_hz/1e6:.1f} MHz"],
        ["Total Duration", f"{total_time_ms:.2f} ms"],
        ["Range Window", window_type],
    ]
    
    if auto_detect_prf and detected_prfs_hz is not None:
        prfs_str = ", ".join([f"{p:.0f}" for p in detected_prfs_hz])
        summary_rows.extend([
            ["Auto-Detect PRF", "Enabled"],
            ["Detected PRFs (Hz)", prfs_str],
            ["PRF Range Used", f"{detected_min_prf_hz:.0f} - {detected_max_prf_hz:.0f} Hz"],
        ])
    else:
        summary_rows.append(["PRF Search Range", f"{min_prf_hz:.0f} - {max_prf_hz:.0f} Hz"])
    
    summary_rows.append(["Shortest PRI", f"{shortest_pri_us:.2f} μs ({shortest_pri_samples} samples)"])
    
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
    
    # Add detection diagnostics
    num_pulses = results.get('num_pulses', 0)
    num_windows = results.get('num_windows', 0)
    pulse_ratio = num_pulses / num_windows if num_windows > 0 else 0
    is_pulsed = num_pulses > 2 and num_windows > 0 and pulse_ratio < 0.9
    summary_rows.extend([
        ["", ""],
        ["**Detection**", ""],
        ["Detected Pulses", str(num_pulses)],
        ["Total Windows", str(num_windows)],
        ["Pulse Ratio", f"{pulse_ratio:.2%}"],
        ["Data Type", "Pulsed" if is_pulsed else "Continuous"],
    ])
    
    # Add NUFFT diagnostics if available
    nyquist_doppler = results.get('nyquist_doppler')
    min_prf = results.get('min_prf')
    doppler_resolution_hz = results.get('doppler_resolution_hz')
    if nyquist_doppler is not None and is_pulsed:
        summary_rows.extend([
            ["", ""],
            ["**NUFFT Doppler**", ""],
            ["Min PRF (Hz)", f"{min_prf:.2f}" if min_prf else "N/A"],
            ["Nyquist Doppler (Hz)", f"{nyquist_doppler:.2f}"],
            ["Doppler Resolution (Hz)", f"{doppler_resolution_hz:.2f}" if doppler_resolution_hz else "N/A"],
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
    max_prf_hz = float(metadata.get('max_prf_hz', 2400))
    shortest_pri_samples = results.get('shortest_pri_samples', 1000)
    
    # Determine if this is pulsed data based on pulse detection results
    num_pulses = results.get('num_pulses', 0)
    num_windows = results.get('num_windows', 0)
    is_pulsed = num_pulses > 2 and num_windows > 0 and (num_pulses / num_windows) < 0.9
    
    # Plot 1: Fixed-PRF pulse array heatmap (only for pulsed data)
    windows_2d_db = results.get('windows_2d_db')
    num_windows = results.get('num_windows', 0)
    if is_pulsed and windows_2d_db is not None:
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
    elif not is_pulsed and windows_2d_db is not None:
        # For continuous data, show 1D average power profile
        avg_power_db = np.mean(windows_2d_db, axis=0)
        fast_time_us = np.arange(len(avg_power_db)) / sample_rate_hz * 1e6
        
        fig2_cont = go.Figure()
        fig2_cont.add_trace(go.Scatter(
            x=fast_time_us, y=avg_power_db,
            mode='lines', name='Average Power',
            line=dict(color='cyan', width=1)
        ))
        fig2_cont.update_layout(
            title="Average Power Profile (Continuous Waveform)",
            xaxis_title="Fast Time (μs)",
            yaxis_title="Power (dB)",
            template='plotly_dark',
            height=600
        )
        workflow.add_plot(fig2_cont)
    
    # Plot 2: Pulse detection (K-means clustering result)
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
    
    # Plot 3: Pulses-only heatmap (only for pulsed data)
    pulses_2d_db = results.get('pulses_2d_db')
    if is_pulsed and pulses_2d_db is not None:
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
    elif not is_pulsed:
        # For continuous data, show signal amplitude and phase as 1D plots
        pulses_2d = results.get('pulses_2d')
        if pulses_2d is not None and len(pulses_2d) > 0:
            # Average across all windows for continuous data
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
            workflow.add_plot(fig_amp)
            
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
            workflow.add_plot(fig_phase)
    
    # Plot 4: PRI plot (only for pulsed data)
    pris_us = results.get('pris_us')
    auto_detect_prf = bool(metadata.get('auto_detect_prf', False))
    detected_min_prf_hz = results.get('detected_min_prf_hz', min_prf_hz)
    detected_max_prf_hz = results.get('detected_max_prf_hz', max_prf_hz)
    
    # Use detected PRFs if auto-detect is enabled
    prf_min_display = detected_min_prf_hz if auto_detect_prf else min_prf_hz
    prf_max_display = detected_max_prf_hz if auto_detect_prf else max_prf_hz
    
    if is_pulsed and pris_us is not None and len(pris_us) > 0:
        fig3d = go.Figure()
        fig3d.add_trace(go.Scatter(
            x=np.arange(len(pris_us)),
            y=pris_us,
            mode='markers+lines',
            name='PRI',
            marker=dict(color='cyan', size=4),
            line=dict(color='cyan', width=1)
        ))
        
        min_pri_us = 1e6 / prf_max_display
        max_pri_us = 1e6 / prf_min_display
        fig3d.add_trace(go.Scatter(
            x=[0, len(pris_us)-1],
            y=[min_pri_us, min_pri_us],
            mode='lines',
            name=f'Min PRI ({min_pri_us:.1f} μs, {prf_max_display:.0f} Hz)',
            line=dict(color='yellow', width=1, dash='dash')
        ))
        fig3d.add_trace(go.Scatter(
            x=[0, len(pris_us)-1],
            y=[max_pri_us, max_pri_us],
            mode='lines',
            name=f'Max PRI ({max_pri_us:.1f} μs, {prf_min_display:.0f} Hz)',
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
    
    # Plot 4a: PRF clustering results (only when auto-detect or fixed PRFs is enabled)
    auto_detect_prf = bool(metadata.get('auto_detect_prf', False))
    use_fixed_prfs = bool(metadata.get('use_fixed_prfs', False))
    detected_prfs_hz = results.get('detected_prfs_hz')
    pri_clusters = results.get('pri_clusters')
    if is_pulsed and (auto_detect_prf or use_fixed_prfs) and detected_prfs_hz is not None and pris_us is not None and len(pris_us) > 0:
        # Convert PRIs to PRFs
        prfs_hz = 1e6 / pris_us
        
        fig_prf_cluster = go.Figure()
        
        # Plot PRF values colored by cluster
        if pri_clusters is not None:
            for cluster_id in range(len(detected_prfs_hz)):
                cluster_mask = pri_clusters == cluster_id
                if np.any(cluster_mask):
                    fig_prf_cluster.add_trace(go.Scatter(
                        x=np.where(cluster_mask)[0],
                        y=prfs_hz[cluster_mask],
                        mode='markers',
                        name=f'Cluster {cluster_id+1}: {detected_prfs_hz[cluster_id]:.0f} Hz',
                        marker=dict(size=6, opacity=0.7)
                    ))
        else:
            fig_prf_cluster.add_trace(go.Scatter(
                x=np.arange(len(prfs_hz)),
                y=prfs_hz,
                mode='markers',
                name='PRF Values',
                marker=dict(color='cyan', size=4)
            ))
        
        # Add horizontal lines for detected PRF centers
        for i, prf_hz in enumerate(detected_prfs_hz):
            fig_prf_cluster.add_trace(go.Scatter(
                x=[0, len(prfs_hz)-1],
                y=[prf_hz, prf_hz],
                mode='lines',
                name=f'PRF {i+1}: {prf_hz:.0f} Hz',
                line=dict(width=2, dash='dash'),
                showlegend=False
            ))
        
        prf_title = "Fixed PRF Snapping" if use_fixed_prfs else "Auto-Detected PRF Clusters (Cluster & Snap Method)"
        fig_prf_cluster.update_layout(
            title=prf_title,
            xaxis_title="Pulse Pair Number",
            yaxis_title="PRF (Hz)",
            template='plotly_dark',
            height=600,
            showlegend=True
        )
        workflow.add_plot(fig_prf_cluster)
    
    # Plot 5: PRI-extracted pulses heatmap (only for pulsed data)
    pulses_extracted_db = results.get('pulses_extracted_db')
    extraction_window_samples = results.get('extraction_window_samples', 1000)
    if is_pulsed and pulses_extracted_db is not None:
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
    
    # Plot 6: Range-Doppler map (only for pulsed data)
    range_doppler_db = results.get('range_doppler_db')
    doppler_freqs_hz = results.get('doppler_freqs_hz')
    is_uniform_prf = results.get('is_uniform_prf', False)
    if is_pulsed and range_doppler_db is not None and doppler_freqs_hz is not None:
        heatmap_rd, skip_x_rd, skip_y_rd = downsample_heatmap(range_doppler_db, max_width=2000, max_height=1000)
        fast_time_rd_us = np.arange(heatmap_rd.shape[1]) * skip_x_rd / sample_rate_hz * 1e6
        doppler_freqs_display = doppler_freqs_hz[::skip_y_rd] if skip_y_rd < len(doppler_freqs_hz) else doppler_freqs_hz
        
        # Calculate default dynamic range
        rd_peak = np.max(heatmap_rd)
        rd_min_default = rd_peak - 60
        
        fig5 = go.Figure(data=go.Heatmap(
            z=heatmap_rd,
            x=fast_time_rd_us,
            y=doppler_freqs_display[:heatmap_rd.shape[0]],
            colorscale='Jet',
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
        fig5.update_layout(
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
        workflow.add_plot(fig5)
