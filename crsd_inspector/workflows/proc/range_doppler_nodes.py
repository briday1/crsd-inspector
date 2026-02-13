"""
Range-Doppler processing node functions for dagex graph
"""
import numpy as np
from scipy.signal import correlate
from sklearn.cluster import KMeans


def matched_filter(inputs):
    """Apply matched filter to continuous signal"""
    signal_data = inputs['signal_data']
    tx_wfm = inputs['tx_wfm']
    range_window_type = inputs['range_window_type']
    
    if signal_data.ndim > 1:
        signal_data = signal_data.ravel()
    
    # Use TX waveform as-is without modification
    ref_wfm = tx_wfm.copy()
    
    mf_output = correlate(signal_data, np.conj(ref_wfm), mode='same', method='fft')
    mf_output_db = 10 * np.log10(np.abs(mf_output) ** 2 + 1e-12)
    
    return {
        'mf_output': mf_output,
        'mf_output_db': mf_output_db
    }


def fixed_prf_windows(inputs):
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


def pulse_detection(inputs):
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
    
    # Determine which cluster is the null/empty pulses (lowest power)
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


def pulse_timing(inputs):
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


def detect_prfs(inputs):
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


def filter_pulses_by_prf_bounds(inputs):
    """Filter out pulses whose measured PRI falls outside the detected PRF bounds"""
    detected_min_prf_hz = inputs.get('detected_min_prf_hz')
    detected_max_prf_hz = inputs.get('detected_max_prf_hz')
    pulse_positions_samples = inputs['pulse_positions_samples']
    pris_us = inputs['pris_us']
    num_pulses = inputs['num_pulses']
    
    # Always need bounds to filter
    if detected_min_prf_hz is None or detected_max_prf_hz is None or num_pulses < 2:
        return {
            'pulse_positions_samples': pulse_positions_samples,
            'pris_us': pris_us,
            'num_pulses': num_pulses
        }
    
    # Convert PRF bounds to PRI bounds (in microseconds)
    # PRF = 1e6 / PRI, so PRI = 1e6 / PRF
    max_pri_us = 1e6 / detected_min_prf_hz  # Slowest (longest) acceptable PRI
    min_pri_us = 1e6 / detected_max_prf_hz  # Fastest (shortest) acceptable PRI
    
    # Find which PRIs are within bounds
    pri_valid_mask = (pris_us >= min_pri_us) & (pris_us <= max_pri_us)
    valid_indices = np.where(pri_valid_mask)[0]
    
    # If no valid PRIs found, still filter (don't pass through unchanged)
    # This prevents propagating out-of-bounds pulses
    filtered_positions = pulse_positions_samples[valid_indices] if len(valid_indices) > 0 else np.array([])
    filtered_pris = pris_us[valid_indices] if len(valid_indices) > 0 else np.array([])
    filtered_num = len(valid_indices)
    
    return {
        'pulse_positions_samples': filtered_positions,
        'pris_us': filtered_pris,
        'num_pulses': filtered_num
    }


def pri_based_extraction(inputs):
    """Re-extract pulses using estimated PRIs"""
    mf_output = inputs['mf_output']
    pulse_positions_samples = inputs['pulse_positions_samples']
    pris_us = inputs['pris_us']
    sample_rate_hz = inputs['sample_rate_hz']
    num_pulses = inputs['num_pulses']
    shortest_pri_samples = inputs['shortest_pri_samples']
    
    # Set extraction window to the longest selected PRI.
    # This keeps each extracted pulse long enough for the slowest pulse spacing.
    if pris_us is not None and len(pris_us) > 0:
        longest_pri_us = float(np.max(pris_us))
        extraction_window_samples = max(1, int(np.ceil(longest_pri_us * sample_rate_hz / 1e6)))
    else:
        # Fallback when PRI estimates are unavailable
        extraction_window_samples = max(1, int(shortest_pri_samples))
    
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


def nufft_doppler(inputs, make_window_fn):
    """Perform NUFFT Doppler compression on extracted pulses (or FFT if uniform PRF)"""
    try:
        pulses_extracted = inputs['pulses_extracted']
        pulse_positions_samples = inputs['pulse_positions_samples']
        pris_us = inputs['pris_us']
        sample_rate_hz = inputs['sample_rate_hz']
        num_pulses = inputs['num_pulses']
        extraction_window_samples = inputs['extraction_window_samples']
        window_type = inputs['window_type']
        nufft_kernel = inputs.get('nufft_kernel', 'direct')
        pulses_to_integrate = int(inputs.get('pulses_to_integrate', 0))
        range_cropping_percentage = float(inputs.get('range_cropping_percentage', 0))
        
        # Debug check
        if num_pulses < 2:
            return {
                'range_doppler': np.array([[]]),
                'range_doppler_db': np.array([[]]),
                'doppler_freqs_hz': np.array([]),
                'doppler_resolution_hz': 0,
                'num_range_bins': 0,
                'pulse_times_relative_s': np.array([]),
                'nyquist_doppler': 0,
                'min_prf': 0,
                'is_uniform_prf': False
            }
        
        # Optionally limit number of pulses used in Doppler integration.
        # 0 means use all available pulses.
        if pulses_to_integrate > 0:
            pulses_to_use = min(num_pulses, pulses_to_integrate)
            pulse_positions_samples = pulse_positions_samples[:pulses_to_use]
            pulses_extracted = pulses_extracted[:pulses_to_use, :]
            if pris_us is not None and len(pris_us) > 0:
                pris_us = pris_us[:max(1, pulses_to_use - 1)]
            num_pulses = pulses_to_use

        # Use the actual pulse positions (in samples) from PRI-based extraction
        pulse_times_s = pulse_positions_samples / sample_rate_hz
        pulse_times_relative_s = pulse_times_s - pulse_times_s[0]
        t_span = pulse_times_relative_s[-1]
        
        # Apply window function across slow time (Doppler dimension)
        slow_time_window = make_window_fn(num_pulses, window_type)
        
        # Compute Doppler parameters
        max_pri_s = np.max(pris_us) / 1e6 if num_pulses > 1 else 1.0
        min_prf = 1.0 / max_pri_s
        nyquist_doppler = min_prf / 2.0
        doppler_resolution_hz = 1.0 / t_span if t_span > 0 else 1.0
        num_doppler_bins = num_pulses
        doppler_freqs_hz = np.linspace(-nyquist_doppler, nyquist_doppler, num_doppler_bins)
        
        # Optional range cropping for faster computation / focused map.
        # 0% means no crop, 100% keeps only 1 bin.
        crop_pct = min(max(range_cropping_percentage, 0.0), 100.0)
        if crop_pct > 0:
            keep_fraction = max(0.0, 1.0 - crop_pct / 100.0)
            max_range_bins = max(1, int(np.ceil(extraction_window_samples * keep_fraction)))
        else:
            max_range_bins = extraction_window_samples
        pulses_trimmed = pulses_extracted[:, :max_range_bins]
        num_range_bins = pulses_trimmed.shape[1]
        
        # Check if PRF is uniform
        pri_variation = np.std(pris_us) / np.mean(pris_us) if len(pris_us) > 0 else 0
        is_uniform_prf = pri_variation < 0.01
        
        if is_uniform_prf and num_pulses > 1:
            # Use regular FFT for uniform PRF
            avg_prf = 1e6 / np.mean(pris_us)
            nyquist_doppler = avg_prf / 2.0
            range_doppler = np.zeros((num_doppler_bins, num_range_bins), dtype=complex)
            
            for range_bin in range(num_range_bins):
                signal = pulses_trimmed[:, range_bin] * slow_time_window
                fft_result = np.fft.fft(signal)
                range_doppler[:, range_bin] = np.fft.fftshift(fft_result)
            
            doppler_freqs_hz = np.fft.fftshift(np.fft.fftfreq(num_pulses, 1.0 / avg_prf))
            doppler_resolution_hz = avg_prf / num_pulses
        else:
            # Use direct NUFFT for non-uniform PRF
            range_doppler = np.zeros((num_doppler_bins, num_range_bins), dtype=complex)
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
    
    except Exception as e:
        import traceback
        return {
            'range_doppler': np.array([[]]),
            'range_doppler_db': np.array([[]]),
            'doppler_freqs_hz': np.array([]),
            'doppler_resolution_hz': 0,
            'num_range_bins': 0,
            'pulse_times_relative_s': np.array([]),
            'nyquist_doppler': 0,
            'min_prf': 0,
            'is_uniform_prf': False
        }
