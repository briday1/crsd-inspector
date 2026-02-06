"""
Range-Doppler Processing
Copied directly from test-dagex/example_workflow.py
"""

import numpy as np


def perform_range_doppler_processing(signal_data, tx_wfm, sample_rate_hz, prf_hz, verbose=False):
    """
    Perform range-Doppler processing on received signal
    
    Args:
        signal_data: Complex signal array (P, N) - pulses x samples
        tx_wfm: Transmit waveform
        sample_rate_hz: Sample rate in Hz (for range axis)
        prf_hz: Pulse repetition frequency in Hz (for Doppler axis)
        
    Returns:
        Dictionary with range-Doppler map and profiles
    """
    if verbose:
        print("    Performing range-Doppler processing...")
    
    num_pulses, num_samples = signal_data.shape
    
    # Step 1: Pulse compression (matched filter)
    if verbose:
        print(f"      - Matched filtering ({num_pulses} pulses)...")
    compressed = np.zeros_like(signal_data)
    matched_filter = np.conj(tx_wfm[::-1])  # Time-reversed conjugate
    
    # Compute matched filter delay compensation
    filter_delay = len(tx_wfm) // 2
    
    for p in range(num_pulses):
        conv_out = np.convolve(signal_data[p, :], matched_filter, mode='same')
        # Shift to compensate for matched filter group delay
        compressed[p, :] = np.roll(conv_out, -filter_delay)
    
    # Step 2: Doppler processing (FFT across slow time)
    if verbose:
        print(f"      - Doppler FFT...")
    range_doppler = np.fft.fftshift(np.fft.fft(compressed, axis=0), axes=0)
    
    # Convert to magnitude (dB)
    rd_mag = np.abs(range_doppler)
    rd_db = 20 * np.log10(rd_mag + 1e-10)
    
    # Step 3: Extract profiles
    # Range profile (coherent integration across pulses)
    range_profile = np.mean(rd_mag, axis=0)
    range_profile_db = 20 * np.log10(range_profile + 1e-10)
    
    # Doppler profile (sum across range bins)
    doppler_profile = np.mean(rd_mag, axis=1)
    doppler_profile_db = 20 * np.log10(doppler_profile + 1e-10)
    
    # Step 4: Compute statistics
    # Dynamic range
    peak_val = np.max(rd_db)
    noise_floor = np.percentile(rd_db, 5)  # 5th percentile as noise estimate
    dynamic_range = peak_val - noise_floor
    
    # Signal statistics
    mean_val = np.mean(rd_db)
    std_val = np.std(rd_db)
    
    # Peak detection
    peak_idx = np.unravel_index(np.argmax(rd_mag), rd_mag.shape)
    peak_doppler_idx, peak_range_idx = peak_idx
    
    # Compute axes
    c = 3e8
    range_axis_m = (np.arange(num_samples) / sample_rate_hz) * c / 2
    doppler_axis_hz = np.fft.fftshift(np.fft.fftfreq(num_pulses, d=1/prf_hz))  # Use PRF, not sample rate!
    
    return {
        'range_doppler_db': rd_db,
        'range_profile_db': range_profile_db,
        'doppler_profile_db': doppler_profile_db,
        'range_axis_m': range_axis_m,
        'doppler_axis_hz': doppler_axis_hz,
        'compressed_signal': compressed,
        # Statistics
        'peak_db': peak_val,
        'noise_floor_db': noise_floor,
        'dynamic_range_db': dynamic_range,
        'mean_db': mean_val,
        'std_db': std_val,
        'peak_range_m': range_axis_m[peak_range_idx],
        'peak_doppler_hz': doppler_axis_hz[peak_doppler_idx],
        # Raw signal statistics
        'raw_peak_db': 20 * np.log10(np.max(np.abs(signal_data)) + 1e-10),
        'raw_mean_db': 20 * np.log10(np.mean(np.abs(signal_data)) + 1e-10),
        'raw_std_db': 20 * np.log10(np.std(np.abs(signal_data)) + 1e-10),
    }
