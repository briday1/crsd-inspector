"""
Signal Analysis Workflow Node Functions

This module contains the processing node functions for the signal analysis workflow.
Each function is designed to be used as a node in a dagex Graph.
"""

import numpy as np


def extract_prf_from_metadata(metadata):
    """
    Extract PRF from CRSD file metadata if available
    
    Parameters
    ----------
    metadata : dict
        Workflow metadata containing file_header_kvps and tx_crsd_file
    
    Returns
    -------
    float or None
        PRF in Hz if found, None otherwise
    """
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


def extract_pulses(inputs):
    """
    Extract individual pulses from continuous signal data
    
    Parameters
    ----------
    inputs : dict
        - signal_data: Complex signal array (channels x samples)
        - pri_samples: PRI duration in samples
        - num_pulses_to_stack: Number of pulses to extract (-1 for all)
        - total_samples: Total number of samples in signal
    
    Returns
    -------
    dict
        - pulses: Array of extracted pulses (num_pulses x pri_samples)
        - actual_num_pulses: Number of pulses actually extracted
    """
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


def compute_statistics(inputs):
    """
    Compute comprehensive statistics on pulse-stacked data
    
    Parameters
    ----------
    inputs : dict
        - pulses: Array of pulses (num_pulses x pri_samples)
    
    Returns
    -------
    dict
        - amplitude: Amplitude array (num_pulses x pri_samples)
        - phase: Phase array (num_pulses x pri_samples)
        - amplitude_stats: Amplitude statistics table
        - phase_stats: Phase statistics table
        - quality_metrics: Quality metrics table (SNR, dynamic range, clipping)
        - iq_stats: I/Q component statistics table
        - amp_flat: Flattened amplitude array
        - phase_flat: Flattened phase array
        - pulses: Pass-through of input pulses for PSD calculation
    """
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
