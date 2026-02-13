"""
Range Doppler Summary Node Functions

This module contains summary generation node functions for the range-doppler workflow.
Each function takes inputs from the graph context and returns formatted table data.
"""


def build_processing_summary(inputs):
    """
    Build processing parameters summary table
    
    Parameters
    ----------
    inputs : dict
        - sample_rate_hz: Sample rate in Hz
        - window_type: Window function type
        - min_prf_hz: Minimum PRF
        - max_prf_hz: Maximum PRF
        - auto_detect_prf: Boolean indicating if auto-detect was enabled
        - total_samples: Total number of samples
        - shortest_pri_samples: Shortest PRI in samples
        - tx_source: Human-readable TX source description
        - detected_min_prf_hz: Auto-detected minimum PRF (optional)
        - detected_max_prf_hz: Auto-detected maximum PRF (optional)
        - detected_prfs_hz: List of detected PRF values (optional)
        - file_header_kvps: File header key-value pairs (optional)
        - num_pulses: Number of detected pulses
        - num_windows: Number of windows
        - is_pulsed: Boolean indicating pulsed vs continuous data
        - nyquist_doppler: Nyquist Doppler frequency (optional)
        - min_prf: Minimum PRF from analysis (optional)
        - doppler_resolution_hz: Doppler resolution (optional)
    
    Returns
    -------
    dict
        - summary_table: Dict with "Parameter" and "Value" keys for workflow.add_table
    """
    sample_rate_hz = inputs.get('sample_rate_hz', 100e6)
    window_type = inputs.get('window_type', 'hamming')
    min_prf_hz = inputs.get('min_prf_hz', 800)
    max_prf_hz = inputs.get('max_prf_hz', 2500)
    auto_detect_prf = inputs.get('auto_detect_prf', False)
    
    total_samples = inputs.get('total_samples', 0)
    total_time_ms = total_samples / sample_rate_hz * 1000
    shortest_pri_samples = inputs.get('shortest_pri_samples', 1000)
    shortest_pri_us = shortest_pri_samples / sample_rate_hz * 1e6
    tx_source = inputs.get('tx_source', 'Unknown')
    
    file_header_kvps = inputs.get('file_header_kvps', {})
    
    # Check if PRFs were detected
    detected_min_prf_hz = inputs.get('detected_min_prf_hz', min_prf_hz)
    detected_max_prf_hz = inputs.get('detected_max_prf_hz', max_prf_hz)
    detected_prfs_hz = inputs.get('detected_prfs_hz')
    
    summary_rows = [
        ["Total Samples", f"{total_samples:,}"],
        ["Sample Rate", f"{sample_rate_hz/1e6:.1f} MHz"],
        ["Total Duration", f"{total_time_ms:.2f} ms"],
        ["Range Window", window_type],
        ["TX Source", tx_source],
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
    num_pulses = inputs.get('num_pulses', 0)
    num_windows = inputs.get('num_windows', 0)
    pulse_ratio = num_pulses / num_windows if num_windows > 0 else 0
    is_pulsed = inputs.get('is_pulsed', False)
    summary_rows.extend([
        ["", ""],
        ["**Detection**", ""],
        ["Detected Pulses", str(num_pulses)],
        ["Total Windows", str(num_windows)],
        ["Pulse Ratio", f"{pulse_ratio:.2%}"],
        ["Data Type", "Pulsed" if is_pulsed else "Continuous"],
    ])
    
    # Add NUFFT diagnostics if available
    nyquist_doppler = inputs.get('nyquist_doppler')
    min_prf = inputs.get('min_prf')
    doppler_resolution_hz = inputs.get('doppler_resolution_hz')
    if nyquist_doppler is not None and is_pulsed:
        summary_rows.extend([
            ["", ""],
            ["**NUFFT Doppler**", ""],
            ["Min PRF (Hz)", f"{min_prf:.2f}" if min_prf else "N/A"],
            ["Nyquist Doppler (Hz)", f"{nyquist_doppler:.2f}"],
            ["Doppler Resolution (Hz)", f"{doppler_resolution_hz:.2f}" if doppler_resolution_hz else "N/A"],
        ])
    
    summary_table = {
        "Parameter": [row[0] for row in summary_rows],
        "Value": [row[1] for row in summary_rows]
    }
    
    return {'summary_table': summary_table}
