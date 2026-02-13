"""
Signal Analysis Summary Table Generation
Builds analysis parameters summary table as graph node
"""


def build_analysis_summary(inputs):
    """
    Build analysis parameters summary table
    
    Args:
        inputs: dict with:
            - prf_hz: Pulse Repetition Frequency in Hz
            - sample_rate_hz: Sample rate in Hz
            - pri_samples: Pulse Repetition Interval in samples
            - actual_num_pulses: Number of pulses extracted
            - downsample_factor: Range downsampling factor
        
    Returns:
        dict: {'params_table': table dict with Parameter/Value columns}
    """
    prf_hz = inputs.get('prf_hz')
    sample_rate_hz = inputs.get('sample_rate_hz')
    pri_samples = inputs.get('pri_samples')
    actual_num_pulses = inputs.get('actual_num_pulses')
    downsample_factor = inputs.get('downsample_factor')
    
    params_table = {
        "Parameter": [
            "PRF",
            "Sample Rate",
            "PRI (samples)",
            "Pulses Extracted",
            "Range Downsample"
        ],
        "Value": [
            f"{prf_hz:.1f} Hz",
            f"{sample_rate_hz/1e6:.1f} MHz",
            f"{pri_samples}",
            f"{actual_num_pulses}",
            f"{downsample_factor}x"
        ]
    }
    
    return {'params_table': params_table}
