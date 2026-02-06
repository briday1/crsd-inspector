"""
Pulse Extraction Workflow

Detects pulse start times from continuous CRSD data and performs pulse stacking.
Handles both uniform and staggered PRF patterns.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from crsd_inspector.workflows.workflow import Workflow
from scipy.signal import find_peaks, correlate


# Create workflow instance
workflow = Workflow(
    name="Pulse Extraction",
    description="Detect pulse start times and perform pulse stacking from continuous data"
)

# Workflow parameters
PARAMS = {
    'min_prf_hz': {
        'label': 'Min PRF (Hz)',
        'type': 'number',
        'default': 100,
        'min': 1,
        'max': 100000,
        'step': 10,
    },
    'max_prf_hz': {
        'label': 'Max PRF (Hz)',
        'type': 'number',
        'default': 10000,
        'min': 100,
        'max': 100000,
        'step': 100,
    },
    'power_threshold_db': {
        'label': 'Power Threshold (dB)',
        'type': 'number',
        'default': -20,
        'min': -100,
        'max': 0,
        'step': 1,
    },
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
}


def detect_pulse_starts(signal_data, sample_rate_hz, reference_waveform=None, 
                        min_prf_hz=100, max_prf_hz=10000, power_threshold_db=-20, window_type='hamming'):
    """
    Detect pulse start times from continuous signal data using matched filtering
    
    Args:
        signal_data: Continuous signal (1 x num_samples) or (num_samples,)
        sample_rate_hz: Sample rate in Hz
        reference_waveform: Reference pulse for matched filtering (if None, uses energy detection)
        min_prf_hz: Minimum expected PRF (Hz) - sets max distance between pulses
        max_prf_hz: Maximum expected PRF (Hz) - sets min distance between pulses
        power_threshold_db: Detection threshold relative to peak (dB)
        window_type: Window function for range processing ('none', 'hamming', 'hanning', 'blackman')
        
    Returns:
        pulse_start_samples: Array of pulse start sample indices
        pulse_start_times: Array of pulse start times (seconds)
        detection_info: Dict with detection statistics (includes mf_output_db for visualization)
    """
    # Ensure 1D
    if signal_data.ndim > 1:
        signal_data = signal_data.ravel()
    
    # Matched filtering if reference waveform provided
    if reference_waveform is not None:
        # Apply window to reference waveform for sidelobe control
        ref_wfm = reference_waveform.copy()
        if window_type and window_type != 'none':
            if window_type == 'hamming':
                window_func = np.hamming(len(ref_wfm))
            elif window_type == 'hanning':
                window_func = np.hanning(len(ref_wfm))
            elif window_type == 'blackman':
                window_func = np.blackman(len(ref_wfm))
            else:
                window_func = np.ones(len(ref_wfm))
            ref_wfm = ref_wfm * window_func
        
        # Matched filter the entire continuous signal
        # The MF output will show high power where pulses exist and low power in gaps
        print(f"Applying matched filter to {len(signal_data)} samples...")
        mf_output = correlate(signal_data, np.conj(ref_wfm), mode='same')
        mf_power = np.abs(mf_output) ** 2
        mf_output_db = 10 * np.log10(mf_power + 1e-12)
        
        # The matched filter naturally compresses the pulse
        # For pulse detection: find transitions from low to high power (pulse starts)
        # Use the MF power directly
        power = mf_power
        use_matched_filter = True
    else:
        # Fallback to instantaneous power
        power = np.abs(signal_data) ** 2
        mf_output_db = None
        use_matched_filter = False
    
    # Convert to dB for visualization
    power_db = 10 * np.log10(power + 1e-12)
    
    # For matched filter output, find pulse starts from power transitions
    if use_matched_filter:
        # The MF output will have contiguous regions of high power (the pulse duration)
        # separated by regions of low power (inter-pulse gaps).
        # We want to find the START of each high-power region.
        
        # Minimum distance between pulses based on max PRF
        min_distance_samples = int(sample_rate_hz / max_prf_hz * 0.8)
        
        # Adaptive threshold
        peak_power_db = np.max(power_db)
        threshold_db = peak_power_db + power_threshold_db
        threshold_linear = 10 ** (threshold_db / 10)
        
        # Find regions above threshold
        above_threshold = power > threshold_linear
        
        # Find ALL rising edges
        transitions = np.diff(above_threshold.astype(int))
        all_rising_edges = np.where(transitions == 1)[0] + 1
        
        # Group rising edges that are close together (within same pulse)
        # Only keep the FIRST edge of each group
        if len(all_rising_edges) > 1:
            pulse_start_samples = [all_rising_edges[0]]
            for edge in all_rising_edges[1:]:
                # If this edge is far from the last selected edge, it's a new pulse
                if edge - pulse_start_samples[-1] >= min_distance_samples:
                    pulse_start_samples.append(edge)
            pulse_start_samples = np.array(pulse_start_samples)
        else:
            pulse_start_samples = all_rising_edges
        
        # For visualization, show the matched filter output
        smoothed_power_db = mf_output_db
        window_samples = 1  # Not using additional smoothing
        
    else:
        # Energy detection with sliding window for raw power
        # Window size: fraction of minimum PRI to capture pulse energy
        window_samples = int(sample_rate_hz / max_prf_hz / 4)  # 1/4 of minimum PRI
        window_samples = max(window_samples, 100)  # At least 100 samples
        
        # Convolve power with rectangular window (moving average)
        window = np.ones(window_samples) / window_samples
        smoothed_power = np.convolve(power, window, mode='same')
        
        # Convert to dB
        smoothed_power_db = 10 * np.log10(smoothed_power + 1e-12)
        
        # Adaptive threshold
        peak_power_db = np.max(smoothed_power_db)
        threshold_db = peak_power_db + power_threshold_db
        threshold_linear = 10 ** (threshold_db / 10)
        
        # Find regions above threshold (pulse bursts)
        above_threshold = smoothed_power > threshold_linear
        
        # Find rising edges (pulse starts)
        transitions = np.diff(above_threshold.astype(int))
        pulse_start_samples = np.where(transitions == 1)[0] + 1
        
        # Minimum distance between pulses
        min_distance_samples = int(sample_rate_hz / max_prf_hz * 0.8)  # 80% of minimum PRI
        
        # Filter out pulses too close together
        if len(pulse_start_samples) > 1:
            valid_pulses = [pulse_start_samples[0]]
            for pulse_start in pulse_start_samples[1:]:
                if pulse_start - valid_pulses[-1] >= min_distance_samples:
                    valid_pulses.append(pulse_start)
            pulse_start_samples = np.array(valid_pulses)
    
    if len(pulse_start_samples) == 0:
        return np.array([]), np.array([]), {
            'num_pulses': 0,
            'threshold_db': threshold_db,
            'peak_power_db': peak_power_db,
            'smoothed_power_db': smoothed_power_db,
            'mf_output_db': mf_output_db,
            'window_samples': window_samples,
            'error': 'No pulses detected above threshold'
        }
    
    # Convert to times
    pulse_start_times = pulse_start_samples / sample_rate_hz
    
    # Compute PRF statistics
    if len(pulse_start_samples) > 1:
        pris = np.diff(pulse_start_times)
        prfs = 1.0 / pris
        avg_prf = np.mean(prfs)
        min_prf = np.min(prfs)
        max_prf = np.max(prfs)
        std_prf = np.std(prfs)
        
        # Detect if staggered (std > 1% of mean)
        is_staggered = (std_prf / avg_prf) > 0.01
    else:
        avg_prf = min_prf = max_prf = std_prf = 0.0
        is_staggered = False
    
    detection_info = {
        'num_pulses': len(pulse_start_samples),
        'threshold_db': threshold_db,
        'peak_power_db': peak_power_db,
        'avg_prf_hz': float(avg_prf),
        'min_prf_hz': float(min_prf),
        'max_prf_hz': float(max_prf),
        'std_prf_hz': float(std_prf),
        'is_staggered': bool(is_staggered),
        'avg_pri_sec': float(1.0 / avg_prf) if avg_prf > 0 else 0.0,
        'smoothed_power_db': smoothed_power_db,
        'mf_output_db': mf_output_db,
        'window_samples': window_samples,
    }
    
    return pulse_start_samples, pulse_start_times, detection_info


def extract_pulses(signal_data, pulse_start_samples, samples_per_pulse):
    """
    Extract pulses from continuous data given start positions
    
    Args:
        signal_data: Continuous signal (1D array)
        pulse_start_samples: Array of pulse start sample indices
        samples_per_pulse: Number of samples to extract per pulse
        
    Returns:
        pulse_stack: Extracted pulses (num_pulses x samples_per_pulse)
        valid_pulses: Boolean mask of successfully extracted pulses
    """
    if signal_data.ndim > 1:
        signal_data = signal_data.ravel()
    
    num_pulses = len(pulse_start_samples)
    total_samples = len(signal_data)
    
    pulse_stack = np.zeros((num_pulses, samples_per_pulse), dtype=signal_data.dtype)
    valid_pulses = np.zeros(num_pulses, dtype=bool)
    
    for i, start_idx in enumerate(pulse_start_samples):
        end_idx = start_idx + samples_per_pulse
        
        if end_idx <= total_samples:
            # Full pulse fits
            pulse_stack[i, :] = signal_data[start_idx:end_idx]
            valid_pulses[i] = True
        elif start_idx < total_samples:
            # Partial pulse - zero pad
            available = total_samples - start_idx
            pulse_stack[i, :available] = signal_data[start_idx:total_samples]
            valid_pulses[i] = True
    
    return pulse_stack, valid_pulses


def run_workflow(signal_data, metadata=None, **kwargs):
    """
    Execute pulse extraction workflow
    
    Args:
        signal_data: Continuous CRSD signal data (1 x total_samples)
        metadata: Dict with:
            - sample_rate_hz: Sample rate
            - samples_per_pulse: Expected samples per pulse (for extraction)
            - min_prf_hz: Minimum PRF constraint (default: 100)
            - max_prf_hz: Maximum PRF constraint (default: 10000)
            - power_threshold_db: Detection threshold (default: -20)
    """
    workflow.clear()
    
    if metadata is None:
        metadata = {}
    
    # Extract parameters
    sample_rate_hz = metadata.get('sample_rate_hz', 100e6)
    min_prf_hz = metadata.get('min_prf_hz', 100)
    max_prf_hz = metadata.get('max_prf_hz', 10000)
    power_threshold_db = metadata.get('power_threshold_db', -20)
    window_type = metadata.get('window_type', 'hamming')
    tx_wfm = metadata.get('tx_wfm', None)  # Reference waveform for matched filtering
    
    # Ensure we have continuous data
    if signal_data.ndim == 1:
        signal_data = signal_data[None, :]  # Add dimension
    
    num_vectors, total_samples = signal_data.shape
    
    workflow.add_text(f"Pulse Extraction from Continuous Data")
    workflow.add_text([
        f"Total samples: {total_samples:,}",
        f"Sample rate: {sample_rate_hz/1e6:.1f} MHz",
        f"Total time: {total_samples/sample_rate_hz:.3f} seconds",
        f"PRF constraints: [{min_prf_hz:.0f}, {max_prf_hz:.0f}] Hz",
        f"Detection threshold: {power_threshold_db:.1f} dB",
        f"Matched filter: {'Enabled' if tx_wfm is not None else 'Disabled (energy detection)'}",
        f"Range window: {window_type.capitalize() if window_type != 'none' else 'None'}",
    ])
    
    try:
        # Step 1: Detect pulse start times
        pulse_start_samples, pulse_start_times, detection_info = detect_pulse_starts(
            signal_data,
            sample_rate_hz,
            reference_waveform=tx_wfm,
            min_prf_hz=min_prf_hz,
            max_prf_hz=max_prf_hz,
            power_threshold_db=power_threshold_db,
            window_type=window_type
        )
        
        num_pulses = detection_info['num_pulses']
        
        workflow.add_text(f"\nDetected {num_pulses} pulses")
        
        # Add PRF statistics
        if num_pulses > 1:
            prf_info = [
                f"Average PRF: {detection_info['avg_prf_hz']:.2f} Hz",
                f"PRF range: [{detection_info['min_prf_hz']:.2f}, {detection_info['max_prf_hz']:.2f}] Hz",
                f"Std deviation: {detection_info['std_prf_hz']:.2f} Hz",
                f"Average PRI: {detection_info['avg_pri_sec']*1000:.3f} ms",
                f"Stagger detected: {'Yes' if detection_info['is_staggered'] else 'No'}",
            ]
            workflow.add_text(prf_info)
        
        # Only extract pulses if we detected some
        if num_pulses > 0:
            # Step 2: Determine pulse width automatically
            # For continuous data: find where each pulse ends (falling edge after start)
            # Conservative: use typical radar duty cycle (~10%) or estimate from data
            if num_pulses > 1:
                # Use average PRI and assume ~10% duty cycle for pulse width
                avg_pri_samples = int(sample_rate_hz / detection_info['avg_prf_hz'])
                samples_per_pulse = max(512, avg_pri_samples // 10)  # At least 512 samples, or 10% of PRI
            else:
                # Single pulse - estimate from reference waveform
                if tx_wfm is not None:
                    samples_per_pulse = len(tx_wfm) * 2  # Conservative: 2x ref waveform
                else:
                    samples_per_pulse = 1000  # Default fallback
            
            workflow.add_text(f"\nEstimated pulse width: {samples_per_pulse} samples ({samples_per_pulse/sample_rate_hz*1e6:.2f} µs)")
            
            # Step 3: Extract pulses
            pulse_stack, valid_pulses = extract_pulses(
                signal_data,
                pulse_start_samples,
                samples_per_pulse
            )
            
            num_valid = np.sum(valid_pulses)
            workflow.add_text(f"Extracted {num_valid}/{num_pulses} valid pulses")
            workflow.add_text(f"Pulse stack shape: {pulse_stack.shape}")
        else:
            workflow.add_text("\nNo pulses to extract. Try adjusting the detection threshold.")
            num_valid = 0
            pulse_stack = None
            samples_per_pulse = 0
        
        # Create statistics table
        stats_table = {
            'Metric': [
                'Pulses Detected',
                'Valid Pulses',
                'Pulse Width (samples)',
                'Average PRF (Hz)',
                'PRF Std Dev (Hz)',
                'Staggered',
                'Threshold (dB)',
                'Peak Power (dB)',
            ],
            'Value': [
                f"{num_pulses}",
                f"{num_valid}",
                f"{samples_per_pulse}" if samples_per_pulse > 0 else "N/A",
                f"{detection_info['avg_prf_hz']:.2f}" if num_pulses > 1 else "N/A",
                f"{detection_info['std_prf_hz']:.2f}" if num_pulses > 1 else "N/A",
                "Yes" if detection_info.get('is_staggered') else "No",
                f"{detection_info['threshold_db']:.1f}",
                f"{detection_info['peak_power_db']:.1f}",
            ]
        }
        workflow.add_table("Pulse Extraction Statistics", stats_table)
        
        # Step 4: Create visualization (always show energy detector plot)
        num_plots = 2 if num_pulses == 0 else 4  # Show fewer plots if no pulses detected
        
        if num_pulses == 0:
            # Just show energy detector output
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    'Energy Detector Output (Full View)',
                    'Energy Detector Output (Zoom)',
                ),
                vertical_spacing=0.12,
                row_heights=[0.5, 0.5]
            )
        else:
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=(
                    'Power Envelope with Detection Threshold',
                    'Detected Pulse Starts (Zoom)',
                    'Pulse Timing Diagram (PRI)',
                    'Extracted Pulse Stack (Range vs Pulse)'
                ),
                vertical_spacing=0.08,
                row_heights=[0.25, 0.25, 0.2, 0.3]
            )
        
        # Plot 1: Energy detector output with threshold
        time_axis = np.arange(total_samples) / sample_rate_hz
        
        # Check if we have matched filter output
        mf_output_db = detection_info.get('mf_output_db')
        if mf_output_db is not None:
            # Use matched filter output
            plot_title_1 = 'Matched Filter Output with Detection Threshold'
            plot_title_2 = 'Matched Filter Output (Zoom)'
            power_to_plot = mf_output_db
        else:
            # Use smoothed power from energy detector
            plot_title_1 = 'Energy Detector Output (Full View)'
            plot_title_2 = 'Energy Detector Output (Zoom)'
            smoothed_power_db = detection_info.get('smoothed_power_db')
            if smoothed_power_db is None:
                # Fallback if not available
                power = np.abs(signal_data.ravel()) ** 2
                power_to_plot = 10 * np.log10(power + 1e-12)
            else:
                power_to_plot = smoothed_power_db
        
        # Update subplot titles
        if num_pulses == 0:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(plot_title_1, plot_title_2),
                vertical_spacing=0.12,
                row_heights=[0.5, 0.5]
            )
        else:
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=(
                    plot_title_1,
                    plot_title_2,
                    'Pulse Timing Diagram (PRI)',
                    'Extracted Pulse Stack (Range vs Pulse)'
                ),
                vertical_spacing=0.08,
                row_heights=[0.25, 0.25, 0.2, 0.3]
            )
        
        # Downsample for plotting if too many samples
        plot_decimation = max(1, total_samples // 10000)
        time_plot = time_axis[::plot_decimation]
        power_plot = power_to_plot[::plot_decimation]
        
        # Main trace (matched filter or energy detector)
        fig.add_trace(
            go.Scatter(
                x=time_plot * 1000,  # ms
                y=power_plot,
                mode='lines',
                name='MF Output' if mf_output_db is not None else 'Energy Detector',
                line=dict(color='cyan', width=1),
                opacity=0.8,
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Threshold line
        fig.add_hline(
            y=detection_info['threshold_db'],
            line=dict(color='red', dash='dash', width=2),
            annotation_text=f"Threshold: {detection_info['threshold_db']:.1f} dB",
            annotation_position="right",
            row=1, col=1
        )
        # Peak power line
        fig.add_hline(
            y=detection_info['peak_power_db'],
            line=dict(color='green', dash='dot', width=1),
            annotation_text=f"Peak: {detection_info['peak_power_db']:.1f} dB",
            annotation_position="right",
            row=1, col=1
        )
        
        # Mark detected pulse starts
        fig.add_trace(
            go.Scatter(
                x=pulse_start_times * 1000,  # ms
                y=[detection_info['peak_power_db']] * len(pulse_start_times),
                mode='markers',
                name=f'Detected Pulses ({num_pulses})',
                marker=dict(color='yellow', size=6, symbol='x', line=dict(width=1, color='orange')),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Plot 2: Zoomed view of first few pulses showing threshold behavior
        # Show first 10 pulses or first 50ms, whichever is less
        zoom_time = 0.05  # 50ms
        zoom_samples = int(zoom_time * sample_rate_hz)
        zoom_decimation = max(1, zoom_samples // 5000)
        
        time_zoom = time_axis[:zoom_samples:zoom_decimation]
        power_zoom = power_to_plot[:zoom_samples:zoom_decimation]
        
        # Matched filter or energy detector
        fig.add_trace(
            go.Scatter(
                x=time_zoom * 1000,
                y=power_zoom,
                mode='lines',
                name='MF Output' if mf_output_db is not None else 'Energy Detector',
                line=dict(color='cyan', width=1.5),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Threshold in zoom
        fig.add_hline(
            y=detection_info['threshold_db'],
            line=dict(color='red', dash='dash', width=2),
            row=2, col=1
        )
        
        # Mark pulses in zoom window if any detected
        if num_pulses > 0:
            zoom_pulses = pulse_start_times[pulse_start_times < zoom_time]
            if len(zoom_pulses) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=zoom_pulses * 1000,
                        y=[detection_info['peak_power_db']] * len(zoom_pulses),
                        mode='markers',
                        name='Pulse Starts',
                        marker=dict(color='yellow', size=10, symbol='x', line=dict(width=2, color='orange')),
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # Plot 3: Pulse timing diagram (PRI vs pulse number) - only if pulses detected
        if num_pulses > 1:
            pris = np.diff(pulse_start_times) * 1000  # ms
            pulse_numbers = np.arange(1, len(pris) + 1)
            avg_pri_ms = detection_info['avg_pri_sec'] * 1000
            
            fig.add_trace(
                go.Scatter(
                    x=pulse_numbers,
                    y=pris,
                    mode='lines+markers',
                    name='PRI',
                    line=dict(color='magenta', width=2),
                    marker=dict(size=6),
                    showlegend=False
                ),
                row=3, col=1
            )
            
            # Average PRI line
            fig.add_hline(
                y=avg_pri_ms,
                line=dict(color='orange', dash='dash', width=1),
                annotation_text=f"Avg: {avg_pri_ms:.2f} ms",
                annotation_position="right",
                row=3, col=1
            )
        
        # Plot 4: Pulse stack heatmap - only if pulses detected and extracted
        if num_pulses > 0 and pulse_stack is not None:
            pulse_stack_db = 20 * np.log10(np.abs(pulse_stack) + 1e-10)
            
            fig.add_trace(
                go.Heatmap(
                    z=pulse_stack_db,
                    x=np.arange(samples_per_pulse),
                    y=np.arange(num_valid),
                    colorscale='Viridis',
                    colorbar=dict(title='dB', y=0.15, len=0.3),
                    name='Pulse Stack',
                    showscale=True
                ),
                row=4, col=1
            )
        
        # Update axes
        fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
        fig.update_yaxes(title_text="Power (dB)", row=1, col=1)
        
        fig.update_xaxes(title_text="Time (ms)", row=2, col=1)
        fig.update_yaxes(title_text="Power (dB)", row=2, col=1)
        
        fig.update_xaxes(title_text="Pulse Number", row=3, col=1)
        fig.update_yaxes(title_text="PRI (ms)", row=3, col=1)
        
        fig.update_xaxes(title_text="Range Sample", row=4, col=1)
        fig.update_yaxes(title_text="Pulse Number", row=4, col=1)
        
        fig.update_layout(
            template="plotly_dark",
            title_text="Pulse Extraction Results",
            height=1200,
            showlegend=True
        )
        
        workflow.add_plot(fig)
        
        return workflow.build()
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        workflow.add_text(f"Error: {str(e)}")
        return workflow.build()
