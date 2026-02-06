"""
Range-Doppler Processing Workflow

Performs matched filtering and Doppler FFT to generate range-doppler map with profiles.
Exact implementation from test-dagex with multi-channel support.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dagex import Graph
from crsd_inspector.workflows.workflow import Workflow


# Create workflow instance
workflow = Workflow(
    name="Range-Doppler Processing",
    description="Matched filtering and Doppler FFT to generate range-doppler maps"
)

# Workflow parameters
PARAMS = {
    'range-window-type': {
        'label': 'Range Windowing',
        'type': 'dropdown',
        'default': 'none',
        'options': [
            {"label": "None", "value": "none"},
            {"label": "Hamming", "value": "hamming"},
            {"label": "Hann", "value": "hann"},
            {"label": "Blackman", "value": "blackman"},
            {"label": "Kaiser", "value": "kaiser"}
        ],
    },
    'doppler-window-type': {
        'label': 'Doppler Windowing',
        'type': 'dropdown',
        'default': 'none',
        'options': [
            {"label": "None", "value": "none"},
            {"label": "Hamming", "value": "hamming"},
            {"label": "Hann", "value": "hann"},
            {"label": "Blackman", "value": "blackman"},
            {"label": "Kaiser", "value": "kaiser"}
        ],
    },
}


def perform_range_doppler_processing(signal_data, tx_wfm, sample_rate_hz, prf_hz, range_window='none', doppler_window='none'):
    """
    Perform range-Doppler processing on received signal using frequency domain matched filtering
    
    Args:
        signal_data: Input signal (num_pulses x num_samples)
        tx_wfm: Transmit waveform for matched filtering
        sample_rate_hz: Sample rate in Hz
        prf_hz: Pulse repetition frequency in Hz
        range_window: Window type for range dimension ('none', 'hamming', 'hann', 'blackman', 'kaiser')
        doppler_window: Window type for Doppler dimension ('none', 'hamming', 'hann', 'blackman', 'kaiser')
    """
    num_pulses, num_samples = signal_data.shape
    
    # Step 1: Pulse compression via frequency domain matched filtering
    # Prepare matched filter in frequency domain
    matched_filter_time = np.conj(tx_wfm[::-1])
    matched_filter_freq = np.fft.fft(matched_filter_time, n=num_samples)
    
    # Apply range windowing to matched filter in frequency domain
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
        
        # Apply window to matched filter frequency response
        matched_filter_freq = matched_filter_freq * window
    
    # FFT along fast-time (range) dimension for all pulses at once
    signal_freq = np.fft.fft(signal_data, axis=1)
    
    # Multiply in frequency domain (broadcasting matched filter across pulses)
    compressed_freq = signal_freq * matched_filter_freq[None, :]
    
    # Transform back to time domain
    compressed = np.fft.ifft(compressed_freq, axis=1)
    
    # Save compressed signal before Doppler windowing for visualization
    compressed_for_display = compressed.copy()
    
    # Apply Doppler windowing if requested (only for Doppler processing)
    if doppler_window != 'none':
        if doppler_window == 'hamming':
            window = np.hamming(num_pulses)
        elif doppler_window == 'hann':
            window = np.hanning(num_pulses)
        elif doppler_window == 'blackman':
            window = np.blackman(num_pulses)
        elif doppler_window == 'kaiser':
            window = np.kaiser(num_pulses, beta=8.6)
        else:
            window = np.ones(num_pulses)
        
        compressed = compressed * window[:, None]
    
    # Step 2: Doppler processing (FFT across slow time)
    range_doppler = np.fft.fftshift(np.fft.fft(compressed, axis=0), axes=0)
    
    rd_mag = np.abs(range_doppler)
    rd_db = 20 * np.log10(rd_mag + 1e-10)
    
    # Step 3: Extract profiles
    range_profile = np.mean(rd_mag, axis=0)
    range_profile_db = 20 * np.log10(range_profile + 1e-10)
    
    doppler_profile = np.mean(rd_mag, axis=1)
    doppler_profile_db = 20 * np.log10(doppler_profile + 1e-10)
    
    # Step 4: Compute statistics
    peak_val = np.max(rd_db)
    noise_floor = np.percentile(rd_db, 5)
    dynamic_range = peak_val - noise_floor
    mean_val = np.mean(rd_db)
    std_val = np.std(rd_db)
    
    peak_idx = np.unravel_index(np.argmax(rd_mag), rd_mag.shape)
    peak_doppler_idx, peak_range_idx = peak_idx
    
    c = 3e8
    range_axis_m = (np.arange(num_samples) / sample_rate_hz) * c / 2
    doppler_axis_hz = np.fft.fftshift(np.fft.fftfreq(num_pulses, d=1/prf_hz))
    
    # Calculate resolutions
    range_resolution_m = c / (2 * sample_rate_hz)
    doppler_resolution_hz = prf_hz / num_pulses
    
    return {
        'range_doppler_db': rd_db,
        'range_profile_db': range_profile_db,
        'doppler_profile_db': doppler_profile_db,
        'range_axis_m': range_axis_m,
        'doppler_axis_hz': doppler_axis_hz,
        'compressed_signal': compressed_for_display,
        'peak_db': peak_val,
        'noise_floor_db': noise_floor,
        'dynamic_range_db': dynamic_range,
        'mean_db': mean_val,
        'std_db': std_val,
        'peak_range_m': range_axis_m[peak_range_idx],
        'peak_doppler_hz': doppler_axis_hz[peak_doppler_idx],
        'raw_peak_db': 20 * np.log10(np.max(np.abs(signal_data)) + 1e-10),
        'raw_mean_db': 20 * np.log10(np.mean(np.abs(signal_data)) + 1e-10),
        'raw_std_db': 20 * np.log10(np.std(np.abs(signal_data)) + 1e-10),
        'range_resolution_m': range_resolution_m,
        'doppler_resolution_hz': doppler_resolution_hz,
    }


def run_workflow(signal_data, metadata=None, **kwargs):
    """Execute range-doppler processing workflow using dagex Graph"""
    workflow.clear()  # Clear any previous results
    
    if metadata is None:
        metadata = {}
    
    tx_wfm = metadata.get('tx_wfm')
    if tx_wfm is None:
        workflow.add_text("No TX waveform found in CRSD file. Cannot perform matched filtering.")
        return workflow.build()
    
    # Get selected channel and all available channels
    selected_channel = metadata.get('selected_channel', None)
    all_channels = metadata.get('all_channels', {})
    
    # If we have multiple channels, use the selected one's data but keep channel info
    if isinstance(signal_data, dict):
        # signal_data is a dict of channel_id -> array
        if selected_channel and selected_channel in signal_data:
            channel_data = signal_data[selected_channel]
        else:
            # Use first channel if no selection
            selected_channel = list(signal_data.keys())[0]
            channel_data = signal_data[selected_channel]
        metadata['signal_data_all_channels'] = signal_data  # Store all for variants
    else:
        # signal_data is a single array
        channel_data = signal_data
        metadata['signal_data_all_channels'] = {selected_channel: signal_data}
    
    # Create and execute graph with variants
    graph, selected_channel, channel_variants = _create_graph(channel_data, metadata)
    
    try:
        dag = graph.build()
    except Exception as e:
        import traceback
        traceback.print_exc()
        workflow.add_text(f"Error building graph: {str(e)}")
        return workflow.build()
    
    # Execute all variants
    try:
        result = dag.execute(True, 4)
        
        # Check if it's a special object with attributes
        if hasattr(result, 'context'):
            context = result.context
        elif hasattr(result, 'results'):
            context = result.results
        elif hasattr(result, 'variants'):
            context = result.variants
        else:
            context = result
    except Exception as e:
        import traceback
        traceback.print_exc()
        workflow.add_text(f"Error executing graph: {str(e)}")
        return workflow.build()
    
    # Extract result for selected channel from variant results
    variant_idx = channel_variants.index(selected_channel) if selected_channel in channel_variants else 0
    
    # The result will be keyed by variant index in the context
    # dagex uses integer indices for variants
    variant_result = {}
    for key, value in context.items():
        if isinstance(key, tuple) and len(key) == 2:
            output_name, var_idx = key
            if var_idx == variant_idx:
                variant_result[output_name] = value
        elif not isinstance(key, tuple):
            # Non-variant outputs
            variant_result[key] = value
    
    # Format and return results
    _format_results(variant_result, metadata, selected_channel)
    return workflow.build()


def _create_graph(signal_data, metadata):
    """Create range-doppler processing workflow graph with modular steps and channel variants"""
    graph = Graph()
    
    tx_wfm = metadata.get('tx_wfm')
    sample_rate_hz = metadata.get('sample_rate_hz', 100e6)
    prf_hz = metadata.get('prf_hz', 1000.0)
    range_window = metadata.get('range_window', 'none')
    doppler_window = metadata.get('doppler_window', 'none')
    
    # Get all channel data for variants
    all_channel_data = metadata.get('signal_data_all_channels', {})
    if not all_channel_data:
        # Fallback to single channel
        selected_channel = metadata.get('selected_channel', 'default')
        all_channel_data = {selected_channel: signal_data}
    
    # Declare variants at graph level
    channel_variants = list(all_channel_data.keys())
    
    def apply_range_window(inputs):
        """Apply windowing to matched filter for range sidelobe control"""
        tx_wfm = inputs.get("tx_wfm")
        signal_data = inputs.get("signal_data")
        range_win_type = inputs.get("range_window_type", "none")
        
        num_samples_tx = len(tx_wfm)
        num_samples_signal = signal_data.shape[1]  # Get signal length for FFT
        
        # Prepare matched filter in frequency domain with signal length
        matched_filter_time = np.conj(tx_wfm[::-1])
        matched_filter_freq = np.fft.fft(matched_filter_time, n=num_samples_signal)
        
        # Apply window to matched filter if requested
        if range_win_type != 'none':
            if range_win_type == 'hamming':
                window = np.hamming(num_samples_tx)
            elif range_win_type == 'hann':
                window = np.hanning(num_samples_tx)
            elif range_win_type == 'blackman':
                window = np.blackman(num_samples_tx)
            elif range_win_type == 'kaiser':
                window = np.kaiser(num_samples_tx, beta=8.6)
            else:
                window = np.ones(num_samples_tx)
            
            # Apply window to time-domain matched filter then re-FFT
            windowed_mf_time = matched_filter_time * window
            matched_filter_freq = np.fft.fft(windowed_mf_time, n=num_samples_signal)
        
        return {"matched_filter_freq": matched_filter_freq}
    
    def range_compression(inputs):
        """Perform pulse compression via frequency domain matched filtering"""
        sig_data = inputs.get("signal_data")
        matched_filter_freq = inputs.get("matched_filter_freq")
        
        # FFT along fast-time (range) dimension for all pulses at once
        signal_freq = np.fft.fft(sig_data, axis=1)
        
        # Multiply in frequency domain (broadcasting matched filter across pulses)
        compressed_freq = signal_freq * matched_filter_freq[None, :]
        
        # Transform back to time domain
        compressed = np.fft.ifft(compressed_freq, axis=1)
        
        # Save for visualization before Doppler windowing
        compressed_for_display = compressed.copy()
        
        return {
            "compressed_signal": compressed,
            "compressed_for_display": compressed_for_display
        }
    
    def apply_doppler_window(inputs):
        """Apply windowing for Doppler sidelobe control"""
        compressed = inputs.get("compressed_signal")
        doppler_win_type = inputs.get("doppler_window_type", "none")
        num_pulses = compressed.shape[0]
        
        # Apply Doppler windowing if requested
        if doppler_win_type != 'none':
            if doppler_win_type == 'hamming':
                window = np.hamming(num_pulses)
            elif doppler_win_type == 'hann':
                window = np.hanning(num_pulses)
            elif doppler_win_type == 'blackman':
                window = np.blackman(num_pulses)
            elif doppler_win_type == 'kaiser':
                window = np.kaiser(num_pulses, beta=8.6)
            else:
                window = np.ones(num_pulses)
            
            compressed = compressed * window[:, None]
        
        return {"windowed_compressed": compressed}
    
    def doppler_compression(inputs):
        """Perform Doppler processing via FFT across slow time"""
        compressed = inputs.get("windowed_compressed")
        
        # FFT across slow time (pulse dimension)
        range_doppler = np.fft.fftshift(np.fft.fft(compressed, axis=0), axes=0)
        
        return {"range_doppler": range_doppler}
    
    def compute_profiles_and_stats(inputs):
        """Compute profiles, statistics, and axes"""
        range_doppler = inputs.get("range_doppler")
        compressed_for_display = inputs.get("compressed_for_display")
        sample_rate = inputs.get("sample_rate_hz")
        prf = inputs.get("prf_hz")
        sig_data = inputs.get("signal_data")
        
        num_pulses, num_samples = sig_data.shape
        
        # Compute magnitude and dB
        rd_mag = np.abs(range_doppler)
        rd_db = 20 * np.log10(rd_mag + 1e-10)
        
        # Extract profiles
        range_profile = np.mean(rd_mag, axis=0)
        range_profile_db = 20 * np.log10(range_profile + 1e-10)
        
        doppler_profile = np.mean(rd_mag, axis=1)
        doppler_profile_db = 20 * np.log10(doppler_profile + 1e-10)
        
        # Compute statistics
        peak_val = np.max(rd_db)
        noise_floor = np.percentile(rd_db, 5)
        dynamic_range = peak_val - noise_floor
        mean_val = np.mean(rd_db)
        std_val = np.std(rd_db)
        
        peak_idx = np.unravel_index(np.argmax(rd_mag), rd_mag.shape)
        peak_doppler_idx, peak_range_idx = peak_idx
        
        # Compute axes
        c = 3e8
        range_axis_m = (np.arange(num_samples) / sample_rate) * c / 2
        doppler_axis_hz = np.fft.fftshift(np.fft.fftfreq(num_pulses, d=1/prf))
        
        # Calculate resolutions
        range_resolution_m = c / (2 * sample_rate)
        doppler_resolution_hz = prf / num_pulses
        
        return {
            'rd_db': rd_db,
            'rd_mag': rd_mag,
            'range_profile_db': range_profile_db,
            'doppler_profile_db': doppler_profile_db,
            'range_axis_m': range_axis_m,
            'doppler_axis_hz': doppler_axis_hz,
            'compressed_for_display': compressed_for_display,
            'peak_db': peak_val,
            'noise_floor_db': noise_floor,
            'dynamic_range_db': dynamic_range,
            'mean_db': mean_val,
            'std_db': std_val,
            'peak_range_m': range_axis_m[peak_range_idx],
            'peak_doppler_hz': doppler_axis_hz[peak_doppler_idx],
            'raw_peak_db': 20 * np.log10(np.max(np.abs(sig_data)) + 1e-10),
            'raw_mean_db': 20 * np.log10(np.mean(np.abs(sig_data)) + 1e-10),
            'raw_std_db': 20 * np.log10(np.std(np.abs(sig_data)) + 1e-10),
            'range_resolution_m': range_resolution_m,
            'doppler_resolution_hz': doppler_resolution_hz,
        }
    
    def generate_plots(inputs):
        """Generate all plots as part of the graph (for memoization)"""
        stats = inputs
        
        # Return all the data needed for plotting
        return {"result": stats}
    
    # Get all channel IDs for variants
    channel_variants = list(all_channel_data.keys())
    selected_channel = metadata.get('selected_channel', channel_variants[0] if channel_variants else 'default')
    
    # Build the graph with modular nodes
    # Use variants to provide data for each channel
    graph.variants(
        [lambda inputs, ch_id=ch_id: {
            "signal_data": all_channel_data[ch_id],
            "tx_wfm": tx_wfm,
            "sample_rate_hz": sample_rate_hz,
            "prf_hz": prf_hz,
            "range_window_type": range_window,
            "doppler_window_type": doppler_window
        } for ch_id in channel_variants],
        "Provide Data",
        [],
        [
            ("signal_data", "signal_data"),
            ("tx_wfm", "tx_wfm"),
            ("sample_rate_hz", "sample_rate_hz"),
            ("prf_hz", "prf_hz"),
            ("range_window_type", "range_window_type"),
            ("doppler_window_type", "doppler_window_type")
        ]
    )
    
    # Range window node: Needs signal_data to know the FFT length
    graph.add(
        apply_range_window,
        label="Apply Range Window",
        inputs=[
            ("signal_data", "signal_data"),
            ("tx_wfm", "tx_wfm"),
            ("range_window_type", "range_window_type")
        ],
        outputs=[("matched_filter_freq", "matched_filter_freq")]
    )
    
    # Range compression: Uses regular function (variant handled by graph)
    graph.add(
        range_compression,
        label="Range Compression",
        inputs=[
            ("signal_data", "signal_data"),
            ("matched_filter_freq", "matched_filter_freq")
        ],
        outputs=[
            ("compressed_signal", "compressed_signal"),
            ("compressed_for_display", "compressed_for_display")
        ]
    )
    
    # Doppler window
    graph.add(
        apply_doppler_window,
        label="Apply Doppler Window",
        inputs=[
            ("compressed_signal", "compressed_signal"),
            ("doppler_window_type", "doppler_window_type")
        ],
        outputs=[("windowed_compressed", "windowed_compressed")]
    )
    
    # Doppler compression
    graph.add(
        doppler_compression,
        label="Doppler Compression",
        inputs=[("windowed_compressed", "windowed_compressed")],
        outputs=[("range_doppler", "range_doppler")]
    )
    
    # Final statistics and plot generation
    graph.add(
        compute_profiles_and_stats,
        label="Compute Profiles and Statistics",
        inputs=[
            ("range_doppler", "range_doppler"),
            ("compressed_for_display", "compressed_for_display"),
            ("sample_rate_hz", "sample_rate_hz"),
            ("prf_hz", "prf_hz"),
            ("signal_data", "signal_data")
        ],
        outputs=[
            ("rd_db", "rd_db"),
            ("rd_mag", "rd_mag"),
            ("range_profile_db", "range_profile_db"),
            ("doppler_profile_db", "doppler_profile_db"),
            ("range_axis_m", "range_axis_m"),
            ("doppler_axis_hz", "doppler_axis_hz"),
            ("compressed_for_display", "compressed_for_display_out"),
            ("peak_db", "peak_db"),
            ("noise_floor_db", "noise_floor_db"),
            ("dynamic_range_db", "dynamic_range_db"),
            ("mean_db", "mean_db"),
            ("std_db", "std_db"),
            ("peak_range_m", "peak_range_m"),
            ("peak_doppler_hz", "peak_doppler_hz"),
            ("raw_peak_db", "raw_peak_db"),
            ("raw_mean_db", "raw_mean_db"),
            ("raw_std_db", "raw_std_db"),
            ("range_resolution_m", "range_resolution_m"),
            ("doppler_resolution_hz", "doppler_resolution_hz")
        ]
    )
    
    # Generate plots (memoized in graph)
    graph.add(
        generate_plots,
        label="Generate Plots",
        inputs=[
            ("rd_db", "rd_db"),
            ("rd_mag", "rd_mag"),
            ("range_profile_db", "range_profile_db"),
            ("doppler_profile_db", "doppler_profile_db"),
            ("range_axis_m", "range_axis_m"),
            ("doppler_axis_hz", "doppler_axis_hz"),
            ("compressed_for_display_out", "compressed_for_display"),
            ("peak_db", "peak_db"),
            ("noise_floor_db", "noise_floor_db"),
            ("dynamic_range_db", "dynamic_range_db"),
            ("mean_db", "mean_db"),
            ("std_db", "std_db"),
            ("peak_range_m", "peak_range_m"),
            ("peak_doppler_hz", "peak_doppler_hz"),
            ("raw_peak_db", "raw_peak_db"),
            ("raw_mean_db", "raw_mean_db"),
            ("raw_std_db", "raw_std_db"),
            ("range_resolution_m", "range_resolution_m"),
            ("doppler_resolution_hz", "doppler_resolution_hz")
        ],
        outputs=[("result", "result")]
    )
    
    return graph, selected_channel, channel_variants


def _format_results(context, metadata, selected_channel=None):
    """Format workflow results from graph execution context"""
    if selected_channel is None:
        selected_channel = metadata.get('selected_channel', 'Unknown')
    
    result = context.get("result")
    
    if not result:
        return workflow.build()
    
    # Summary text
    workflow.add_text(
        f"Range-Doppler processing complete for {selected_channel}\n"
        f"Dynamic Range: {result['dynamic_range_db']:.1f} dB\n"
        f"Peak: {result['peak_db']:.1f} dB | Noise Floor: {result['noise_floor_db']:.1f} dB"
    )
    
    # Plot 1: Stacked Range Profiles (Range-Time Image)
    compressed_mag = np.abs(result['compressed_for_display'])
    compressed_db = 20 * np.log10(compressed_mag + 1e-10)
    
    # Calculate default colormap range
    compressed_peak = np.max(compressed_db)
    compressed_min = compressed_peak - 60
    
    fig_compressed = go.Figure(data=go.Heatmap(
        x=result['range_axis_m'] / 1000,
        y=np.arange(compressed_mag.shape[0]),
        z=compressed_db,
        colorscale='Jet',
        zmin=compressed_min,
        zmax=compressed_peak,
        colorbar=dict(
            title='Magnitude (dB)',
            x=1.15
        )
    ))
    
    # Create interactive colormap sliders for compressed range profiles
    compressed_steps_min = []
    compressed_steps_max = []
    compressed_vals_min = np.linspace(np.min(compressed_db), compressed_peak - 10, 20)
    compressed_vals_max = np.linspace(compressed_peak - 70, compressed_peak, 20)
    
    for val in compressed_vals_min:
        compressed_steps_min.append(dict(
            method="restyle",
            args=[{"zmin": val}],
            label=f"{val:.0f}"
        ))
    
    for val in compressed_vals_max:
        compressed_steps_max.append(dict(
            method="restyle",
            args=[{"zmax": val}],
            label=f"{val:.0f}"
        ))
    
    fig_compressed.update_layout(
        title=f'Range Profiles Across Pulses (After Compression) - {selected_channel}',
        xaxis_title='Range (km)',
        yaxis_title='Pulse Number',
        height=600,
        template='plotly_dark',
        sliders=[
            dict(
                active=10,
                yanchor="top",
                y=-0.12,
                xanchor="left",
                currentvalue=dict(
                    prefix="Min: ",
                    visible=True,
                    xanchor="right"
                ),
                pad=dict(b=10, t=10),
                len=0.42,
                x=0.0,
                steps=compressed_steps_min
            ),
            dict(
                active=10,
                yanchor="top",
                y=-0.12,
                xanchor="right",
                currentvalue=dict(
                    prefix="Max: ",
                    visible=True,
                    xanchor="left"
                ),
                pad=dict(b=10, t=10),
                len=0.42,
                x=1.0,
                steps=compressed_steps_max
            )
        ]
    )
    workflow.add_plot(fig_compressed)
    
    # Plot 2: Range-Doppler Map with colormap sliders
    z_min_default = result['peak_db'] - 60
    z_max_default = result['peak_db']
    
    # Calculate magnitude for finding peak doppler
    rd_db_for_plot = result['rd_db']
    rd_mag = result['rd_mag']
    
    fig_rd = go.Figure(data=go.Heatmap(
        x=result['range_axis_m'] / 1000,
        y=result['doppler_axis_hz'],
        z=result['rd_db'],
        colorscale='Jet',
        zmin=z_min_default,
        zmax=z_max_default,
        colorbar=dict(
            title='Magnitude (dB)',
            x=1.15
        )
    ))
    
    # Create interactive colormap sliders
    rd_db_data = result['rd_db']
    steps_min = []
    steps_max = []
    range_vals_min = np.linspace(np.min(rd_db_data), result['peak_db'] - 10, 20)
    range_vals_max = np.linspace(result['peak_db'] - 70, result['peak_db'], 20)
    
    for val in range_vals_min:
        steps_min.append(dict(
            method="restyle",
            args=[{"zmin": val}],
            label=f"{val:.0f}"
        ))
    
    for val in range_vals_max:
        steps_max.append(dict(
            method="restyle",
            args=[{"zmax": val}],
            label=f"{val:.0f}"
        ))
    
    fig_rd.update_layout(
        title=f'Range-Doppler Map - {selected_channel}',
        xaxis_title='Range (km)',
        yaxis_title='Doppler (Hz)',
        height=700,
        width=1000,
        template='plotly_dark',
        sliders=[
            dict(
                active=10,
                yanchor="top",
                y=-0.12,
                xanchor="left",
                currentvalue=dict(
                    prefix="Min: ",
                    visible=True,
                    xanchor="right"
                ),
                pad=dict(b=10, t=10),
                len=0.42,
                x=0.0,
                steps=steps_min
            ),
            dict(
                active=10,
                yanchor="top",
                y=-0.12,
                xanchor="right",
                currentvalue=dict(
                    prefix="Max: ",
                    visible=True,
                    xanchor="left"
                ),
                pad=dict(b=10, t=10),
                len=0.42,
                x=1.0,
                steps=steps_max
            )
        ]
    )
    workflow.add_plot(fig_rd)
    
    # Plot 3: Range Profiles at Different Doppler Frequencies
    fig_range_cuts = go.Figure()
    
    # Find indices for different doppler frequencies
    zero_doppler_idx = np.argmin(np.abs(result['doppler_axis_hz']))
    peak_doppler_idx = np.argmax(np.mean(rd_mag, axis=1))
    
    # Add range profile at zero Doppler
    fig_range_cuts.add_trace(go.Scatter(
        x=result['range_axis_m'] / 1000,
        y=result['rd_db'][zero_doppler_idx, :],
        mode='lines',
        line=dict(color='cyan', width=2),
        name=f'Zero Doppler ({result["doppler_axis_hz"][zero_doppler_idx]:.1f} Hz)'
    ))
    
    # Add range profile at peak Doppler
    fig_range_cuts.add_trace(go.Scatter(
        x=result['range_axis_m'] / 1000,
        y=result['rd_db'][peak_doppler_idx, :],
        mode='lines',
        line=dict(color='magenta', width=2),
        name=f'Peak Doppler ({result["doppler_axis_hz"][peak_doppler_idx]:.1f} Hz)'
    ))
    
    # Add averaged range profile
    fig_range_cuts.add_trace(go.Scatter(
        x=result['range_axis_m'] / 1000,
        y=result['range_profile_db'],
        mode='lines',
        line=dict(color='yellow', width=2, dash='dash'),
        name='Average (all Doppler)'
    ))
    
    fig_range_cuts.update_layout(
        title=f'Range Profiles at Different Doppler Frequencies - {selected_channel}',
        xaxis_title='Range (km)',
        yaxis_title='Magnitude (dB)',
        height=450,
        hovermode='x',
        template='plotly_dark',
        showlegend=True
    )
    workflow.add_plot(fig_range_cuts)
    
    # Plot 3: Range Profile (averaged)
    fig_range = go.Figure()
    fig_range.add_trace(go.Scatter(
        x=result['range_axis_m'] / 1000,
        y=result['range_profile_db'],
        mode='lines',
        line=dict(color='blue', width=2),
    ))
    
    fig_range.update_layout(
        title=f'Range Profile - {selected_channel}',
        xaxis_title='Range (km)',
        yaxis_title='Magnitude (dB)',
        height=450,
        hovermode='x',
        template='plotly_dark'
    )
    workflow.add_plot(fig_range)
    
    # Plot 4: Doppler Profile
    fig_doppler = go.Figure()
    fig_doppler.add_trace(go.Scatter(
        x=result['doppler_axis_hz'],
        y=result['doppler_profile_db'],
        mode='lines',
        line=dict(color='red', width=2),
    ))
    
    fig_doppler.update_layout(
        title=f'Doppler Profile - {selected_channel}',
        xaxis_title='Doppler (Hz)',
        yaxis_title='Magnitude (dB)',
        height=450,
        hovermode='x',
        template='plotly_dark'
    )
    workflow.add_plot(fig_doppler)
    
    # Table: Summary Statistics
    stats_table = {
        "Metric": [
            "Dynamic Range",
            "Peak Value",
            "Noise Floor",
            "Range Resolution",
            "Doppler Resolution"
        ],
        "Value": [
            f"{result['dynamic_range_db']:.2f} dB",
            f"{result['peak_db']:.2f} dB",
            f"{result['noise_floor_db']:.2f} dB",
            f"{result['range_resolution_m']:.3f} m",
            f"{result['doppler_resolution_hz']:.3f} Hz"
        ]
    }
    workflow.add_table("Processing Statistics", stats_table)
    
    return workflow.build()
