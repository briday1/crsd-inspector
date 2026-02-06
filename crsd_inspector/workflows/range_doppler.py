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


def perform_range_doppler_processing(signal_data, tx_wfm, sample_rate_hz, prf_hz):
    """
    Perform range-Doppler processing on received signal using frequency domain matched filtering
    """
    num_pulses, num_samples = signal_data.shape
    
    # Step 1: Pulse compression via frequency domain matched filtering
    compressed = np.zeros_like(signal_data)
    
    # Prepare matched filter in frequency domain
    matched_filter_time = np.conj(tx_wfm[::-1])
    matched_filter_freq = np.fft.fft(matched_filter_time, n=num_samples)
    
    for p in range(num_pulses):
        # Transform to frequency domain
        signal_freq = np.fft.fft(signal_data[p, :])
        
        # Multiply in frequency domain (equivalent to convolution in time domain)
        compressed_freq = signal_freq * matched_filter_freq
        
        # Transform back to time domain
        compressed[p, :] = np.fft.ifft(compressed_freq)
    
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
    
    return {
        'range_doppler_db': rd_db,
        'range_profile_db': range_profile_db,
        'doppler_profile_db': doppler_profile_db,
        'range_axis_m': range_axis_m,
        'doppler_axis_hz': doppler_axis_hz,
        'compressed_signal': compressed,
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
    
    # Create and execute graph
    graph = _create_graph(signal_data, metadata)
    dag = graph.build()
    context = dag.execute(True, 4)
    
    # Format and return results
    _format_results(context, metadata)
    return workflow.build()
    
    # Format and return results
    return _format_results(context, metadata)


def _create_graph(signal_data, metadata):
    """Create range-doppler processing workflow graph"""
    graph = Graph()
    
    tx_wfm = metadata.get('tx_wfm')
    sample_rate_hz = metadata.get('sample_rate_hz', 100e6)
    prf_hz = metadata.get('prf_hz', 1000.0)
    
    def provide_data(_inputs):
        return {
            "signal_data": signal_data,
            "tx_wfm": tx_wfm,
            "sample_rate_hz": sample_rate_hz,
            "prf_hz": prf_hz
        }
    
    def process_channel(inputs):
        """Process single channel with range-doppler"""
        sig_data = inputs.get("signal_data")
        tx_wfm = inputs.get("tx_wfm")
        sample_rate = inputs.get("sample_rate_hz")
        prf = inputs.get("prf_hz")
        
        result = perform_range_doppler_processing(
            sig_data, tx_wfm, sample_rate, prf
        )
        
        return {"result": result}
    
    graph.add(
        provide_data,
        label="Provide Data",
        inputs=[],
        outputs=[
            ("signal_data", "signal_data"),
            ("tx_wfm", "tx_wfm"),
            ("sample_rate_hz", "sample_rate_hz"),
            ("prf_hz", "prf_hz")
        ]
    )
    
    graph.add(
        process_channel,
        label="Range-Doppler Processing",
        inputs=[
            ("signal_data", "signal_data"),
            ("tx_wfm", "tx_wfm"),
            ("sample_rate_hz", "sample_rate_hz"),
            ("prf_hz", "prf_hz")
        ],
        outputs=[("result", "result")]
    )
    
    return graph


def _format_results(context, metadata):
    """Format workflow results from graph execution context"""
    result = context.get("result")
    selected_channel = metadata.get('selected_channel', 'Unknown')
    
    if not result:
        return workflow.build()
    
    # Summary text
    workflow.add_text(
        f"Range-Doppler processing complete for {selected_channel}\n"
        f"Dynamic Range: {result['dynamic_range_db']:.1f} dB\n"
        f"Peak: {result['peak_db']:.1f} dB | Noise Floor: {result['noise_floor_db']:.1f} dB"
    )
    
    # Plot 1: Range-Doppler Map with colormap sliders
    z_min_default = result['peak_db'] - 60
    z_max_default = result['peak_db']
    
    fig_rd = go.Figure(data=go.Heatmap(
        x=result['range_axis_m'] / 1000,
        y=result['doppler_axis_hz'],
        z=result['range_doppler_db'],
        colorscale='Jet',
        zmin=z_min_default,
        zmax=z_max_default,
        colorbar=dict(
            title='Magnitude (dB)',
            x=1.15
        )
    ))
    
    # Create interactive colormap sliders
    rd_db_data = result['range_doppler_db']
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
    
    # Plot 2: Range Profile
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
    )
    workflow.add_plot(fig_range)
    
    # Plot 3: Doppler Profile
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
