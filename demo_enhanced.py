#!/usr/bin/env python3
"""
Demo script for enhanced CRSD Inspector with Plotly visualizations
Shows the dagex workflow and generates sample diagnostic plots
"""

import numpy as np
from dagex import Graph
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os


def generate_synthetic_signal(rows, cols):
    """Generate synthetic CRSD signal data"""
    signal = np.zeros((rows, cols), dtype=np.complex64)
    
    # Add synthetic targets
    for i in range(3):
        target_row = 80 + i * 50
        target_col = 100 + i * 40
        for dr in range(-5, 6):
            for dc in range(-5, 6):
                r = target_row + dr
                c = target_col + dc
                if 0 <= r < rows and 0 <= c < cols:
                    amp = 10.0 * np.exp(-0.1 * (dr**2 + dc**2))
                    phase = np.random.uniform(-np.pi, np.pi)
                    signal[r, c] += amp * np.exp(1j * phase)
    
    # Add noise
    noise = 0.1 * (np.random.randn(rows, cols) + 1j * np.random.randn(rows, cols))
    signal += noise.astype(np.complex64)
    
    return signal


def load_data(_inputs):
    """Generate synthetic data"""
    signal = generate_synthetic_signal(256, 256)
    return {
        "signal_data": signal,
        "shape": signal.shape,
        "data_source": "synthetic"
    }


def compute_statistics(inputs):
    """Compute comprehensive statistics"""
    signal_data = inputs.get("signal_data")
    amplitude = np.abs(signal_data)
    phase = np.angle(signal_data)
    
    stats = {
        "amplitude": {
            "min": float(np.min(amplitude)),
            "max": float(np.max(amplitude)),
            "mean": float(np.mean(amplitude)),
            "std": float(np.std(amplitude)),
            "median": float(np.median(amplitude)),
        },
        "phase": {
            "min": float(np.min(phase)),
            "max": float(np.max(phase)),
            "mean": float(np.mean(phase)),
            "std": float(np.std(phase)),
        },
        "quality": {
            "snr_estimate_db": 20 * np.log10(np.mean(amplitude) / (np.std(amplitude) + 1e-10)),
            "dynamic_range_db": 20 * np.log10(np.max(amplitude) / (np.min(amplitude) + 1e-10)),
        }
    }
    
    return {
        "statistics": stats,
        "amplitude": amplitude,
        "phase": phase
    }


def create_demo_workflow():
    """Create enhanced dagex workflow"""
    graph = Graph()
    
    graph.add(
        load_data,
        label="Generate Data",
        inputs=None,
        outputs=[("signal_data", "signal"), ("shape", "shape"), ("data_source", "source")]
    )
    
    graph.add(
        compute_statistics,
        label="Compute Statistics",
        inputs=[("signal", "signal_data")],
        outputs=[("statistics", "stats"), ("amplitude", "amp"), ("phase", "phase")]
    )
    
    return graph


def main():
    print("=" * 70)
    print("CRSD Inspector - Enhanced Demo with Plotly Visualizations")
    print("=" * 70)
    
    # Build workflow
    print("\n1. Building enhanced dagex workflow...")
    graph = create_demo_workflow()
    dag = graph.build()
    print("   ✓ Workflow built")
    
    # Show Mermaid diagram
    try:
        print("\n2. Workflow structure:")
        print("   " + "-" * 66)
        mermaid = dag.to_mermaid()
        for line in mermaid.split('\n'):
            print("   " + line)
        print("   " + "-" * 66)
    except Exception as e:
        print(f"   Could not generate diagram: {e}")
    
    # Execute workflow
    print("\n3. Executing workflow (parallel)...")
    context = dag.execute(True, 4)
    print("   ✓ Execution complete")
    
    # Display results
    print("\n4. Comprehensive Diagnostics:")
    print("   " + "-" * 66)
    
    stats = context.get("stats")
    if stats:
        print("\n   📊 Amplitude Statistics:")
        for key, val in stats['amplitude'].items():
            print(f"      {key:12s}: {val:.4f}")
        
        print("\n   🌊 Phase Statistics:")
        for key, val in stats['phase'].items():
            print(f"      {key:12s}: {val:.4f}")
        
        print("\n   ⚡ Quality Metrics:")
        for key, val in stats['quality'].items():
            print(f"      {key:20s}: {val:.2f}")
    
    print("\n   " + "-" * 66)
    
    # Generate Plotly visualizations
    print("\n5. Generating Plotly visualizations...")
    
    amp = context.get("amp")
    phase = context.get("phase")
    
    if amp is not None and phase is not None:
        # Create amplitude plot
        amp_db = 20 * np.log10(amp + 1e-10)
        fig_amp = go.Figure(data=go.Heatmap(
            z=amp_db,
            colorscale='Gray',
            colorbar=dict(title="Amplitude (dB)")
        ))
        fig_amp.update_layout(
            title="Signal Amplitude (dB) - Interactive Plotly",
            xaxis_title="Range Sample",
            yaxis_title="Azimuth Vector"
        )
        
        # Create phase plot
        fig_phase = go.Figure(data=go.Heatmap(
            z=phase,
            colorscale='HSV',
            colorbar=dict(title="Phase (rad)"),
            zmid=0
        ))
        fig_phase.update_layout(
            title="Signal Phase - Interactive Plotly",
            xaxis_title="Range Sample",
            yaxis_title="Azimuth Vector"
        )
        
        # Create histograms
        fig_hist = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Amplitude Distribution", "Phase Distribution")
        )
        fig_hist.add_trace(
            go.Histogram(x=amp.flatten(), nbinsx=50, name="Amplitude"),
            row=1, col=1
        )
        fig_hist.add_trace(
            go.Histogram(x=phase.flatten(), nbinsx=50, name="Phase"),
            row=1, col=2
        )
        fig_hist.update_layout(showlegend=False)
        
        # Create profiles
        mid_row = amp.shape[0] // 2
        mid_col = amp.shape[1] // 2
        
        fig_profiles = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Azimuth Profile", "Range Profile")
        )
        fig_profiles.add_trace(
            go.Scatter(y=20 * np.log10(amp[:, mid_col] + 1e-10), mode='lines'),
            row=1, col=1
        )
        fig_profiles.add_trace(
            go.Scatter(y=20 * np.log10(amp[mid_row, :] + 1e-10), mode='lines'),
            row=1, col=2
        )
        fig_profiles.update_yaxes(title_text="Amplitude (dB)", row=1, col=1)
        fig_profiles.update_yaxes(title_text="Amplitude (dB)", row=1, col=2)
        fig_profiles.update_layout(showlegend=False)
        
        print("   ✓ Generated 4 interactive Plotly visualizations:")
        print("      - Amplitude heatmap (dB scale)")
        print("      - Phase heatmap (HSV colormap)")
        print("      - Distribution histograms")
        print("      - Azimuth and range profiles")
        
        # Save plots
        try:
            temp_dir = tempfile.gettempdir()
            fig_amp.write_html(os.path.join(temp_dir, "amplitude_plot.html"))
            fig_phase.write_html(os.path.join(temp_dir, "phase_plot.html"))
            fig_hist.write_html(os.path.join(temp_dir, "histograms.html"))
            fig_profiles.write_html(os.path.join(temp_dir, "profiles.html"))
            print(f"\n   💾 Saved interactive plots to {temp_dir}/*.html")
            print("      Open these files in a browser for full interactivity")
        except Exception as e:
            print(f"\n   Note: Could not save plots: {e}")
    
    print("\n" + "=" * 70)
    print("✅ Enhanced demo completed successfully!")
    print("\nTo see the full interactive application, run:")
    print("  streamlit run app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
