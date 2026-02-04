#!/usr/bin/env python3
"""
Demo script showing the dagex workflow for CRSD processing
This demonstrates the workflow without requiring a real CRSD file
"""

import numpy as np
from dagex import Graph


def generate_mock_crsd_data(_inputs):
    """Generate mock CRSD data for demo purposes"""
    # Create synthetic complex signal data (simulating SAR signal)
    rows, cols = 256, 256
    signal = np.random.randn(rows, cols) + 1j * np.random.randn(rows, cols)
    
    # Add some structure (simulating a target)
    signal[100:150, 100:150] *= 10
    
    return {
        "crsd_object": {
            "signal": signal,
            "metadata": "Mock CRSD File\nChannels: 1\nSize: 256x256"
        },
        "file_path": "mock.crsd"
    }


def extract_metadata(inputs):
    """Extract metadata from CRSD object"""
    crsd_obj = inputs.get("crsd_object", {})
    metadata = crsd_obj.get("metadata", "No metadata")
    
    return {
        "metadata": metadata,
        "crsd_object": crsd_obj
    }


def read_signal_data(inputs):
    """Read signal data from CRSD object"""
    crsd_obj = inputs.get("crsd_object", {})
    signal = crsd_obj.get("signal")
    
    if signal is not None:
        return {
            "signal_data": signal,
            "shape": str(signal.shape)
        }
    return {
        "signal_data": None,
        "shape": "No signal data"
    }


def compute_amplitude(inputs):
    """Compute amplitude from complex signal"""
    signal = inputs.get("signal_data")
    if signal is not None:
        amplitude = np.abs(signal)
        return {"amplitude": amplitude}
    return {"amplitude": None}


def compute_phase(inputs):
    """Compute phase from complex signal"""
    signal = inputs.get("signal_data")
    if signal is not None:
        phase = np.angle(signal)
        return {"phase": phase}
    return {"phase": None}


def create_demo_workflow():
    """Create the CRSD processing workflow"""
    graph = Graph()
    
    # Node 0: Generate mock CRSD data
    graph.add(
        generate_mock_crsd_data,
        label="Load CRSD",
        inputs=None,
        outputs=[("crsd_object", "crsd"), ("file_path", "path")]
    )
    
    # Node 1: Extract metadata
    graph.add(
        extract_metadata,
        label="Extract Metadata",
        inputs=[("crsd", "crsd_object")],
        outputs=[("metadata", "meta"), ("crsd_object", "crsd_pass")]
    )
    
    # Node 2: Read signal data
    graph.add(
        read_signal_data,
        label="Read Signal",
        inputs=[("crsd_pass", "crsd_object")],
        outputs=[("signal_data", "signal"), ("shape", "signal_shape")]
    )
    
    # Node 3: Compute amplitude
    graph.add(
        compute_amplitude,
        label="Compute Amplitude",
        inputs=[("signal", "signal_data")],
        outputs=[("amplitude", "amp")]
    )
    
    # Node 4: Compute phase
    graph.add(
        compute_phase,
        label="Compute Phase",
        inputs=[("signal", "signal_data")],
        outputs=[("phase", "phase_out")]
    )
    
    return graph


def main():
    print("=" * 60)
    print("CRSD Inspector - dagex Workflow Demo")
    print("=" * 60)
    
    # Create the workflow
    print("\n1. Building workflow graph...")
    graph = create_demo_workflow()
    dag = graph.build()
    print("   ✓ Workflow built successfully")
    
    # Try to generate Mermaid diagram
    try:
        print("\n2. Generating Mermaid diagram...")
        mermaid = dag.to_mermaid()
        print("   Workflow structure:")
        print("   " + "-" * 56)
        for line in mermaid.split('\n'):
            print("   " + line)
        print("   " + "-" * 56)
    except Exception as e:
        print(f"   Note: Could not generate Mermaid diagram: {e}")
    
    # Execute sequentially
    print("\n3. Executing workflow (sequential)...")
    context_seq = dag.execute(False, None)
    print("   ✓ Sequential execution completed")
    
    # Execute in parallel
    print("\n4. Executing workflow (parallel)...")
    context_par = dag.execute(True, 4)
    print("   ✓ Parallel execution completed")
    
    # Display results
    print("\n5. Results:")
    print("   " + "-" * 56)
    
    metadata = context_par.get("meta")
    if metadata:
        print(f"   Metadata:\n   {metadata.replace(chr(10), chr(10) + '   ')}")
    
    signal_shape = context_par.get("signal_shape")
    if signal_shape:
        print(f"\n   Signal Shape: {signal_shape}")
    
    amp = context_par.get("amp")
    if amp is not None:
        print(f"   Amplitude: min={np.min(amp):.2f}, max={np.max(amp):.2f}, mean={np.mean(amp):.2f}")
    
    phase = context_par.get("phase_out")
    if phase is not None:
        print(f"   Phase: min={np.min(phase):.2f}, max={np.max(phase):.2f}, mean={np.mean(phase):.2f}")
    
    print("   " + "-" * 56)
    print("\n✓ Demo completed successfully!")
    print("\nTo visualize CRSD files interactively, run:")
    print("  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
