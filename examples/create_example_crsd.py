#!/usr/bin/env python3
"""
Script to create a synthetic example CRSD file for testing
"""
import numpy as np
from sarkit.crsd import Writer, Metadata
from datetime import datetime
import os

def create_example_crsd(output_path="example.crsd"):
    """Create a small synthetic CRSD file for demonstration"""
    
    # Create synthetic signal data (small for repository inclusion)
    num_vectors = 256  # Number of pulse vectors
    num_samples = 256  # Samples per vector
    
    # Generate complex signal data with some structure
    signal_data = np.zeros((num_vectors, num_samples), dtype=np.complex64)
    
    # Add some synthetic targets
    for i in range(3):
        target_vec = 80 + i * 50
        target_sample = 100 + i * 40
        # Create a point target with some spread
        for dv in range(-5, 6):
            for ds in range(-5, 6):
                v = target_vec + dv
                s = target_sample + ds
                if 0 <= v < num_vectors and 0 <= s < num_samples:
                    # Add amplitude that decreases with distance
                    amp = 10.0 * np.exp(-0.1 * (dv**2 + ds**2))
                    phase = np.random.uniform(-np.pi, np.pi)
                    signal_data[v, s] += amp * np.exp(1j * phase)
    
    # Add noise
    noise = 0.1 * (np.random.randn(num_vectors, num_samples) + 
                   1j * np.random.randn(num_vectors, num_samples))
    signal_data += noise.astype(np.complex64)
    
    # Create minimal metadata
    meta = Metadata()
    
    # Set basic collection info
    meta.CollectionInfo = Metadata.CollectionInfoType()
    meta.CollectionInfo.CollectorName = "Example Synthetic Radar"
    meta.CollectionInfo.CoreName = "EXAMPLE_CRSD_001"
    meta.CollectionInfo.CollectType = "MONOSTATIC"
    meta.CollectionInfo.Classification = "UNCLASSIFIED"
    
    # Set radar mode
    meta.CollectionInfo.RadarMode = Metadata.CollectionInfoType.RadarModeType()
    meta.CollectionInfo.RadarMode.ModeID = "SPOTLIGHT"
    
    # Set timeline
    meta.Global = Metadata.GlobalType()
    meta.Global.CollectStart = datetime.utcnow().isoformat() + "Z"
    meta.Global.CollectDuration = 1.0  # seconds
    meta.Global.TxTime1 = 0.0
    meta.Global.TxTime2 = 1.0
    
    # Set image area (simple example)
    meta.SceneCoordinates = Metadata.SceneCoordinatesType()
    meta.SceneCoordinates.EarthModel = "WGS_84"
    meta.SceneCoordinates.ReferenceSurface = Metadata.SceneCoordinatesType.ReferenceSurfaceType()
    meta.SceneCoordinates.ReferenceSurface.Planar = Metadata.SceneCoordinatesType.ReferenceSurfaceType.PlanarType()
    meta.SceneCoordinates.ReferenceSurface.Planar.uIAX = Metadata.XyzType(X=1.0, Y=0.0, Z=0.0)
    meta.SceneCoordinates.ReferenceSurface.Planar.uIAY = Metadata.XyzType(X=0.0, Y=1.0, Z=0.0)
    
    # Reference point
    meta.ReferenceGeometry = Metadata.ReferenceGeometryType()
    meta.ReferenceGeometry.SRP = Metadata.ReferenceGeometryType.SRPType()
    meta.ReferenceGeometry.SRP.ECF = Metadata.XyzType(X=0.0, Y=0.0, Z=6378137.0)  # On equator
    meta.ReferenceGeometry.SRP.IAC = Metadata.XyzType(X=0.0, Y=0.0, Z=0.0)
    
    # Channel info
    meta.Channel = [Metadata.ChannelType()]
    meta.Channel[0].Identifier = "Channel1"
    meta.Channel[0].RefVectorIndex = 0
    meta.Channel[0].FXFixed = False
    meta.Channel[0].TOAFixed = False
    meta.Channel[0].SRPFixed = False
    
    # Signal array format
    meta.Data = Metadata.DataType()
    meta.Data.SignalArrayFormat = "CF8"  # Complex float 8 bytes
    meta.Data.NumBytesPVP = 0  # Simplified: no PVP for this example
    meta.Data.Channel = [Metadata.DataType.ChannelType()]
    meta.Data.Channel[0].Identifier = "Channel1"
    meta.Data.Channel[0].NumVectors = num_vectors
    meta.Data.Channel[0].NumSamples = num_samples
    meta.Data.Channel[0].SignalArrayByteOffset = 0
    
    # Write the CRSD file
    try:
        writer = Writer(output_path, meta)
        writer.write_signal_block(signal_data, 0, 0)
        writer.close()
        print(f"✓ Created example CRSD file: {output_path}")
        print(f"  Size: {os.path.getsize(output_path) / 1024:.1f} KB")
        print(f"  Dimensions: {num_vectors} vectors × {num_samples} samples")
        return True
    except Exception as e:
        print(f"✗ Error creating CRSD file: {e}")
        print(f"  This is expected - sarkit Writer may not support all features")
        print(f"  The app will work with real CRSD files or generate synthetic data")
        return False

if __name__ == "__main__":
    output_file = os.path.join(os.path.dirname(__file__), "example_small.crsd")
    create_example_crsd(output_file)
