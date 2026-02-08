#!/usr/bin/env python3
"""
Check ground truth parameters for example_2.crsd
"""
import sarkit.crsd as skcrsd
import numpy as np

# Load example_2.crsd
with open('examples/example_2.crsd', 'rb') as f:
    reader = skcrsd.Reader(f)
    
    print("=" * 80)
    print("EXAMPLE_2.CRSD GROUND TRUTH")
    print("=" * 80)
    print()
    
    # File Header KVPs
    print("File Header Information:")
    if hasattr(reader.metadata.file_header_part, 'additional_kvps'):
        kvps = reader.metadata.file_header_part.additional_kvps
        for k, v in kvps.items():
            print(f"  {k}: {v}")
    print()
    
    # Check for PPP data (pulse timing)
    print("Pulse Timing (from PPP):")
    try:
        ppp = reader.get_ppp('TX1')
        if ppp is not None and len(ppp) > 0:
            print(f"  Number of pulses: {len(ppp)}")
            if 'TxTime' in ppp.dtype.names:
                tx_times = ppp['TxTime']
                if len(tx_times) > 1:
                    pris = tx_times[1:] - tx_times[:-1]
                    prfs = 1.0 / pris
                    print(f"  PRF range: {prfs.min():.2f} - {prfs.max():.2f} Hz")
                    print(f"  PRF mean: {prfs.mean():.2f} Hz")
                    print(f"  PRF std: {prfs.std():.2f} Hz")
                    print(f"  First 10 PRIs (ms): {pris[:10] * 1000}")
    except Exception as e:
        print(f"  Could not read PPP data: {e}")
    print()
    
    # Target information from create script
    print("Target Information (from generation script):")
    print("  Target 1: Fast Car")
    print("    Range: 1500 m (15 μs round-trip)")
    print("    Doppler: +100 Hz")
    print("    RCS: 8 dBsm")
    print()
    print("  Target 2: Truck")
    print("    Range: 2800 m (28 μs round-trip)")
    print("    Doppler: -50 Hz")
    print("    RCS: 12 dBsm")
    print()
    print("  Target 3: Motorcycle")
    print("    Range: 3200 m (32 μs round-trip)")
    print("    Doppler: +20 Hz")
    print("    RCS: 6 dBsm")
    print()
    print("  Target 4: Helicopter")
    print("    Range: 4500 m (45 μs round-trip)")
    print("    Doppler: -80 Hz")
    print("    RCS: 10 dBsm")
    print()
    print("  Target 5: Tower")
    print("    Range: 6000 m (60 μs round-trip)")
    print("    Doppler: +5 Hz")
    print("    RCS: 18 dBsm")
    print()
    
    print("=" * 80)
