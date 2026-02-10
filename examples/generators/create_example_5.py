#!/usr/bin/env python3
"""
Create example_5.crsd - PRF "walk" pattern
Uses a stepped PRF pattern: pulses at one PRF, then step to next, repeat
Pattern: 3 pulses @ 1000 Hz, 3 pulses @ 1100 Hz, 3 pulses @ 1200 Hz, repeat
"""

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from create_example_crsd import SceneConfig, RadarTarget, CRSDGenerator, StaggerPattern

# Custom PRF walk pattern generator
class PRFWalkGenerator(CRSDGenerator):
    """Generator with custom PRF walk pattern"""
    
    def _generate_pulse_times(self) -> np.ndarray:
        """Generate pulse timing with PRF walk pattern"""
        P = self.config.num_pulses
        
        # Define PRF walk pattern: 3 pulses each at 1000, 1100, 1200 Hz
        prf_steps = [1000.0, 1100.0, 1200.0]
        pulses_per_step = 3
        
        pulse_times = [0.0]
        for i in range(1, P):
            # Determine which PRF to use based on position in cycle
            cycle_position = (i - 1) % (len(prf_steps) * pulses_per_step)
            step_idx = cycle_position // pulses_per_step
            current_prf = prf_steps[step_idx]
            current_pri = 1.0 / current_prf
            
            # Add next pulse time
            pulse_times.append(pulse_times[-1] + current_pri)
        
        return np.array(pulse_times, dtype=np.float64)


print("PRF Walk Configuration:")
print("  Pattern: 3 pulses @ 1000Hz, 3 @ 1100Hz, 3 @ 1200Hz, repeat")
print("  PRIs: 1000 μs, 909 μs, 833 μs")
print()

scene = SceneConfig(
    num_pulses=128,
    samples_per_pulse=512,
    sample_rate_hz=100e6,
    prf_hz=1100.0,  # Nominal (middle value)
    stagger_pattern=StaggerPattern.NONE,  # We override the pattern anyway
    num_channels=1,
    bandwidth_hz=10e6,
    snr_db=25.0,
    targets=[
        RadarTarget(range_m=2500.0, doppler_hz=75.0, rcs_dbsm=15.0, label="Target_A"),
        RadarTarget(range_m=4000.0, doppler_hz=-45.0, rcs_dbsm=12.0, label="Target_B"),
    ],
    output_file=str(Path(__file__).parent / "examples" / "example_5.crsd"),
    verbose=True,
)

print("Generating example_5.crsd...")
print("=" * 80)
generator = PRFWalkGenerator(scene)
stats = generator.generate()

# Calculate actual PRF stats from generated pulse times
if generator.pulse_times is not None and len(generator.pulse_times) > 1:
    pris = np.diff(generator.pulse_times)
    prfs = 1.0 / pris
    
    print()
    print("=" * 80)
    print("EXAMPLE_5.CRSD GROUND TRUTH")
    print("=" * 80)
    print()
    print("PRF Walk Pattern:")
    print("  3 pulses @ 1000 Hz (PRI = 1000 μs)")
    print("  3 pulses @ 1100 Hz (PRI = 909 μs)")
    print("  3 pulses @ 1200 Hz (PRI = 833 μs)")
    print("  ...repeating cycle")
    print(f"  Actual PRF range: {prfs.min():.1f} - {prfs.max():.1f} Hz")
    print(f"  Mean PRF: {prfs.mean():.1f} Hz")
    print()
    print("Targets:")
    print("  1. Target_A:")
    print(f"     Range: 2500 m ({2*2500/3e8*1e6:.2f} μs round-trip)")
    print(f"     Doppler: +75 Hz")
    print(f"     RCS: 15 dBsm")
    print()
    print("  2. Target_B:")
    print(f"     Range: 4000 m ({2*4000/3e8*1e6:.2f} μs round-trip)")
    print(f"     Doppler: -45 Hz")
    print(f"     RCS: 12 dBsm")
    print()
    print("File Statistics:")
    print(f"  Number of pulses: {len(generator.pulse_times)}")
    print(f"  Peak SNR: {stats['peak_snr_db']:.1f} dB")
    print(f"  Range resolution: {stats['range_resolution_m']:.2f} m")
    print(f"  Doppler resolution: {stats['doppler_resolution_hz']:.2f} Hz")
    print(f"  Compression gain: {stats['compression_gain_db']:.1f} dB")
    print()
    print("First 15 PRIs (μs):")
    print(f"  {pris[:15] * 1e6}")
    print()
    print("=" * 80)
    print("✓ Generation complete!")
    print(f"File: examples/example_5.crsd")
    print("=" * 80)
