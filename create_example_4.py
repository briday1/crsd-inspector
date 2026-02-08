#!/usr/bin/env python3
"""
Create example_4.crsd - Simple 2-step stagger test case
Alternates between 1000 Hz and 1200 Hz PRF
2 targets for easy validation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'examples'))

from create_example_crsd import SceneConfig, RadarTarget, CRSDGenerator, StaggerPattern

# Create scene with 2-step stagger: 1000 Hz and 1200 Hz
# The nominal PRF should be the average: (1000 + 1200) / 2 = 1100 Hz
# With ratio calculated to hit exactly 1000 and 1200 Hz

nominal_prf = 1100.0  # Average of 1000 and 1200
prf_low = 1000.0
prf_high = 1200.0

# For 2-step stagger, the code alternates between:
#   pri_high = nominal_pri * (1 + stagger_ratio)  -> corresponds to prf_low
#   pri_low = nominal_pri * (1 - stagger_ratio)   -> corresponds to prf_high

# We need:
#   1/prf_low = (1/nominal_prf) * (1 + ratio)
#   1/prf_high = (1/nominal_prf) * (1 - ratio)

# Solving:
#   ratio = (nominal_prf / prf_low) - 1 = (1100/1000) - 1 = 0.1
# Verify: (nominal_prf / prf_high) - 1 = (1100/1200) - 1 ≈ -0.0833... doesn't match

# Let's calculate differently: for alternating pattern
# pri_1 = 1/1200 = 0.000833 s = 833 μs
# pri_2 = 1/1000 = 0.001000 s = 1000 μs
# avg_pri = (833 + 1000) / 2 = 916.5 μs -> nominal_prf = 1091 Hz

nominal_prf = 2.0 / (1.0/prf_low + 1.0/prf_high)  # Harmonic mean
pri_nominal = 1.0 / nominal_prf
pri_low = 1.0 / prf_high   # Short PRI = high PRF
pri_high = 1.0 / prf_low   # Long PRI = low PRF

# Calculate stagger ratio
stagger_ratio = (pri_high - pri_nominal) / pri_nominal

print("PRF Configuration:")
print(f"  Nominal PRF: {nominal_prf:.2f} Hz (harmonic mean)")
print(f"  Target PRFs: {prf_low:.0f} Hz and {prf_high:.0f} Hz")
print(f"  PRIs: {pri_low*1e6:.1f} μs and {pri_high*1e6:.1f} μs")
print(f"  Stagger ratio: {stagger_ratio:.4f} ({stagger_ratio*100:.2f}%)")
print()

scene = SceneConfig(
    num_pulses=128,
    samples_per_pulse=512,
    sample_rate_hz=100e6,
    prf_hz=nominal_prf,
    stagger_pattern=StaggerPattern.TWO_STEP,
    stagger_ratio=stagger_ratio,
    num_channels=1,
    bandwidth_hz=10e6,
    snr_db=25.0,
    targets=[
        RadarTarget(range_m=2500.0, doppler_hz=75.0, rcs_dbsm=15.0, label="Target_A"),
        RadarTarget(range_m=4000.0, doppler_hz=-45.0, rcs_dbsm=12.0, label="Target_B"),
    ],
    output_file=str(Path(__file__).parent / "examples" / "example_4.crsd"),
    verbose=True,
)

print("Generating example_4.crsd...")
print("=" * 80)
generator = CRSDGenerator(scene)
stats = generator.generate()

print()
print("=" * 80)
print("EXAMPLE_4.CRSD GROUND TRUTH")
print("=" * 80)
print()
print("PRF Stagger:")
print(f"  Pattern: 2-step")
print(f"  Nominal PRF: {nominal_prf:.2f} Hz")
print(f"  Actual PRFs: {prf_low:.0f} Hz and {prf_high:.0f} Hz")
print(f"  PRIs: {pri_high*1e6:.1f} μs and {pri_low*1e6:.1f} μs")
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
print(f"  Number of pulses: {stats.get('num_targets', scene.num_pulses)}")
print(f"  Peak SNR: {stats['peak_snr_db']:.1f} dB")
print(f"  Range resolution: {stats['range_resolution_m']:.2f} m")
print(f"  Doppler resolution: {stats['doppler_resolution_hz']:.2f} Hz")
print(f"  Compression gain: {stats['compression_gain_db']:.1f} dB")
print()
print("=" * 80)
print("✓ Generation complete!")
print(f"File: examples/example_4.crsd")
print("=" * 80)
