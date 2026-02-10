#!/usr/bin/env python3
"""
Create example_6.crsd - Dual-stage PRI stagger pattern
Alternates between high and low PRF sets, each with internal stagger.
Pattern:
  - High set: alternates 1500 Hz, 1600 Hz (PRI = 666.67 μs, 625 μs)
  - Low set: alternates 2000 Hz, 2100 Hz (PRI = 500 μs, 476.19 μs)
  - Every other pulse switches between high/low sets
  - After sequence, adds random PRIs before repeating
"""

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from create_example_crsd import SceneConfig, RadarTarget, CRSDGenerator, StaggerPattern

# Custom dual-stage stagger pattern generator
class DualStagePRIGenerator(CRSDGenerator):
    """Generator with dual-stage PRI stagger pattern"""
    
    def _generate_pulse_times(self) -> np.ndarray:
        """Generate pulse timing with dual-stage stagger pattern"""
        P = self.config.num_pulses
        
        # Define high and low PRF sets
        high_prfs = [1500.0, 1600.0]  # Hz
        low_prfs = [2000.0, 2100.0]   # Hz
        
        # Sequence parameters
        sequence_length = 16  # pulses per sequence (before random PRIs)
        num_random = 4        # random PRIs at end of sequence
        
        pulse_times = [0.0]
        high_idx = 0
        low_idx = 0
        
        for i in range(1, P):
            # Determine position in overall cycle
            cycle_pos = (i - 1) % (sequence_length + num_random)
            
            if cycle_pos < sequence_length:
                # Regular dual-stage pattern
                is_high = (cycle_pos % 2) == 0
                
                if is_high:
                    # Use high set with internal stagger
                    current_prf = high_prfs[high_idx % len(high_prfs)]
                    high_idx += 1
                else:
                    # Use low set with internal stagger
                    current_prf = low_prfs[low_idx % len(low_prfs)]
                    low_idx += 1
            else:
                # Random PRI region
                # Use random PRF between 1200-2200 Hz
                current_prf = np.random.uniform(1200.0, 2200.0)
            
            current_pri = 1.0 / current_prf
            pulse_times.append(pulse_times[-1] + current_pri)
        
        return np.array(pulse_times, dtype=np.float64)
    
    def _get_ground_truth_kvps(self) -> dict:
        """Get custom ground truth KVPs for file header"""
        kvps = {
            "CREATOR": "crsd-inspector",
            "NUM_CHANNELS": str(self.config.num_channels),
            "NUM_TARGETS": str(len(self.config.targets)),
            "SNR_DB": f"{self.config.snr_db:.2f}",
            "SAMPLE_RATE_HZ": f"{self.config.sample_rate_hz:.0f}",
            "PRF_HZ": f"{self.config.prf_hz:.1f}",
            "BANDWIDTH_HZ": f"{self.config.bandwidth_hz:.0f}",
            "STAGGER_PATTERN": "dual_stage",
            "NUM_PULSES": str(self.config.num_pulses),
            "GROUND_TRUTH": "Dual-Stage PRI Stagger||High set: 1500 Hz, 1600 Hz (PRI = 666.67 μs, 625 μs)||Low set: 2000 Hz, 2100 Hz (PRI = 500 μs, 476.19 μs)||Pattern: H1, L1, H2, L2, ... (16 pulses)||Then: 4 random PRIs (1200-2200 Hz)||Cycle length: 20 pulses",
        }
        return kvps


print("Dual-Stage PRI Stagger Configuration:")
print("  High set: 1500 Hz, 1600 Hz (PRI = 666.67 μs, 625 μs)")
print("  Low set: 2000 Hz, 2100 Hz (PRI = 500 μs, 476.19 μs)")
print("  Pattern: H1, L1, H2, L2, H1, L1, H2, L2, ... (16 pulses)")
print("  Then: 4 random PRIs (1200-2200 Hz)")
print("  Total cycle: 20 pulses, then repeat")
print()

scene = SceneConfig(
    num_pulses=128,
    samples_per_pulse=512,
    sample_rate_hz=100e6,
    prf_hz=1800.0,  # Nominal (middle value)
    stagger_pattern=StaggerPattern.NONE,  # We override the pattern anyway
    num_channels=1,
    bandwidth_hz=10e6,
    snr_db=25.0,
    targets=[
        RadarTarget(range_m=3000.0, doppler_hz=120.0, rcs_dbsm=18.0, label="Target_A"),
        RadarTarget(range_m=4500.0, doppler_hz=-80.0, rcs_dbsm=14.0, label="Target_B"),
        RadarTarget(range_m=2200.0, doppler_hz=30.0, rcs_dbsm=10.0, label="Target_C"),
    ],
    output_file=str(Path(__file__).parent / "examples" / "example_6.crsd"),
    verbose=True,
)

print("Generating example_6.crsd...")
print("=" * 80)
generator = DualStagePRIGenerator(scene)
stats = generator.generate()

# Calculate actual PRF stats from generated pulse times
if generator.pulse_times is not None and len(generator.pulse_times) > 1:
    pris = np.diff(generator.pulse_times)
    prfs = 1.0 / pris
    
    print()
    print("=" * 80)
    print("EXAMPLE_6.CRSD GROUND TRUTH")
    print("=" * 80)
    print()
    print("Dual-Stage PRI Stagger Pattern:")
    print("  Sequence structure (20 pulses):")
    print("    Pulses 1-16: Alternating high/low sets")
    print("      - High set (odd pulses): 1500, 1600 Hz (666.67, 625 μs)")
    print("      - Low set (even pulses): 2000, 2100 Hz (500, 476.19 μs)")
    print("    Pulses 17-20: Random PRIs (1200-2200 Hz)")
    print("    Then cycle repeats...")
    print()
    print(f"  Actual PRF range: {prfs.min():.1f} - {prfs.max():.1f} Hz")
    print(f"  Mean PRF: {prfs.mean():.1f} Hz")
    print(f"  PRI std dev: {pris.std()*1e6:.2f} μs")
    print()
    print("Targets:")
    print("  1. Target_A:")
    print(f"     Range: 3000 m ({2*3000/3e8*1e6:.2f} μs round-trip)")
    print(f"     Doppler: +120 Hz")
    print(f"     RCS: 18 dBsm")
    print()
    print("  2. Target_B:")
    print(f"     Range: 4500 m ({2*4500/3e8*1e6:.2f} μs round-trip)")
    print(f"     Doppler: -80 Hz")
    print(f"     RCS: 14 dBsm")
    print()
    print("  3. Target_C:")
    print(f"     Range: 2200 m ({2*2200/3e8*1e6:.2f} μs round-trip)")
    print(f"     Doppler: +30 Hz")
    print(f"     RCS: 10 dBsm")
    print()
    print("File Statistics:")
    print(f"  Number of pulses: {len(generator.pulse_times)}")
    print(f"  Peak SNR: {stats['peak_snr_db']:.1f} dB")
    print(f"  Range resolution: {stats['range_resolution_m']:.2f} m")
    print()
    print("PRI Pattern (first 22 pulses):")
    for i in range(min(22, len(pris))):
        print(f"  Pulse {i+1:3d}: PRI = {pris[i]*1e6:7.2f} μs, PRF = {prfs[i]:7.1f} Hz")
