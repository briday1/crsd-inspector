#!/usr/bin/env python
"""Test pulse detection on continuous file"""
from sarkit.crsd import Reader
from crsd_inspector.workflows.pulse_extraction import detect_pulse_starts
import numpy as np

with open('examples/example_continuous.crsd', 'rb') as f:
    reader = Reader(f)
    signal = reader.read_signal('CHAN1')

print(f'Signal shape: {signal.shape}')
print(f'Expected: 64 pulses at 1000 Hz PRF, 100 MHz sample rate')
print(f'Expected pulse spacing: 100,000 samples (1 ms PRI)\n')

# Test detection with various thresholds
for thresh in [-10, -20, -30, -40, -50]:
    pulse_starts, pulse_times, stats = detect_pulse_starts(
        signal[0],
        sample_rate_hz=100e6,
        min_prf_hz=500,
        max_prf_hz=2000,
        power_threshold_db=thresh
    )
    
    print(f'Threshold {thresh:3d} dB: {stats["num_pulses"]:3d} pulses', end='')
    if stats["num_pulses"] > 0:
        print(f', avg PRF: {stats.get("avg_prf_hz", 0):.1f} Hz')
        if stats["num_pulses"] <= 10:
            print(f'  Pulse starts: {pulse_starts}')
    else:
        print()
