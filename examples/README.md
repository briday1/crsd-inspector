# Example CRSD Files

This directory contains example CRSD files for testing the CRSD Inspector application.

## Files

- `example_1.crsd` - Simple scene: 3 targets, single channel (1 MB)
- `example_2.crsd` - Complex scene: 5 targets, single channel (1 MB)
- `example_3.crsd` - Multi-channel: 2 targets, 2 channels (1 MB)

## CRSD Generator

The `generate_crsd.py` script (in the main directory) generates realistic synthetic CRSD files based on test-dagex capabilities.

### Features

- **Realistic TX Waveforms**: LFM chirps and BPSK codes stored as support arrays
- **Multi-Channel Support**: Generate files with 1-N receiver channels
- **Point Targets**: Configurable range, Doppler, RCS for each target
- **Proper Metadata**: Complete radar parameters (sample rate, PRF, bandwidth)
- **Matched Filtering Ready**: TX waveforms available for pulse compression

### Generating Files

Run the script to regenerate all 3 example files:
from the main directory to regenerate all 3 example files:

```bash
python generat

### File Details

**Example 1: Simple Scene**
- 3 targets: Vehicle, UAV, Building
- Range: 2-5 km
- Doppler: -30 to +50 Hz
- Single channel
- SNR: 20 dB

**Example 2: Complex Scene**
- 5 targets: Fast Car, Truck, Motorcycle, Helicopter, Tower
- Range: 1.5-6 km
- Doppler: -80 to +100 Hz
- Single channel
- SNR: 18 dB

**Example 3: Multi-Channel**
- 2 targets: Aircraft, Boat
- Range: 3-4 km
- Doppler: -40 to +60 Hz
- **2 channels** with independent noise
- SNR: 22 dB

### Technical Specifications

All example files use:
- **Sample Rate**: 100 MHz
- **PRF**: 1000 Hz
- **Bandwidth**: 10 MHz
- **Pulses**: 256
- **Samples per Pulse**: 512
- **Waveform**: LFM chirp with time-bandwidth product = 50
- **Range Resolution**: 15 m
- **Doppler Resolution**: 3.91 Hz
- **Compression Gain**: 17 dB

### Custom Scene Generation

The script can be easily modified to create custom scenes. Key parameters:

```python
from generate_crsd import SceneConfig, RadarTarget, CRSDGenerator

# Configure scene
scene = SceneConfig(
    num_pulses=256,
    samples_per_pulse=512,
    sample_rate_hz=100e6,
    prf_hz=1000.0,
    num_channels=1,
    bandwidth_hz=10e6,
    snr_db=20.0,
    targets=[
        RadarTarget(range_m=3000.0, doppler_hz=50.0, rcs_dbsm=10.0, label="MyTarget"),
    ],
    output_file="custom.crsd",
    verbose=True,
)

# Generate
generator = CRSDGenerator(scene)
stats = generator.generate()
```

## Notes

- All files conform to **NGA.STND.0080** (CRSD 1.0 specification)
- Generated using **sarkit** for proper CRSD I/O
- TX waveforms stored in support arrays enable matched filtering workflows
- Multi-channel files demonstrate array processing capabilities
- Files are suitable for testing range-Doppler processing, signal quality analysis, and performance metrics
