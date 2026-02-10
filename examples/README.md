# CRSD Example Files

This directory contains synthetic CRSD (Compensated Received Signal Data) files for testing and demonstration.

## Example Files

All examples use the same nominal TX waveform (10 MHz LFM chirp at 10 GHz X-band, 100 MHz sample rate) for consistency.

### 1. `uniform_prf_1target_1ch.crsd`
**Simple baseline example**
- 1 channel
- 1 target (2000m range, 50 Hz Doppler)
- Uniform PRF: 1000 Hz
- 256 pulses
- Perfect for basic workflow testing

### 2. `uniform_prf_1target_4ch.crsd`
**Multi-channel example**
- 4 channels (CHAN1-CHAN4)
- 1 target (same as above)
- Uniform PRF: 1000 Hz
- 256 pulses
- Tests multi-channel processing

### 3. `fixed_stagger_3targets_1ch.crsd`
**Fixed PRF stagger cycle**
- 1 channel
- 3 targets at different ranges (1500m, 2500m, 3500m)
- 3-step stagger pattern (1043-1411 Hz)
- 300 pulses (100 cycles)
- Tests fixed stagger PRF handling

### 4. `random_stagger_3targets_1ch.crsd`
**Random PRF variation**
- 1 channel
- 3 targets (same as above)
- Random PRF variation (1002-1494 Hz)
- 256 pulses
- Tests random stagger PRF handling

## Generating Examples

Run the generation script to recreate all examples:

```bash
python generate_examples.py
```

Or use create_example_crsd.py for custom scenarios.

## Metadata

Each file includes ground truth metadata in the file header KVPs:
- `NUM_TARGETS`: Number of simulated targets
- `SNR_DB`: Target signal-to-noise ratio
- `SAMPLE_RATE_HZ`: ADC sample rate
- `PRF_HZ`: Nominal pulse repetition frequency
- `BANDWIDTH_HZ`: TX waveform bandwidth
- `STAGGER_PATTERN`: PRF stagger type (none, 3-step, random)

## Future Work

Separate TX/RX file support (decoupled TX waveform files) is planned for future implementation when workflow support is added for loading external TX waveforms.
