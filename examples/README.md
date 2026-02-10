# CRSD Example Files

This directory contains synthetic CRSD (Compensated Received Signal Data) files for testing and demonstration.

## TX Waveform Variants

Each example is available in **4 variants** based on TX waveform storage:

### TX Waveform Types

1. **Single Pulse TX** (`single`): One pulse worth of TX waveform
   - Workflows must perform PRF selection and pulse extraction
   - Smaller file size (original examples use this)
   - File contains only the template waveform
   
2. **Full Sequence TX** (`full`): Complete transmitted sequence for all pulses
   - Simplifies PRF extraction logic - timing already known
   - ~2x file size (includes full TX sequence)
   - File contains TX waveform at every pulse time

### TX Storage Locations

- **Embedded** (`embedded`): TX waveform stored in same CRSD file
  - Self-contained, single file
  - Standard approach, no external dependencies
  
- **External** (`external`): TX waveform stored in separate CRSD file
  - Main RX file has placeholder (single 0 sample)
  - Must load companion `*_tx.crsd` file via `tx_crsd_file` parameter
  - Useful for shared TX waveforms across multiple RX collections
  - TX file contains complete waveform information and PPP timing

### Variant Naming Convention

Format: `{example}_{tx-type}-{storage}.crsd`

**Examples:**
- `uniform_prf_1target_1ch_single-embedded.crsd` - single pulse TX embedded (original format)
- `uniform_prf_1target_1ch_single-external.crsd` + `*_single_tx.crsd` - single pulse TX in separate file
- `uniform_prf_1target_1ch_full-embedded.crsd` - full sequence TX embedded  
- `uniform_prf_1target_1ch_full-external.crsd` + `*_full_tx.crsd` - full sequence TX in separate file

### File Sizes

- **single-embedded**: ~160-200 MB (original format)
- **single-external**: ~160-200 MB + 8 KB TX file
- **full-embedded**: ~320-400 MB (2x due to full TX sequence)
- **full-external**: ~160-200 MB + ~160-200 MB TX file

## Base Examples

All examples use the same nominal TX waveform (10 MHz LFM chirp at 10 GHz X-band, 100 MHz sample rate) for consistency.

### 1. uniform_prf_1target_1ch
**Simple baseline example**
- 1 channel
- 1 target (3000m range, 50 Hz Doppler)
- Uniform PRF: 1000 Hz
- 256 pulses
- SNR: 20 dB
- Perfect for basic workflow testing
- **Variants**: 4 (single/full × embedded/external) = 8 files

### 2. uniform_prf_1target_4ch
**Multi-channel example**
- 4 channels (CHAN1-CHAN4)
- 1 target (same as above)
- Uniform PRF: 1000 Hz
- 256 pulses
- SNR: 20 dB
- Tests multi-channel processing and beamforming
- **Variants**: 4 (single/full × embedded/external) = 8 files

### 3. fixed_stagger_3targets_1ch
**Fixed PRF stagger cycle**
- 1 channel
- 3 targets at different ranges (2500m, 3500m, 4500m)
- 3-step stagger pattern (1043-1411 Hz)
- 300 pulses (100 cycles)
- SNR: 18 dB
- Tests fixed stagger PRF handling
- **Variants**: 4 (single/full × embedded/external) = 8 files

### 4. random_stagger_3targets_1ch
**Random PRF variation**
- 1 channel
- 3 targets (2000m, 3000m, 5000m)
- Random PRF variation (1002-1494 Hz)
- 256 pulses
- SNR: 18 dB
- Tests random stagger PRF handling
- **Variants**: 4 (single/full × embedded/external) = 8 files

## Total Files

- **4 base examples** (original format, kept for compatibility)
- **16 variant files** (4 examples × 4 variants each)
- **8 external TX files** (4 examples × 2 external variants)
- **Total: 28 CRSD files**

## Generating Examples

### Generate all variants (recommended)
```bash
python generate_all_tx_variants.py
```

This creates all 24 files (16 main variants + 8 external TX files).

### Generate only base examples
```bash
python generate_examples.py
```

This creates only the 4 original single-embedded examples.

### Custom scenarios
Use `create_example_crsd.py` module to create custom configurations.

## Workflow Usage

### Using Embedded TX (Default)
Simply select the CRSD file - TX waveform is already included:
```bash
crsd-inspector run signal_analysis --file uniform_prf_1target_1ch_single-embedded.crsd
```

### Using External TX
Specify both RX and TX files:
```bash
crsd-inspector run range_doppler_processing \
  --file uniform_prf_1target_1ch_single-external.crsd \
  --tx_crsd_file uniform_prf_1target_1ch_single-external_single_tx.crsd
```

### Full Sequence TX (Simplified PRF Detection)
Use full sequence variants when PRF timing is critical:
```bash
crsd-inspector run signal_analysis --file uniform_prf_1target_1ch_full-embedded.crsd
```

The workflow will automatically detect full sequence TX and skip PRF detection.

## Metadata

Each file includes ground truth metadata in the file header KVPs:
- `NUM_TARGETS`: Number of simulated targets
- `SNR_DB`: Target signal-to-noise ratio
- `SAMPLE_RATE_HZ`: ADC sample rate (100 MHz)
- `PRF_HZ`: Nominal pulse repetition frequency
- `BANDWIDTH_HZ`: TX waveform bandwidth (10 MHz)
- `STAGGER_PATTERN`: PRF stagger type (none, 3-step, random)
- `NUM_PULSES`: Total number of pulses

External TX files also include:
- `TX_TYPE`: Either `SINGLE_PULSE` or `FULL_SEQUENCE`
- `CREATOR`: `crsd-inspector`

## File Organization

```
examples/
├── README.md
├── create_example_crsd.py          # Base generator module
├── generate_examples.py            # Generate 4 base examples
├── generate_all_tx_variants.py     # Generate all 24 variants
│
├── uniform_prf_1target_1ch.crsd                        # Original (single-embedded)
├── uniform_prf_1target_1ch_single-embedded.crsd        # Explicit variant name
├── uniform_prf_1target_1ch_single-external.crsd        # + TX file
├── uniform_prf_1target_1ch_single-external_single_tx.crsd
├── uniform_prf_1target_1ch_full-embedded.crsd
├── uniform_prf_1target_1ch_full-external.crsd          # + TX file
├── uniform_prf_1target_1ch_full-external_full_tx.crsd
│
└── [same pattern repeated for other 3 examples]
```
