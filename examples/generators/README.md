# CRSD Example Generators

This directory contains scripts that generate synthetic CRSD files with various PRF patterns.

## Generators

### create_example_4.py
Generates `example_4.crsd` with continuous hold-and-move PRF stagger pattern.
- Hold at one PRF for multiple pulses
- Then move to different PRF
- Good for testing stagger detection algorithms

### create_example_5.py
Generates `example_5.crsd` with PRF walk pattern.
- 3 pulses at 1000 Hz
- 3 pulses at 1100 Hz  
- 3 pulses at 1200 Hz
- Then repeats (9-pulse cycle)

### create_example_6.py
Generates `example_6.crsd` with dual-stage PRI stagger pattern.
- High set: alternates between 1500 Hz and 1600 Hz
- Low set: alternates between 2000 Hz and 2100 Hz
- Every other pulse switches between high/low sets (16 pulses)
- Then 4 random PRIs before repeating (20-pulse cycle)

## Usage

Run any generator script:
```bash
python examples/generators/create_example_5.py
```

Generated `.crsd` files are written to the `examples/` directory but are not committed to git (per `.gitignore`).

## Creating Custom Generators

All generators inherit from `CRSDGenerator` in `examples/create_example_crsd.py`.
Override `_generate_pulse_times()` to create custom PRF patterns.
