# Utilities

This directory contains utility scripts for the CRSD Inspector project.

## Scripts

### pulse_extraction_viewer.py
Generates static HTML reports using staticdash for pulse extraction workflow results.
Useful for offline viewing and sharing analysis reports.

**Usage:**
```bash
python utils/pulse_extraction_viewer.py examples/example_6.crsd
```

Output is written to `pulse_extraction_output/index.html`

### analyze_prf_sequence.py
Analyzes the PRF/PRI sequence in a CRSD file and displays detailed statistics.
Shows pulse-by-pulse PRF values, pattern detection, and cycle analysis.

**Usage:**
```bash
python utils/analyze_prf_sequence.py
```

Analyzes `examples/example_6.crsd` by default.
