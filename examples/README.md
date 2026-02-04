# Example CRSD Files

This directory contains example and synthetic CRSD files for testing the CRSD Inspector application.

## Files

- `create_example_crsd.py` - Script to generate a small synthetic CRSD file

## Generating Example Files

To create a synthetic example CRSD file:

```bash
python3 create_example_crsd.py
```

This will create `example_small.crsd` (approximately 256 KB) with:
- 256 vectors × 256 samples
- Synthetic point targets
- Realistic noise characteristics
- Full CRSD metadata structure

## Notes

- The synthetic files are useful for testing the application without requiring real radar data
- Real CRSD files can be quite large (GB scale) and are not included in the repository
- The app will automatically generate synthetic data when a real file is not available
