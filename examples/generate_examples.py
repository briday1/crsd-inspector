#!/usr/bin/env python3
"""
Create comprehensive set of CRSD example files

Generates 4 examples with embedded TX waveforms:
1. uniform_prf_1target_1ch.crsd - Simple: 1 target, uniform PRF, 1 channel
2. uniform_prf_1target_4ch.crsd - Multi-channel: 1 target, uniform PRF, 4 channels
3. fixed_stagger_3targets_1ch.crsd - Fixed stagger: 3 targets, 3-step PRF cycle, 1 channel
4. random_stagger_3targets_1ch.crsd - Random stagger: 3 targets, random PRF, 1 channel

All use same nominal TX waveform (10 MHz LFM chirp at 10 GHz X-band).

Note: Separate TX file support (decoupled TX/RX) is planned for future implementation
when workflow support is added for loading external TX waveforms.
"""

from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from create_example_crsd import (
    CRSDGenerator, SceneConfig, RadarTarget, 
    WaveformType, StaggerPattern
)


def create_all_examples():
    """Create comprehensive example set with embedded TX"""
    examples_dir = Path(__file__).parent
    
    # Common TX waveform parameters (consistent across all examples)
    common_params = {
        'samples_per_pulse': 512,
        'sample_rate_hz': 100e6,  # 100 MHz
        'bandwidth_hz': 10e6,      # 10 MHz
        'waveform_type': WaveformType.LFM,
        'carrier_freq_hz': 10e9,   # 10 GHz X-band
    }
    
    # Single target at 2000m range
    single_target = [
        RadarTarget(range_m=2000.0, doppler_hz=50.0, rcs_dbsm=10.0, label="Target_1")
    ]
    
    # Three targets at different ranges
    three_targets = [
        RadarTarget(range_m=1500.0, doppler_hz=80.0, rcs_dbsm=8.0, label="Target_1"),
        RadarTarget(range_m=2500.0, doppler_hz=-40.0, rcs_dbsm=12.0, label="Target_2"),
        RadarTarget(range_m=3500.0, doppler_hz=20.0, rcs_dbsm=10.0, label="Target_3"),
    ]
    
    examples = []
    
    # 1. Uniform PRF, 1 target, 1 channel
    examples.append(("uniform_prf_1target_1ch.crsd", SceneConfig(
        **common_params,
        num_pulses=256,
        prf_hz=1000.0,
        num_channels=1,
        snr_db=20.0,
        stagger_pattern=StaggerPattern.NONE,
        targets=single_target,
        output_file=str(examples_dir / "uniform_prf_1target_1ch.crsd"),
        verbose=True,
    )))
    
    # 2. Uniform PRF, 1 target, 4 channels
    examples.append(("uniform_prf_1target_4ch.crsd", SceneConfig(
        **common_params,
        num_pulses=256,
        prf_hz=1000.0,
        num_channels=4,
        snr_db=20.0,
        stagger_pattern=StaggerPattern.NONE,
        targets=single_target,
        output_file=str(examples_dir / "uniform_prf_1target_4ch.crsd"),
        verbose=True,
    )))
    
    # 3. Fixed stagger, 3 targets, 1 channel
    examples.append(("fixed_stagger_3targets_1ch.crsd", SceneConfig(
        **common_params,
        num_pulses=300,  # Multiple of 3 for clean cycle
        prf_hz=1200.0,
        num_channels=1,
        snr_db=18.0,
        stagger_pattern=StaggerPattern.THREE_STEP,
        stagger_ratio=0.15,
        targets=three_targets,
        output_file=str(examples_dir / "fixed_stagger_3targets_1ch.crsd"),
        verbose=True,
    )))
    
    # 4. Random stagger, 3 targets, 1 channel
    examples.append(("random_stagger_3targets_1ch.crsd", SceneConfig(
        **common_params,
        num_pulses=256,
        prf_hz=1200.0,
        num_channels=1,
        snr_db=18.0,
        stagger_pattern=StaggerPattern.RANDOM,
        stagger_ratio=0.20,
        targets=three_targets,
        output_file=str(examples_dir / "random_stagger_3targets_1ch.crsd"),
        verbose=True,
    )))
    
    return examples


if __name__ == "__main__":
    print("=" * 80)
    print("CRSD EXAMPLE GENERATOR - Creating Example Set")
    print("=" * 80)
    print()
    
    examples = create_all_examples()
    
    for i, (name, config) in enumerate(examples, 1):
        print(f"\n{'='*80}")
        print(f"Example {i}/{len(examples)}: {name}")
        print(f"{'='*80}")
        
        try:
            generator = CRSDGenerator(config)
            stats = generator.generate()
            
            print(f"\n  File: {config.output_file}")
            print(f"  Size: {stats['file_size_bytes'] / 1024:.1f} KB")
            print(f"  Targets: {stats['num_targets']}")
            print(f"  Channels: {config.num_channels}")
            if 'stagger_pattern' in stats:
                print(f"  Stagger: {stats['stagger_pattern']}")
                print(f"  PRF Range: [{stats['min_prf_hz']:.1f}, {stats['max_prf_hz']:.1f}] Hz")
            else:
                print(f"  PRF: {config.prf_hz:.1f} Hz (uniform)")
            print(f"  Peak SNR: {stats['peak_snr_db']:.1f} dB")
            print(f"\n  ✅ Success")
            
        except Exception as e:
            print(f"\n  ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("GENERATION COMPLETE")
    print(f"All {len(examples)} examples created successfully")
    print(f"{'='*80}")
