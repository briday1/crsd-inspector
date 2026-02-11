#!/usr/bin/env python3
"""
Generate TX waveform variants for each example:
- Single pulse TX, embedded
- Single pulse TX, external file
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from create_example_crsd import (
    CRSDGenerator, SceneConfig, RadarTarget, 
    WaveformType, StaggerPattern
)
import numpy as np
import lxml.etree as ET
import sarkit.crsd as skcrsd


class TXVariantGenerator(CRSDGenerator):
    """Extended generator that can create all TX waveform variants"""
    
    def __init__(self, config: SceneConfig, tx_mode: str = "embedded"):
        """
        tx_mode: 
            - "embedded": Single pulse TX in same file
            - "external": Single pulse TX in separate file, placeholder in main
        """
        super().__init__(config)
        self.tx_mode = tx_mode
        self.tx_file_path = None
        
    def generate(self) -> dict:
        """Generate CRSD with specified TX variant"""
        if self.config.verbose:
            print(f"Generating CRSD: {self.config.output_file}")
            print(f"  TX Mode: {self.tx_mode}")
            print(f"  Pulses: {self.config.num_pulses}, Samples: {self.config.samples_per_pulse}")
            print(f"  Channels: {self.config.num_channels}")
            print(f"  Targets: {len(self.config.targets)}")
        
        # Generate TX waveform(s)
        if self.config.verbose:
            print("  - Generating TX waveform...")
        
        single_pulse_tx = self._generate_waveform()  # Single pulse
        
        # Generate pulse times (needed for full sequence)
        self.pulse_times = self._generate_pulse_times()
        
        # Generate RX signals
        all_channels = []
        for ch_idx in range(self.config.num_channels):
            if self.config.verbose and self.config.num_channels > 1:
                print(f"  - Synthesizing channel {ch_idx+1}/{self.config.num_channels}...")
            
            self.rng = np.random.default_rng(42 + ch_idx)
            rx_signal, target_powers = self._generate_targets(single_pulse_tx)
            rx_signal, noise_power = self._add_noise(rx_signal)
            all_channels.append(rx_signal)
        
        # Write files based on mode
        if self.config.verbose:
            print("  - Writing CRSD file(s)...")
            
        if self.tx_mode == "embedded":
            # Standard: single pulse TX in same file
            file_size = self._write_crsd(all_channels, single_pulse_tx)
            
        elif self.tx_mode == "external":
            # Write TX file with single pulse
            self._write_tx_file(single_pulse_tx, is_full_sequence=False)
            # Write main file with placeholder
            file_size = self._write_crsd(all_channels, np.array([0.0+0.0j], dtype=np.complex64))
        else:
            raise ValueError(f"Unknown tx_mode: {self.tx_mode}")
        
        # Compute statistics
        peak_snr = 10 * np.log10(max(target_powers) / noise_power) if target_powers else 0.0
        
        # Waveform analysis
        T = len(single_pulse_tx) / self.config.sample_rate_hz
        tbp = T * self.config.bandwidth_hz
        compression_gain = 10 * np.log10(tbp)
        
        # Resolution
        c = 3e8
        range_res = c / (2 * self.config.bandwidth_hz)
        doppler_res = self.config.prf_hz / self.config.num_pulses
        
        # PRF statistics
        if self.pulse_times is not None and len(self.pulse_times) > 1:
            pris = np.diff(self.pulse_times)
            prfs = 1.0 / pris
            avg_prf = np.mean(prfs)
            min_prf = np.min(prfs)
            max_prf = np.max(prfs)
            std_prf = np.std(prfs)
        else:
            avg_prf = min_prf = max_prf = self.config.prf_hz
            std_prf = 0.0
        
        if self.config.verbose:
            if self.config.stagger_pattern != StaggerPattern.NONE:
                print(f"  PRF Stats: Avg {avg_prf:.1f} Hz, Range [{min_prf:.1f}, {max_prf:.1f}] Hz")
            if "external" in self.tx_mode:
                print(f"  TX File: {self.tx_file_path}")
            print(f"  ✓ Complete")
        
        stats = {
            'file_size_bytes': file_size,
            'tx_file_path': self.tx_file_path,
            'num_targets': len(self.config.targets),
            'peak_snr_db': float(peak_snr),
            'time_bandwidth_product': float(tbp),
            'compression_gain_db': float(compression_gain),
            'range_resolution_m': float(range_res),
            'doppler_resolution_hz': float(doppler_res),
        }
        
        if self.config.stagger_pattern != StaggerPattern.NONE:
            stats.update({
                'avg_prf_hz': float(avg_prf),
                'min_prf_hz': float(min_prf),
                'max_prf_hz': float(max_prf),
                'prf_std_hz': float(std_prf),
                'stagger_pattern': self.config.stagger_pattern.value,
            })
        
        return stats
    
    def _create_full_tx_sequence(self, single_pulse_tx: np.ndarray) -> np.ndarray:
        """Create full TX sequence by repeating single pulse at each pulse time"""
        fs = self.config.sample_rate_hz
        
        # Calculate total time needed
        total_time = self.pulse_times[-1] + (2.0 / self.config.prf_hz)
        total_samples = int(np.ceil(total_time * fs))
        
        # Create sequence without the per-pulse window
        # Generate raw chirp without window for each pulse to avoid amplitude modulation
        full_tx = np.zeros(total_samples, dtype=np.complex64)
        
        # Get waveform parameters
        T = len(single_pulse_tx) / fs
        bw = self.config.bandwidth_hz
        k = bw / T
        
        for pulse_time in self.pulse_times:
            pulse_start = int(pulse_time * fs)
            pulse_end = min(pulse_start + len(single_pulse_tx), total_samples)
            pulse_len = pulse_end - pulse_start
            
            if pulse_len > 0:
                # Generate raw LFM chirp without window for this pulse
                t = np.arange(pulse_len, dtype=np.float64) / fs
                phase = np.pi * k * (t - T / 2.0) ** 2
                pulse = np.exp(1j * phase).astype(np.complex64)
                # Normalize to match single pulse level
                pulse /= np.sqrt(np.mean(np.abs(pulse) ** 2) + 1e-12).astype(np.float32)
                full_tx[pulse_start:pulse_end] = pulse
        
        return full_tx
    
    def _write_tx_file(self, tx_wfm: np.ndarray, is_full_sequence: bool):
        """Write separate TX waveform file"""
        # Determine TX filename
        main_path = Path(self.config.output_file)
        tx_path = main_path.parent / (main_path.stem + "_tx.crsd")
        self.tx_file_path = str(tx_path)
        
        # Create minimal RX channel (placeholder)
        placeholder_rx = np.array([[0.0+0.0j]], dtype=np.complex64)
        
        # Write as standard CRSD with TX waveform
        num_vectors = 1
        num_samples = 1
        num_channels = 1
        num_pulses = len(self.pulse_times)
        
        ch_ids = ["TX_CH"]
        tx_id = "TX1"
        sa_id = "TX_WFM"
        
        # Create XML
        xmltree = self._make_crsd_xml(
            num_vectors=num_vectors,
            num_samples=num_samples,
            num_pulses=num_pulses,
            tx_wfm_len=len(tx_wfm),
            ch_ids=ch_ids,
            tx_id=tx_id,
            tx_wfm_sa_id=sa_id,
        )
        
        # PVP/PPP
        rcv_time = np.array([0.0], dtype=np.float64)
        tx_time = self.pulse_times.copy()
        
        # Support array
        tx_sa = tx_wfm[None, :].astype(np.complex64)
        
        # KVPs
        kvps = {
            "CREATOR": "crsd-inspector",
            "TX_TYPE": "FULL_SEQUENCE" if is_full_sequence else "SINGLE_PULSE",
            "NUM_PULSES": str(num_pulses),
            "SAMPLE_RATE_HZ": f"{self.config.sample_rate_hz:.0f}",
            "PRF_HZ": f"{self.config.prf_hz:.1f}",
            "BANDWIDTH_HZ": f"{self.config.bandwidth_hz:.0f}",
            "STAGGER_PATTERN": self.config.stagger_pattern.value,
        }
        
        metadata = skcrsd.Metadata(
            xmltree=xmltree,
            file_header_part=skcrsd.FileHeaderPart(additional_kvps=kvps),
        )
        
        # Write
        tx_path.parent.mkdir(parents=True, exist_ok=True)
        with tx_path.open("wb") as f, skcrsd.Writer(f, metadata) as writer:
            writer.write_signal("TX_CH", placeholder_rx)
            
            pvps = np.zeros(num_vectors, dtype=skcrsd.get_pvp_dtype(xmltree))
            pvps["RcvTime"] = rcv_time
            writer.write_pvp("TX_CH", pvps)
            
            ppps = np.zeros(num_pulses, dtype=skcrsd.get_ppp_dtype(xmltree))
            ppps["TxTime"] = tx_time
            writer.write_ppp(tx_id, ppps)
            
            writer.write_support_array(sa_id, tx_sa)
            writer.done()


def generate_all_variants():
    """Generate all 4 TX variants for each of the 4 examples"""
    
    # Common TX waveform parameters
    common_tx = {
        'bandwidth_hz': 10e6,
        'carrier_freq_hz': 10e9,
        'sample_rate_hz': 100e6,
        'waveform_type': WaveformType.LFM,
        'verbose': True,
    }
    
    # Define the 4 base examples
    examples = [
        # Example 1: Uniform PRF, 1 target, 1 channel
        {
            'name': 'uniform_prf_1target_1ch',
            'config': SceneConfig(
                num_pulses=256,
                samples_per_pulse=512,
                prf_hz=1000.0,
                num_channels=1,
                snr_db=20.0,
                stagger_pattern=StaggerPattern.NONE,
                targets=[RadarTarget(range_m=3000.0, doppler_hz=50.0, rcs_dbsm=10.0)],
                **common_tx,
            )
        },
        # Example 2: Uniform PRF, 1 target, 4 channels
        {
            'name': 'uniform_prf_1target_4ch',
            'config': SceneConfig(
                num_pulses=256,
                samples_per_pulse=512,
                prf_hz=1000.0,
                num_channels=4,
                snr_db=20.0,
                stagger_pattern=StaggerPattern.NONE,
                targets=[RadarTarget(range_m=3000.0, doppler_hz=50.0, rcs_dbsm=10.0)],
                **common_tx,
            )
        },
        # Example 3: Fixed stagger, 3 targets, 1 channel
        {
            'name': 'fixed_stagger_3targets_1ch',
            'config': SceneConfig(
                num_pulses=300,
                samples_per_pulse=512,
                prf_hz=1200.0,
                num_channels=1,
                snr_db=18.0,
                stagger_pattern=StaggerPattern.THREE_STEP,
                stagger_ratio=0.15,
                targets=[
                    RadarTarget(range_m=2500.0, doppler_hz=80.0, rcs_dbsm=12.0),
                    RadarTarget(range_m=3500.0, doppler_hz=-40.0, rcs_dbsm=8.0),
                    RadarTarget(range_m=4500.0, doppler_hz=20.0, rcs_dbsm=10.0),
                ],
                **common_tx,
            )
        },
        # Example 4: Random stagger, 3 targets, 1 channel
        {
            'name': 'random_stagger_3targets_1ch',
            'config': SceneConfig(
                num_pulses=256,
                samples_per_pulse=512,
                prf_hz=1200.0,
                num_channels=1,
                snr_db=18.0,
                stagger_pattern=StaggerPattern.RANDOM,
                stagger_ratio=0.20,
                targets=[
                    RadarTarget(range_m=2000.0, doppler_hz=100.0, rcs_dbsm=15.0),
                    RadarTarget(range_m=3000.0, doppler_hz=-60.0, rcs_dbsm=10.0),
                    RadarTarget(range_m=5000.0, doppler_hz=30.0, rcs_dbsm=12.0),
                ],
                **common_tx,
            )
        },
    ]
    
    # TX modes
    tx_modes = [
        "embedded",
        "external",
    ]
    
    results = []
    base_dir = Path(__file__).parent
    
    print("=" * 80)
    print("GENERATING ALL TX VARIANTS")
    print("=" * 80)
    print(f"Examples: {len(examples)}")
    print(f"TX Modes: {len(tx_modes)}")
    print(f"Total Files: {len(examples) * len(tx_modes)} (+{len(examples)} external TX files)")
    print()
    
    for ex_idx, example in enumerate(examples, 1):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {ex_idx}/{len(examples)}: {example['name']}")
        print(f"{'='*80}")
        
        for mode_idx, tx_mode in enumerate(tx_modes, 1):
            print(f"\n  [{mode_idx}/{len(tx_modes)}] Mode: {tx_mode}")
            print("  " + "-" * 76)
            
            # Set output filename
            config = example['config']
            suffix = f"_{tx_mode}"
            config.output_file = str(base_dir / f"{example['name']}{suffix}.crsd")
            
            try:
                gen = TXVariantGenerator(config, tx_mode=tx_mode)
                stats = gen.generate()
                
                file_info = {
                    'example': example['name'],
                    'tx_mode': tx_mode,
                    'main_file': config.output_file,
                    'tx_file': stats.get('tx_file_path'),
                    'size_mb': stats['file_size_bytes'] / (1024**2),
                    'targets': stats['num_targets'],
                    'channels': config.num_channels,
                    'snr_db': stats['peak_snr_db'],
                }
                results.append(file_info)
                
                print(f"  ✅ Main: {Path(config.output_file).name} ({file_info['size_mb']:.1f} MB)")
                if file_info['tx_file']:
                    tx_size = Path(file_info['tx_file']).stat().st_size / (1024**2)
                    print(f"  ✅ TX:   {Path(file_info['tx_file']).name} ({tx_size:.1f} MB)")
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Summary
    print(f"\n\n{'='*80}")
    print("GENERATION SUMMARY")
    print(f"{'='*80}\n")
    
    by_example = {}
    for r in results:
        if r['example'] not in by_example:
            by_example[r['example']] = []
        by_example[r['example']].append(r)
    
    for ex_name, ex_results in by_example.items():
        print(f"\n{ex_name}:")
        for r in ex_results:
            main_name = Path(r['main_file']).name
            print(f"  • {r['tx_mode']:18s} → {main_name}")
            if r['tx_file']:
                tx_name = Path(r['tx_file']).name
                print(f"    {'':18s}   + {tx_name}")
    
    print(f"\n{'='*80}")
    print(f"✅ COMPLETE: {len(results)} files generated")
    print(f"{'='*80}")
    
    return results


if __name__ == "__main__":
    generate_all_variants()
