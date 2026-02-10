#!/usr/bin/env python3
"""
Comprehensive CRSD Generator - Create realistic synthetic CRSD files

This script generates multi-channel CRSD files with:
- Proper TX waveform (LFM chirp) stored in support arrays
- Multiple point targets with realistic radar returns
- Thermal noise
- Complete metadata for proper range-Doppler processing
- Multi-channel support

Based on test-dagex CRSD generator capabilities.
"""

import numpy as np
import lxml.etree as ET
import sarkit.crsd as skcrsd
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum


# CRSD namespace
CRSD_NS = "http://api.nsgreg.nga.mil/schema/crsd/1.0"
NSMAP = {None: CRSD_NS}


class WaveformType(Enum):
    """Supported waveform types"""
    LFM = "lfm"  # Linear Frequency Modulation (chirp)
    BPSK = "bpsk"  # Binary Phase Shift Keying


class StaggerPattern(Enum):
    """PRF stagger patterns"""
    NONE = "none"  # Uniform PRF
    TWO_STEP = "2-step"  # Alternates between two PRIs
    THREE_STEP = "3-step"  # Cycles through three PRIs
    RANDOM = "random"  # Random PRI variation
    JITTERED = "jittered"  # Small random jitter around nominal


@dataclass
class RadarTarget:
    """Point target definition"""
    range_m: float  # Slant range in meters
    doppler_hz: float = 0.0  # Doppler frequency in Hz
    rcs_dbsm: float = 0.0  # Radar cross section in dBsm
    label: str = ""
    
    def __post_init__(self):
        if not self.label:
            self.label = f"T_R{self.range_m:.0f}m_D{self.doppler_hz:.0f}Hz"


@dataclass
class SceneConfig:
    """Radar scene configuration"""
    # System parameters
    num_pulses: int = 256
    samples_per_pulse: int = 512
    sample_rate_hz: float = 100e6  # 100 MHz
    prf_hz: float = 1000.0  # Nominal PRF
    num_channels: int = 1
    
    # PRF Stagger (new)
    stagger_pattern: StaggerPattern = StaggerPattern.NONE
    stagger_ratio: float = 0.15  # +/- variation for stagger (15% default)
    
    # Waveform
    waveform_type: WaveformType = WaveformType.LFM
    bandwidth_hz: float = 10e6  # 10 MHz
    carrier_freq_hz: float = 10e9  # X-band
    
    # Noise
    snr_db: float = 20.0  # Target SNR
    
    # Targets
    targets: list[RadarTarget] = field(default_factory=list)
    
    # Output
    output_file: str = "synthetic.crsd"
    verbose: bool = False


class CRSDGenerator:
    """Generate realistic CRSD files with multiple channels and TX waveforms"""
    
    def __init__(self, config: SceneConfig):
        self.config = config
        self.rng = np.random.default_rng(42)
        self.pulse_times = None  # Will store computed pulse start times
    
    def generate(self) -> dict:
        """Generate CRSD file and return statistics"""
        if self.config.verbose:
            print(f"Generating CRSD: {self.config.output_file}")
            print(f"  Pulses: {self.config.num_pulses}, Samples: {self.config.samples_per_pulse}")
            print(f"  Channels: {self.config.num_channels}")
            print(f"  Targets: {len(self.config.targets)}")
        
        # Generate TX waveform
        if self.config.verbose:
            print("  - Generating TX waveform...")
        tx_wfm = self._generate_waveform()
        
        # Generate signals for all channels
        all_channels = []
        for ch_idx in range(self.config.num_channels):
            if self.config.verbose and self.config.num_channels > 1:
                print(f"  - Synthesizing channel {ch_idx+1}/{self.config.num_channels}...")
            
            # Independent noise per channel
            self.rng = np.random.default_rng(42 + ch_idx)
            rx_signal, target_powers = self._generate_targets(tx_wfm)
            rx_signal, noise_power = self._add_noise(rx_signal)
            all_channels.append(rx_signal)
        
        # Write CRSD file
        if self.config.verbose:
            print("  - Writing CRSD file...")
        file_size = self._write_crsd(all_channels, tx_wfm)
        
        # Compute statistics
        signal_power = np.mean(np.abs(all_channels[0]) ** 2)
        peak_snr = 10 * np.log10(max(target_powers) / noise_power) if target_powers else 0.0
        
        # Waveform analysis
        T = len(tx_wfm) / self.config.sample_rate_hz
        tbp = T * self.config.bandwidth_hz
        compression_gain = 10 * np.log10(tbp)
        
        # Resolution
        c = 3e8
        range_res = c / (2 * self.config.bandwidth_hz)
        doppler_res = self.config.prf_hz / self.config.num_pulses
        
        # PRF statistics (from generated pulse times)
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
            print(f"  ✓ Complete")
        
        stats = {
            'file_size_bytes': file_size,
            'num_targets': len(self.config.targets),
            'peak_snr_db': float(peak_snr),
            'time_bandwidth_product': float(tbp),
            'compression_gain_db': float(compression_gain),
            'range_resolution_m': float(range_res),
            'doppler_resolution_hz': float(doppler_res),
        }
        
        # Add PRF stagger stats if applicable
        if self.config.stagger_pattern != StaggerPattern.NONE:
            stats.update({
                'avg_prf_hz': float(avg_prf),
                'min_prf_hz': float(min_prf),
                'max_prf_hz': float(max_prf),
                'prf_std_hz': float(std_prf),
                'stagger_pattern': self.config.stagger_pattern.value,
            })
        
        return stats
    
    def _generate_pulse_times(self) -> np.ndarray:
        """Generate pulse timing with optional stagger"""
        P = self.config.num_pulses
        nominal_prf = self.config.prf_hz
        nominal_pri = 1.0 / nominal_prf
        
        if self.config.stagger_pattern == StaggerPattern.NONE:
            # Uniform PRF
            pulse_times = np.arange(P, dtype=np.float64) * nominal_pri
            
        elif self.config.stagger_pattern == StaggerPattern.TWO_STEP:
            # Alternate between two PRIs
            pri_high = nominal_pri * (1 + self.config.stagger_ratio)
            pri_low = nominal_pri * (1 - self.config.stagger_ratio)
            pulse_times = np.zeros(P, dtype=np.float64)
            for i in range(1, P):
                pulse_times[i] = pulse_times[i-1] + (pri_high if i % 2 else pri_low)
                
        elif self.config.stagger_pattern == StaggerPattern.THREE_STEP:
            # Cycle through three PRIs
            pris = [
                nominal_pri * (1 - self.config.stagger_ratio),
                nominal_pri,
                nominal_pri * (1 + self.config.stagger_ratio)
            ]
            pulse_times = np.zeros(P, dtype=np.float64)
            for i in range(1, P):
                pulse_times[i] = pulse_times[i-1] + pris[i % 3]
                
        elif self.config.stagger_pattern == StaggerPattern.RANDOM:
            # Random PRI variation
            pris = nominal_pri * (1 + self.config.stagger_ratio * self.rng.uniform(-1, 1, P-1))
            pulse_times = np.concatenate([[0], np.cumsum(pris)])
            
        elif self.config.stagger_pattern == StaggerPattern.JITTERED:
            # Small jitter around nominal
            pris = nominal_pri * (1 + self.config.stagger_ratio * 0.1 * self.rng.standard_normal(P-1))
            pulse_times = np.concatenate([[0], np.cumsum(pris)])
            
        else:
            pulse_times = np.arange(P, dtype=np.float64) * nominal_pri
            
        return pulse_times
    
    def _generate_waveform(self) -> np.ndarray:
        """Generate TX waveform (LFM chirp)"""
        fs = self.config.sample_rate_hz
        bw = self.config.bandwidth_hz
        
        # Time-bandwidth product of 50 for good sidelobe performance
        time_bandwidth_product = 50
        T_wfm = time_bandwidth_product / bw
        num_wfm_samples = int(T_wfm * fs)
        
        if self.config.waveform_type == WaveformType.LFM:
            # LFM chirp
            t = np.arange(num_wfm_samples, dtype=np.float64) / fs
            T = num_wfm_samples / fs
            k = bw / T
            phase = np.pi * k * (t - T / 2.0) ** 2
            x = np.exp(1j * phase).astype(np.complex64)
            
            # Apply Hamming window
            w = np.hamming(num_wfm_samples).astype(np.float64)
            x = (x * w).astype(np.complex64)
            
            # Normalize
            x /= np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12).astype(np.float32)
            return x
        
        else:  # BPSK
            code_len = min(64, num_wfm_samples // 16)
            code = self.rng.choice([-1, 1], size=code_len)
            samples_per_chip = num_wfm_samples // code_len
            waveform = np.repeat(code, samples_per_chip)
            if len(waveform) < num_wfm_samples:
                waveform = np.pad(waveform, (0, num_wfm_samples - len(waveform)))
            return waveform[:num_wfm_samples].astype(np.complex64)
    
    def _generate_targets(self, tx_wfm: np.ndarray) -> tuple[np.ndarray, list]:
        """Generate target returns in continuous format (not pre-stacked)"""
        P = self.config.num_pulses
        N = self.config.samples_per_pulse
        fs = self.config.sample_rate_hz
        
        # Generate pulse times (with optional stagger)
        self.pulse_times = self._generate_pulse_times()
        
        # Calculate total continuous signal length
        # Total time span + some buffer for last pulse
        total_time = self.pulse_times[-1] + (2.0 / self.config.prf_hz)
        total_samples = int(np.ceil(total_time * fs))
        
        # Create continuous signal
        rx_continuous = np.zeros(total_samples, dtype=np.complex64)
        
        target_powers = []
        
        for target in self.config.targets:
            # Convert range to sample delay
            c = 3e8
            two_way_delay = 2 * target.range_m / c
            delay_samples = int(two_way_delay * fs)
            
            # Amplitude from range (simplified radar equation)
            amp = np.sqrt(10 ** (self.config.snr_db / 10)) * (1000.0 / target.range_m)
            
            # For each pulse, place waveform at appropriate time + range delay
            for pulse_idx, pulse_time in enumerate(self.pulse_times):
                # Pulse start sample in continuous data
                pulse_start_sample = int(pulse_time * fs)
                
                # Target echo sample = pulse start + range delay
                echo_sample = pulse_start_sample + delay_samples
                
                if echo_sample < 0 or echo_sample >= total_samples:
                    continue
                
                # Apply Doppler modulation for this pulse
                doppler_phase = np.exp(1j * 2.0 * np.pi * target.doppler_hz * pulse_time)
                
                # Place waveform
                wfm_len = len(tx_wfm)
                end_sample = min(echo_sample + wfm_len, total_samples)
                copy_len = end_sample - echo_sample
                
                if copy_len > 0:
                    rx_continuous[echo_sample:end_sample] += (amp * doppler_phase * tx_wfm[:copy_len]).astype(np.complex64)
            
            # Track power
            tgt_power = np.mean(np.abs(rx_continuous) ** 2)
            target_powers.append(float(tgt_power))
        
        # Reshape into (1, total_samples) to represent continuous data
        # The workflow will handle pulse extraction
        rx_reshaped = rx_continuous[None, :]  # Shape: (1, total_samples)
        
        return rx_reshaped, target_powers
    
    def _add_noise(self, rx_signal: np.ndarray) -> tuple[np.ndarray, float]:
        """Add thermal noise"""
        signal_power = np.mean(np.abs(rx_signal) ** 2)
        snr_linear = 10 ** (self.config.snr_db / 10)
        noise_power = signal_power / snr_linear if signal_power > 0 else 1e-6
        noise_sigma = np.sqrt(noise_power / 2)
        
        noise = noise_sigma * (
            self.rng.standard_normal(rx_signal.shape) +
            1j * self.rng.standard_normal(rx_signal.shape)
        )
        
        rx_signal += noise.astype(np.complex64)
        return rx_signal, float(noise_power)
    
    def _write_crsd(self, rx_signals: list[np.ndarray], tx_wfm: np.ndarray) -> int:
        """Write CRSD file with continuous data and pulse timing"""
        # rx_signals shape: (1, total_samples) - continuous data
        num_vectors = 1  # One long vector per channel
        num_samples = rx_signals[0].shape[1]  # Total continuous samples
        num_channels = len(rx_signals)
        num_pulses = len(self.pulse_times)  # Number of pulses for PPP array
        
        # Create channel/waveform IDs
        ch_ids = [f"CHAN{i+1}" for i in range(num_channels)]
        tx_id = "TX1"
        sa_id = "TX_WFM"
        
        # Create XML metadata
        xmltree = self._make_crsd_xml(
            num_vectors=num_vectors,
            num_samples=num_samples,
            num_pulses=num_pulses,
            tx_wfm_len=len(tx_wfm),
            ch_ids=ch_ids,
            tx_id=tx_id,
            tx_wfm_sa_id=sa_id,
        )
        
        # PVP: One entry for the single continuous vector per channel
        total_time = self.pulse_times[-1] if len(self.pulse_times) > 0 else 0.0
        rcv_time = np.array([total_time / 2.0], dtype=np.float64)
        
        # PPP: Actual pulse times from generator
        tx_time = self.pulse_times.copy()
        
        # Prepare support array (TX waveform)
        tx_sa = tx_wfm[None, :].astype(np.complex64)
        
        # Get ground truth KVPs (allow subclasses to customize)
        if hasattr(self, '_get_ground_truth_kvps'):
            kvps = self._get_ground_truth_kvps()
        else:
            kvps = {
                "CREATOR": "crsd-inspector",
                "NUM_CHANNELS": str(num_channels),
                "NUM_TARGETS": str(len(self.config.targets)),
                "SNR_DB": f"{self.config.snr_db:.2f}",
                "SAMPLE_RATE_HZ": f"{self.config.sample_rate_hz:.0f}",
                "PRF_HZ": f"{self.config.prf_hz:.1f}",
                "BANDWIDTH_HZ": f"{self.config.bandwidth_hz:.0f}",
                "STAGGER_PATTERN": self.config.stagger_pattern.value,
                "NUM_PULSES": str(num_pulses),
            }
        
        # Create metadata object
        metadata = skcrsd.Metadata(
            xmltree=xmltree,
            file_header_part=skcrsd.FileHeaderPart(
                additional_kvps=kvps
            ),
        )
        
        # Write file
        out_path = Path(self.config.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with out_path.open("wb") as f, skcrsd.Writer(f, metadata) as writer:
            # Write all channels
            for ch_id, rx_signal in zip(ch_ids, rx_signals):
                writer.write_signal(ch_id, rx_signal)
                
                # Write PVP for this channel (one entry)
                pvps = np.zeros(num_vectors, dtype=skcrsd.get_pvp_dtype(xmltree))
                pvps["RcvTime"] = rcv_time
                writer.write_pvp(ch_id, pvps)
            
            # Write PPP (pulse timing array)
            ppps = np.zeros(num_pulses, dtype=skcrsd.get_ppp_dtype(xmltree))
            ppps["TxTime"] = tx_time
            writer.write_ppp(tx_id, ppps)
            
            # Write support array (TX waveform)
            writer.write_support_array(sa_id, tx_sa)
            writer.done()
        
        return out_path.stat().st_size
    
    def _make_crsd_xml(
        self, *, num_vectors: int, num_samples: int, num_pulses: int,
        tx_wfm_len: int, ch_ids: list[str], tx_id: str, tx_wfm_sa_id: str
    ) -> ET.ElementTree:
        """Build CRSD 1.0 XML metadata for continuous data with pulse timing"""
        root = ET.Element(f"{{{CRSD_NS}}}CRSD", nsmap=NSMAP)
        
        # ProductInfo
        prod = ET.SubElement(root, f"{{{CRSD_NS}}}ProductInfo")
        ET.SubElement(prod, f"{{{CRSD_NS}}}Classification").text = "UNCLASSIFIED"
        
        # Data section
        data = ET.SubElement(root, f"{{{CRSD_NS}}}Data")
        
        # Transmit
        tx = ET.SubElement(data, f"{{{CRSD_NS}}}Transmit")
        ET.SubElement(tx, f"{{{CRSD_NS}}}NumBytesPPP").text = "8"
        seq = ET.SubElement(tx, f"{{{CRSD_NS}}}TxSequence")
        ET.SubElement(seq, f"{{{CRSD_NS}}}TxId").text = tx_id
        ET.SubElement(seq, f"{{{CRSD_NS}}}NumPulses").text = str(num_pulses)
        ET.SubElement(seq, f"{{{CRSD_NS}}}PPPArrayByteOffset").text = "0"
        
        # Receive (multiple channels)
        rcv = ET.SubElement(data, f"{{{CRSD_NS}}}Receive")
        ET.SubElement(rcv, f"{{{CRSD_NS}}}SignalArrayFormat").text = "CF8"
        ET.SubElement(rcv, f"{{{CRSD_NS}}}NumBytesPVP").text = "8"
        
        for ch_id in ch_ids:
            ch = ET.SubElement(rcv, f"{{{CRSD_NS}}}Channel")
            ET.SubElement(ch, f"{{{CRSD_NS}}}ChId").text = ch_id
            ET.SubElement(ch, f"{{{CRSD_NS}}}NumVectors").text = str(num_vectors)  # One continuous vector
            ET.SubElement(ch, f"{{{CRSD_NS}}}NumSamples").text = str(num_samples)  # Total continuous samples
            ET.SubElement(ch, f"{{{CRSD_NS}}}SignalArrayByteOffset").text = "0"
            ET.SubElement(ch, f"{{{CRSD_NS}}}PVPArrayByteOffset").text = "0"
        
        # Support array (TX waveform)
        supp = ET.SubElement(data, f"{{{CRSD_NS}}}Support")
        dsa = ET.SubElement(supp, f"{{{CRSD_NS}}}SupportArray")
        ET.SubElement(dsa, f"{{{CRSD_NS}}}SAId").text = tx_wfm_sa_id
        ET.SubElement(dsa, f"{{{CRSD_NS}}}ArrayByteOffset").text = "0"
        ET.SubElement(dsa, f"{{{CRSD_NS}}}NumRows").text = "1"
        ET.SubElement(dsa, f"{{{CRSD_NS}}}NumCols").text = str(tx_wfm_len)
        ET.SubElement(dsa, f"{{{CRSD_NS}}}BytesPerElement").text = "8"
        
        # Root-level SupportArray descriptor
        sa_desc_root = ET.SubElement(root, f"{{{CRSD_NS}}}SupportArray")
        sa_desc = ET.SubElement(sa_desc_root, f"{{{CRSD_NS}}}AddedSupportArray")
        ET.SubElement(sa_desc, f"{{{CRSD_NS}}}Identifier").text = tx_wfm_sa_id
        ET.SubElement(sa_desc, f"{{{CRSD_NS}}}ElementFormat").text = "CF8"
        
        # Root-level PVP descriptor
        pvp = ET.SubElement(root, f"{{{CRSD_NS}}}PVP")
        rcvtime = ET.SubElement(pvp, f"{{{CRSD_NS}}}RcvTime")
        ET.SubElement(rcvtime, f"{{{CRSD_NS}}}Format").text = "F8"
        ET.SubElement(rcvtime, f"{{{CRSD_NS}}}Offset").text = "0"
        
        # Root-level PPP descriptor
        ppp = ET.SubElement(root, f"{{{CRSD_NS}}}PPP")
        txtime = ET.SubElement(ppp, f"{{{CRSD_NS}}}TxTime")
        ET.SubElement(txtime, f"{{{CRSD_NS}}}Format").text = "F8"
        ET.SubElement(txtime, f"{{{CRSD_NS}}}Offset").text = "0"
        
        return ET.ElementTree(root)


def create_example_scenes():
    """Create 3 example CRSD files with different characteristics"""
    
    # Example 1: Simple - 3 targets, single channel
    scene1 = SceneConfig(
        num_pulses=256,
        samples_per_pulse=512,
        sample_rate_hz=100e6,
        prf_hz=1000.0,
        num_channels=1,
        bandwidth_hz=10e6,
        snr_db=20.0,
        targets=[
            RadarTarget(range_m=2000.0, doppler_hz=50.0, rcs_dbsm=10.0, label="Vehicle"),
            RadarTarget(range_m=3500.0, doppler_hz=-30.0, rcs_dbsm=5.0, label="UAV"),
            RadarTarget(range_m=5000.0, doppler_hz=0.0, rcs_dbsm=15.0, label="Building"),
        ],
        output_file=str(Path(__file__).parent / "example_1.crsd"),
        verbose=True,
    )
    
    # Example 2: Complex - 5 targets, single channel
    scene2 = SceneConfig(
        num_pulses=256,
        samples_per_pulse=512,
        sample_rate_hz=100e6,
        prf_hz=1000.0,
        num_channels=1,
        bandwidth_hz=10e6,
        snr_db=18.0,
        targets=[
            RadarTarget(range_m=1500.0, doppler_hz=100.0, rcs_dbsm=8.0, label="Fast Car"),
            RadarTarget(range_m=2800.0, doppler_hz=-50.0, rcs_dbsm=12.0, label="Truck"),
            RadarTarget(range_m=3200.0, doppler_hz=20.0, rcs_dbsm=6.0, label="Motorcycle"),
            RadarTarget(range_m=4500.0, doppler_hz=-80.0, rcs_dbsm=10.0, label="Helicopter"),
            RadarTarget(range_m=6000.0, doppler_hz=5.0, rcs_dbsm=18.0, label="Tower"),
        ],
        output_file=str(Path(__file__).parent / "example_2.crsd"),
        verbose=True,
    )
    
    # Example 3: Multi-channel - 2 targets, 2 channels
    scene3 = SceneConfig(
        num_pulses=256,
        samples_per_pulse=512,
        sample_rate_hz=100e6,
        prf_hz=1000.0,
        num_channels=2,
        bandwidth_hz=10e6,
        snr_db=22.0,
        targets=[
            RadarTarget(range_m=3000.0, doppler_hz=-40.0, rcs_dbsm=14.0, label="Aircraft"),
            RadarTarget(range_m=4000.0, doppler_hz=60.0, rcs_dbsm=10.0, label="Boat"),
        ],
        output_file=str(Path(__file__).parent / "example_3.crsd"),
        verbose=True,
    )
    
    return [scene1, scene2, scene3]


if __name__ == "__main__":
    print("=" * 80)
    print("CRSD GENERATOR - Creating Example Files")
    print("=" * 80)
    print()
    
    scenes = create_example_scenes()
    
    for i, scene in enumerate(scenes, 1):
        print(f"\n{'='*80}")
        print(f"Example {i} of {len(scenes)}")
        print(f"{'='*80}")
        
        try:
            generator = CRSDGenerator(scene)
            stats = generator.generate()
            
            print(f"\n  File: {scene.output_file}")
            print(f"  Size: {stats['file_size_bytes'] / 1024:.1f} KB")
            print(f"  Targets: {stats['num_targets']}")
            print(f"  Channels: {scene.num_channels}")
            print(f"  Peak SNR: {stats['peak_snr_db']:.1f} dB")
            print(f"  Range Resolution: {stats['range_resolution_m']:.2f} m")
            print(f"  Doppler Resolution: {stats['doppler_resolution_hz']:.2f} Hz")
            print(f"  Compression Gain: {stats['compression_gain_db']:.1f} dB")
            print(f"  Time-Bandwidth Product: {stats['time_bandwidth_product']:.1f}")
            print(f"\n  ✅ Success")
            
        except Exception as e:
            print(f"\n  ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("GENERATION COMPLETE")
    print(f"{'='*80}")
