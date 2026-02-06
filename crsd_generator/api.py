"""
CRSD Generator API
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Optional
import numpy as np
import lxml.etree as ET
import sarkit.crsd as skcrsd

from .models import (
    RadarTarget, ClutterConfig, SceneConfig, GenerationReport,
    WaveformType, TargetModel, ClutterModel, DataFormat
)

CRSD_NS = "http://api.nsgreg.nga.mil/schema/crsd/1.0"
NSMAP = {None: CRSD_NS}


class CRSDGenerator:
    """
    Synthetic CRSD file generator with comprehensive scene simulation
    """
    
    def __init__(self, config: SceneConfig):
        """Initialize generator with scene configuration"""
        self.config = config
        self.rng = np.random.default_rng(42)  # Deterministic by default
        self.pulse_times = None  # Store pulse times for continuous format
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        self.rng = np.random.default_rng(seed)
    
    def generate(self, verbose: bool | None = None) -> GenerationReport:
        """
        Generate CRSD file with configured scene
        
        Args:
            verbose: Override config verbose setting
        
        Returns:
            GenerationReport with statistics
        """
        start_time = time.time()
        verbose = verbose if verbose is not None else self.config.verbose
        
        if verbose:
            print(f"Generating CRSD: {self.config.output_file}")
            print(f"  Pulses: {self.config.num_pulses}, Samples: {self.config.samples_per_pulse}")
            print(f"  Channels: {self.config.num_channels}")
            print(f"  Targets: {len(self.config.targets)}")
        
        # Generate waveform
        if verbose:
            print("  - Generating TX waveform...")
        tx_wfm = self._generate_waveform()
        
        # Generate signals for all channels
        all_channels = []
        for ch_idx in range(self.config.num_channels):
            if verbose and self.config.num_channels > 1:
                print(f"  - Synthesizing channel {ch_idx+1}/{self.config.num_channels}...")
            
            # Generate signal for this channel (with slight variations per channel)
            # Use different random seed per channel for independent noise
            self.rng = np.random.default_rng(42 + ch_idx)
            
            # Generate signal (same for both formats, just reshaped differently)
            rx_signal, target_stats = self._generate_targets(tx_wfm)
            
            # Add clutter
            clutter_patches = 0
            if self.config.clutter and self.config.clutter.enabled:
                rx_signal, clutter_patches = self._add_clutter(rx_signal, tx_wfm)
            
            # Add noise
            rx_signal, noise_power = self._add_noise(rx_signal)
            
            all_channels.append(rx_signal)
        
        # Store first channel stats for report
        rx_signal_ch1 = all_channels[0]
        
        # Write CRSD file
        if verbose:
            print("  - Writing CRSD file...")
        file_size = self._write_crsd(all_channels, tx_wfm)
        
        # Compute statistics
        signal_power = np.mean(np.abs(rx_signal_ch1) ** 2)
        peak_snr = 10 * np.log10(max(target_stats["powers"]) / noise_power) if target_stats["powers"] else 0.0
        mean_snr = 10 * np.log10(np.mean(target_stats["powers"]) / noise_power) if target_stats["powers"] else 0.0
        
        # Waveform analysis
        T = len(tx_wfm) / self.config.sample_rate_hz
        tbp = T * self.config.bandwidth_hz
        compression_gain = 10 * np.log10(tbp)
        
        # Resolution
        c = 3e8
        range_res = c / (2 * self.config.bandwidth_hz)
        doppler_res = self.config.prf_hz / self.config.num_pulses
        
        # CNR
        cnr_db = None
        if self.config.clutter and self.config.clutter.enabled:
            cnr_db = self.config.clutter.cnr_db
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"  ✓ Complete in {elapsed:.2f}s")
        
        report = GenerationReport(
            output_path=self.config.output_file,
            file_size_bytes=file_size,
            num_targets=len(self.config.targets),
            num_clutter_patches=clutter_patches,
            peak_snr_db=float(peak_snr),
            mean_snr_db=float(mean_snr),
            cnr_db=cnr_db,
            time_bandwidth_product=float(tbp),
            compression_gain_db=float(compression_gain),
            range_resolution_m=float(range_res),
            doppler_resolution_hz=float(doppler_res),
            target_ranges_m=[t.range_m for t in self.config.targets],
            target_dopplers_hz=[t.doppler_hz for t in self.config.targets],
            target_snrs_db=target_stats["snrs_db"],
            target_labels=[t.label for t in self.config.targets],
            generation_time_s=elapsed,
        )
        
        return report
    
    def _generate_waveform(self) -> np.ndarray:
        """Generate TX waveform based on configuration
        
        The waveform duration is determined by bandwidth, not the receive window.
        For LFM, duration is typically 1-10x the inverse bandwidth.
        """
        fs = self.config.sample_rate_hz
        bw = self.config.bandwidth_hz
        
        # Waveform duration based on time-bandwidth product
        # For LFM, we want T*B >= 20 for good sidelobe performance
        # But not so long that it fills the entire receive window
        time_bandwidth_product = 50  # Good compromise
        T_wfm = time_bandwidth_product / bw
        num_wfm_samples = int(T_wfm * fs)
        
        if self.config.waveform_type == WaveformType.LFM:
            return self._lfm_chirp(num_wfm_samples, fs, bw)
        elif self.config.waveform_type == WaveformType.BPSK:
            return self._bpsk_code(num_wfm_samples)
        elif self.config.waveform_type == WaveformType.FRANK:
            return self._frank_code(num_wfm_samples)
        else:
            # Default to LFM
            return self._lfm_chirp(num_wfm_samples, fs, bw)
    
    def _lfm_chirp(self, num_samples: int, fs: float, bw: float) -> np.ndarray:
        """Generate LFM chirp"""
        t = np.arange(num_samples, dtype=np.float64) / fs
        T = num_samples / fs
        k = bw / T
        phase = np.pi * k * (t - T / 2.0) ** 2
        x = np.exp(1j * phase).astype(np.complex64)
        
        # Apply Hamming window
        w = np.hamming(num_samples).astype(np.float64)
        x = (x * w).astype(np.complex64)
        
        # Normalize
        x /= np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12).astype(np.float32)
        return x
    
    def _bpsk_code(self, num_samples: int) -> np.ndarray:
        """Generate BPSK coded waveform"""
        # Generate random binary code
        code_len = min(64, num_samples // 16)
        code = self.rng.choice([-1, 1], size=code_len)
        
        # Upsample
        samples_per_chip = num_samples // code_len
        waveform = np.repeat(code, samples_per_chip)
        
        # Pad if necessary
        if len(waveform) < num_samples:
            waveform = np.pad(waveform, (0, num_samples - len(waveform)))
        
        return waveform[:num_samples].astype(np.complex64)
    
    def _frank_code(self, num_samples: int) -> np.ndarray:
        """Generate Frank polyphase code"""
        # Frank code length must be perfect square
        M = int(np.sqrt(num_samples))
        if M * M < num_samples:
            M += 1
        
        # Generate Frank code phases
        phases = np.zeros(M * M)
        for i in range(M):
            for j in range(M):
                phases[i * M + j] = (2 * np.pi / M) * i * j
        
        waveform = np.exp(1j * phases[:num_samples]).astype(np.complex64)
        
        # Normalize
        waveform /= np.sqrt(np.mean(np.abs(waveform) ** 2) + 1e-12)
        return waveform
    
    def _generate_pulse_times(self) -> np.ndarray:
        """Generate uniform pulse timing"""
        P = self.config.num_pulses
        prf = self.config.prf_hz
        pri = 1.0 / prf
        return np.arange(P, dtype=np.float64) * pri
    
    def _generate_targets(self, tx_wfm: np.ndarray) -> tuple[np.ndarray, dict]:
        """Generate target returns - same for both formats, just reshape differently"""
        P = self.config.num_pulses
        N = self.config.samples_per_pulse
        prf = self.config.prf_hz
        fs = self.config.sample_rate_hz
        
        # Generate pulse-stacked data (P x N)
        rx = np.zeros((P, N), dtype=np.complex64)
        p = np.arange(P, dtype=np.float64)
        
        # Generate pulse times for metadata
        self.pulse_times = self._generate_pulse_times()
        
        target_powers = []
        
        for target in self.config.targets:
            # Convert range to sample delay
            c = 3e8
            two_way_delay = 2 * target.range_m / c
            delay_samples = int(two_way_delay * fs)
            
            if not (0 <= delay_samples < N):
                continue  # Target out of range window
            
            # Use SNR to set amplitude (simplified)
            amp = np.sqrt(10 ** (self.config.snr_db / 10)) * (1000.0 / target.range_m)
            
            # Apply target model
            if target.model == TargetModel.SWERLING_1:
                # Constant per CPI, fluctuates CPI-to-CPI
                amp *= np.sqrt(self.rng.exponential(1.0))
            elif target.model == TargetModel.SWERLING_2:
                # Fluctuates pulse-to-pulse
                amp *= np.sqrt(self.rng.exponential(1.0, size=P))[:, None]
            
            # Delayed waveform - place the TX waveform at the target's range bin
            delayed = np.zeros(N, dtype=np.complex64)
            wfm_len = len(tx_wfm)
            if delay_samples < N and delay_samples + wfm_len <= N:
                # Waveform fits completely
                delayed[delay_samples:delay_samples+wfm_len] = tx_wfm
            elif delay_samples < N:
                # Waveform is truncated at the end of the receive window
                samples_to_copy = min(wfm_len, N - delay_samples)
                delayed[delay_samples:delay_samples+samples_to_copy] = tx_wfm[:samples_to_copy]
            
            # Doppler modulation
            doppler_phase = np.exp(1j * 2.0 * np.pi * target.doppler_hz * (p / prf))
            
            # Combine
            if isinstance(amp, np.ndarray):
                target_signal = doppler_phase[:, None] * (amp * delayed[None, :])
            else:
                target_signal = doppler_phase[:, None] * (amp * delayed[None, :])
            
            rx += target_signal.astype(np.complex64)
            
            # Track power
            tgt_power = np.mean(np.abs(target_signal) ** 2)
            target_powers.append(float(tgt_power))
        
        # If continuous format, reshape to include inter-pulse gaps
        if self.config.data_format == DataFormat.CONTINUOUS:
            # Calculate samples per PRI (pulse repetition interval)
            fs = self.config.sample_rate_hz
            pri_samples = int(fs / self.config.prf_hz)
            
            # Create continuous signal with proper timing
            total_samples = P * pri_samples
            rx_continuous = np.zeros(total_samples, dtype=np.complex64)
            
            # Place each pulse at its proper time
            for p_idx in range(P):
                start_sample = p_idx * pri_samples
                end_sample = start_sample + N
                rx_continuous[start_sample:end_sample] = rx[p_idx, :]
            
            rx = rx_continuous[None, :]  # Reshape to (1, total_samples)
        
        return rx, {"powers": target_powers, "snrs_db": []}
    
    def _add_clutter(self, rx_signal: np.ndarray, tx_wfm: np.ndarray) -> tuple[np.ndarray, int]:
        """Add clutter to signal"""
        clutter_cfg = self.config.clutter
        if not clutter_cfg or not clutter_cfg.enabled:
            return rx_signal, 0
        
        P, N = rx_signal.shape
        
        # Generate clutter patches across range
        fs = self.config.sample_rate_hz
        c = 3e8
        
        min_range_samples = int((2 * clutter_cfg.range_extent_m[0] / c) * fs)
        max_range_samples = int((2 * clutter_cfg.range_extent_m[1] / c) * fs)
        min_range_samples = max(0, min_range_samples)
        max_range_samples = min(N, max_range_samples)
        
        if max_range_samples <= min_range_samples:
            return rx_signal, 0
        
        # Clutter power from CNR
        noise_power = (self.config.snr_db / 10) ** -1  # Approximate
        clutter_power = noise_power * (10 ** (clutter_cfg.cnr_db / 10))
        clutter_sigma = np.sqrt(clutter_power)
        
        # Generate clutter based on model
        if clutter_cfg.model == ClutterModel.RAYLEIGH:
            # Rayleigh amplitude, uniform phase
            amplitude = self.rng.rayleigh(clutter_sigma, size=(P, N))
            phase = self.rng.uniform(-np.pi, np.pi, size=(P, N))
            clutter = amplitude * np.exp(1j * phase)
        
        elif clutter_cfg.model == ClutterModel.WEIBULL:
            # Weibull distribution
            shape = clutter_cfg.shape_parameter
            scale = clutter_cfg.scale_parameter * clutter_sigma
            amplitude = self.rng.weibull(shape, size=(P, N)) * scale
            phase = self.rng.uniform(-np.pi, np.pi, size=(P, N))
            clutter = amplitude * np.exp(1j * phase)
        
        elif clutter_cfg.model == ClutterModel.K_DISTRIBUTION:
            # K-distribution (compound Gaussian)
            shape = clutter_cfg.shape_parameter
            texture = self.rng.gamma(shape, scale=1.0/shape, size=(P, N))
            speckle = self.rng.rayleigh(1.0, size=(P, N))
            amplitude = clutter_sigma * np.sqrt(texture * speckle)
            phase = self.rng.uniform(-np.pi, np.pi, size=(P, N))
            clutter = amplitude * np.exp(1j * phase)
        
        else:
            # Default to Rayleigh
            amplitude = self.rng.rayleigh(clutter_sigma, size=(P, N))
            phase = self.rng.uniform(-np.pi, np.pi, size=(P, N))
            clutter = amplitude * np.exp(1j * phase)
        
        # Apply only in clutter range extent
        clutter[:, :min_range_samples] = 0
        clutter[:, max_range_samples:] = 0
        
        # Add to signal
        rx_signal += clutter.astype(np.complex64)
        
        num_patches = max_range_samples - min_range_samples
        return rx_signal, num_patches
    
    def _add_noise(self, rx_signal: np.ndarray) -> tuple[np.ndarray, float]:
        """Add thermal noise"""
        # Noise power from system temperature and SNR
        # Simplified: use SNR to determine noise level
        signal_power = np.mean(np.abs(rx_signal) ** 2)
        
        # Target noise level
        snr_linear = 10 ** (self.config.snr_db / 10)
        noise_power = signal_power / snr_linear if signal_power > 0 else 1e-6
        noise_sigma = np.sqrt(noise_power / 2)  # Complex noise (I and Q)
        
        # Generate noise
        noise = noise_sigma * (
            self.rng.standard_normal(rx_signal.shape) +
            1j * self.rng.standard_normal(rx_signal.shape)
        )
        
        rx_signal += noise.astype(np.complex64)
        
        return rx_signal, float(noise_power)
    
    def _write_crsd(self, rx_signals: list[np.ndarray], tx_wfm: np.ndarray) -> int:
        """Write CRSD file with multiple channels"""
        # Get dimensions from first channel
        num_vectors, num_samples = rx_signals[0].shape
        num_channels = len(rx_signals)
        
        # For continuous format, vectors=1, for stacked vectors=num_pulses
        is_continuous = (self.config.data_format == DataFormat.CONTINUOUS)
        num_pulses = self.config.num_pulses
        
        # Create channel IDs
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
        
        # Build PVP/PPP
        if is_continuous:
            # Continuous: one PVP entry per channel, num_pulses PPP entries
            total_time = self.pulse_times[-1] if self.pulse_times is not None else 0.0
            rcv_time = np.array([total_time / 2.0], dtype=np.float64)
            tx_time = self.pulse_times.copy() if self.pulse_times is not None else np.arange(num_pulses, dtype=np.float64) / self.config.prf_hz
        else:
            # Stacked: P PVP entries per channel, P PPP entries
            prf = self.config.prf_hz
            tx_time = np.arange(num_pulses, dtype=np.float64) / prf
            rcv_time = tx_time.copy()
        
        # Prepare support array
        tx_sa = tx_wfm[None, :].astype(np.complex64)
        
        # Create metadata object
        metadata = skcrsd.Metadata(
            xmltree=xmltree,
            file_header_part=skcrsd.FileHeaderPart(
                additional_kvps={
                    "CREATOR": "crsd_generator",
                    "NUM_CHANNELS": str(num_channels),
                    "NUM_TARGETS": str(len(self.config.targets)),
                    "SNR_DB": f"{self.config.snr_db:.2f}",
                    "SAMPLE_RATE_HZ": f"{self.config.sample_rate_hz:.0f}",
                    "PRF_HZ": f"{self.config.prf_hz:.1f}",
                    "NUM_PULSES": str(num_pulses),
                    "DATA_FORMAT": self.config.data_format.value,
                }
            ),
        )
        
        # Write file
        out_path = Path(self.config.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with out_path.open("wb") as f, skcrsd.Writer(f, metadata) as writer:
            # Write all channels
            for ch_id, rx_signal in zip(ch_ids, rx_signals):
                writer.write_signal(ch_id, rx_signal)
                
                # Write PVP for this channel
                pvps = np.zeros(num_vectors, dtype=skcrsd.get_pvp_dtype(xmltree))
                pvps["RcvTime"] = rcv_time
                writer.write_pvp(ch_id, pvps)
            
            # Write PPP (pulse timing)
            ppps = np.zeros(num_pulses, dtype=skcrsd.get_ppp_dtype(xmltree))
            ppps["TxTime"] = tx_time
            writer.write_ppp(tx_id, ppps)
            
            # Write support array
            writer.write_support_array(sa_id, tx_sa)
            writer.done()
        
        return out_path.stat().st_size
    
    def _make_crsd_xml(
        self,
        *,
        num_vectors: int,
        num_samples: int,
        num_pulses: int,
        tx_wfm_len: int,
        ch_ids: list[str],
        tx_id: str,
        tx_wfm_sa_id: str,
    ) -> ET.ElementTree:
        """Build CRSD 1.0 XML metadata with multiple channels"""
        root = ET.Element(f"{{{CRSD_NS}}}CRSD", nsmap=NSMAP)
        
        # ProductInfo
        prod = ET.SubElement(root, f"{{{CRSD_NS}}}ProductInfo")
        ET.SubElement(prod, f"{{{CRSD_NS}}}Classification").text = "UNCLASSIFIED"
        ET.SubElement(prod, f"{{{CRSD_NS}}}ReleaseInfo").text = "UNRESTRICTED"
        
        # Data
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
        
        # Add a channel entry for each channel
        for ch_id in ch_ids:
            ch = ET.SubElement(rcv, f"{{{CRSD_NS}}}Channel")
            ET.SubElement(ch, f"{{{CRSD_NS}}}ChId").text = ch_id
            ET.SubElement(ch, f"{{{CRSD_NS}}}NumVectors").text = str(num_vectors)
            ET.SubElement(ch, f"{{{CRSD_NS}}}NumSamples").text = str(num_samples)
            ET.SubElement(ch, f"{{{CRSD_NS}}}SignalArrayByteOffset").text = "0"
            ET.SubElement(ch, f"{{{CRSD_NS}}}PVPArrayByteOffset").text = "0"
        
        # Support
        supp = ET.SubElement(data, f"{{{CRSD_NS}}}Support")
        dsa = ET.SubElement(supp, f"{{{CRSD_NS}}}SupportArray")
        ET.SubElement(dsa, f"{{{CRSD_NS}}}SAId").text = tx_wfm_sa_id
        ET.SubElement(dsa, f"{{{CRSD_NS}}}ArrayByteOffset").text = "0"
        ET.SubElement(dsa, f"{{{CRSD_NS}}}NumRows").text = "1"
        ET.SubElement(dsa, f"{{{CRSD_NS}}}NumCols").text = str(tx_wfm_len)  # Use actual TX waveform length
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
