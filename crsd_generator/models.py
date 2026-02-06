"""
Data models for radar scene generation
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal
import numpy as np


class WaveformType(Enum):
    """Supported waveform types"""
    LFM = "lfm"  # Linear Frequency Modulation (chirp)
    BPSK = "bpsk"  # Binary Phase Shift Keying
    FRANK = "frank"  # Frank polyphase code
    COSTAS = "costas"  # Costas frequency hopping


class TargetModel(Enum):
    """Target scattering models"""
    POINT = "point"  # Point scatterer
    SWERLING_1 = "swerling1"  # Constant RCS, pulse-to-pulse
    SWERLING_2 = "swerling2"  # Rayleigh fluctuating, pulse-to-pulse
    SWERLING_3 = "swerling3"  # Constant RCS, scan-to-scan
    SWERLING_4 = "swerling4"  # Rayleigh fluctuating, scan-to-scan
    EXTENDED = "extended"  # Multiple scatterers


class ClutterModel(Enum):
    """Clutter statistical models"""
    RAYLEIGH = "rayleigh"  # Sea clutter (low sea state)
    WEIBULL = "weibull"  # Land/sea clutter
    K_DISTRIBUTION = "k_dist"  # High-resolution radar clutter
    LOG_NORMAL = "lognormal"  # Volume clutter


class DataFormat(Enum):
    """Signal data format"""
    PULSE_STACKED = "pulse_stacked"  # Pre-stacked: (num_pulses x samples_per_pulse)
    CONTINUOUS = "continuous"  # Continuous: (1 x total_samples) with embedded pulses


@dataclass
class RadarTarget:
    """Point target with motion"""
    
    # Position
    range_m: float  # Slant range in meters
    doppler_hz: float = 0.0  # Doppler frequency in Hz
    azimuth_deg: float = 0.0  # Azimuth angle
    elevation_deg: float = 0.0  # Elevation angle
    
    # RCS
    rcs_dbsm: float = 10.0  # Radar cross section in dBsm
    model: TargetModel = TargetModel.POINT
    
    # Motion (for extended scenarios)
    range_rate_ms: float = 0.0  # Range rate in m/s
    acceleration_ms2: float = 0.0  # Radial acceleration
    
    # Fluctuation (for Swerling models)
    correlation_time_s: float | None = None
    
    # Optional label
    label: str = ""
    
    def __post_init__(self):
        if not self.label:
            self.label = f"Target_R{self.range_m:.0f}m_D{self.doppler_hz:.0f}Hz"


@dataclass
class ClutterConfig:
    """Clutter generation configuration"""
    
    enabled: bool = True
    model: ClutterModel = ClutterModel.RAYLEIGH
    
    # Power levels
    cnr_db: float = 20.0  # Clutter-to-noise ratio in dB
    power_watts: float | None = None  # Alternative: absolute power
    
    # Spatial characteristics
    range_extent_m: tuple[float, float] = (0.0, 10000.0)  # Range extent
    correlation_range_m: float = 50.0  # Range correlation length
    correlation_doppler_hz: float = 10.0  # Doppler correlation
    
    # Statistical parameters
    shape_parameter: float = 1.0  # For Weibull, K-dist
    scale_parameter: float = 1.0
    
    # Spectral characteristics
    doppler_spread_hz: float = 50.0  # Doppler spectrum width
    doppler_center_hz: float = 0.0  # Mean Doppler shift


@dataclass
class SceneConfig:
    """Complete radar scene configuration"""
    
    # System parameters
    num_pulses: int = 128
    samples_per_pulse: int = 2048
    sample_rate_hz: float = 100e6  # 100 MHz
    prf_hz: float = 1000.0  # Pulse repetition frequency
    num_channels: int = 1  # Number of receiver channels
    
    # Waveform
    waveform_type: WaveformType = WaveformType.LFM
    bandwidth_hz: float = 10e6  # 10 MHz
    pulse_width_s: float | None = None  # Auto-calculated if None
    
    # RF parameters
    carrier_freq_hz: float = 10e9  # X-band (10 GHz)
    tx_power_watts: float = 1000.0
    
    # Noise
    noise_temp_k: float = 290.0  # System noise temperature
    snr_db: float = 20.0  # Target SNR (for reference target at 1000m)
    
    # Targets
    targets: list[RadarTarget] = field(default_factory=list)
    
    # Clutter
    clutter: ClutterConfig | None = None
    
    # Multipath/Propagation
    enable_multipath: bool = False
    multipath_delay_s: float = 1e-6
    multipath_attenuation_db: float = 10.0
    
    # Data format
    data_format: 'DataFormat' = None  # Will default to PULSE_STACKED in __post_init__
    
    # Output
    output_file: str = "synthetic_radar.crsd"
    verbose: bool = False
    
    def __post_init__(self):
        """Auto-calculate derived parameters"""
        if self.pulse_width_s is None:
            # Match pulse width to samples
            self.pulse_width_s = self.samples_per_pulse / self.sample_rate_hz
        
        if self.clutter is None:
            # Default: disable clutter
            self.clutter = ClutterConfig(enabled=False)
        
        if self.data_format is None:
            # Default to pulse-stacked for backward compatibility
            self.data_format = DataFormat.PULSE_STACKED


@dataclass
class GenerationReport:
    """Report from CRSD generation"""
    
    # File info
    output_path: str
    file_size_bytes: int
    
    # Scene summary
    num_targets: int
    num_clutter_patches: int
    
    # Signal characteristics
    peak_snr_db: float
    mean_snr_db: float
    cnr_db: float | None  # Clutter-to-noise ratio
    
    # Waveform stats
    time_bandwidth_product: float
    compression_gain_db: float
    range_resolution_m: float
    doppler_resolution_hz: float
    
    # Target details
    target_ranges_m: list[float] = field(default_factory=list)
    target_dopplers_hz: list[float] = field(default_factory=list)
    target_snrs_db: list[float] = field(default_factory=list)
    target_labels: list[str] = field(default_factory=list)
    
    # Timing
    generation_time_s: float = 0.0
    
    def summary(self) -> str:
        """Generate text summary"""
        lines = []
        lines.append("=" * 80)
        lines.append("CRSD GENERATION REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Output File: {self.output_path}")
        lines.append(f"File Size: {self.file_size_bytes / 1024 / 1024:.2f} MB")
        lines.append(f"Generation Time: {self.generation_time_s:.2f} s")
        lines.append("")
        lines.append(f"Targets: {self.num_targets}")
        if self.num_clutter_patches > 0:
            lines.append(f"Clutter Patches: {self.num_clutter_patches}")
            lines.append(f"CNR: {self.cnr_db:.1f} dB")
        lines.append("")
        lines.append(f"Peak SNR: {self.peak_snr_db:.1f} dB")
        lines.append(f"Mean SNR: {self.mean_snr_db:.1f} dB")
        lines.append("")
        lines.append(f"Range Resolution: {self.range_resolution_m:.2f} m")
        lines.append(f"Doppler Resolution: {self.doppler_resolution_hz:.2f} Hz")
        lines.append(f"Time-Bandwidth Product: {self.time_bandwidth_product:.1f}")
        lines.append(f"Compression Gain: {self.compression_gain_db:.1f} dB")
        lines.append("")
        
        if self.target_labels:
            lines.append("Target Details:")
            for i, (label, rng, dop, snr) in enumerate(zip(
                self.target_labels, self.target_ranges_m, 
                self.target_dopplers_hz, self.target_snrs_db
            ), 1):
                lines.append(f"  {i}. {label}")
                lines.append(f"     Range: {rng:.1f} m, Doppler: {dop:.1f} Hz, SNR: {snr:.1f} dB")
        
        lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines)
