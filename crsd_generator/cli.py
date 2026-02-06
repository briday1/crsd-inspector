"""
CRSD Generator Command Line Interface
"""

import sys
import click
from rich.console import Console
from rich.table import Table
from rich import box

from .models import (
    RadarTarget, ClutterConfig, SceneConfig,
    WaveformType, TargetModel, ClutterModel
)
from .api import CRSDGenerator

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="crsd-generate")
def cli():
    """
    CRSD Generator - Synthetic radar data generation
    
    Create realistic CRSD files with targets, clutter, and noise for testing
    radar signal processing algorithms.
    """
    pass


@cli.command()
@click.argument("output", type=click.Path())
@click.option("--pulses", "-p", default=128, help="Number of pulses")
@click.option("--samples", "-s", default=2048, help="Samples per pulse")
@click.option("--sample-rate", default=100.0, help="Sample rate in MHz")
@click.option("--prf", default=12.8, help="PRF in kHz")
@click.option("--bandwidth", "-b", default=10.0, help="Bandwidth in MHz")
@click.option("--waveform", type=click.Choice(["lfm", "bpsk", "frank"]), default="lfm", help="Waveform type")
@click.option("--snr", default=20.0, help="Target SNR in dB")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def simple(output: str, pulses: int, samples: int, sample_rate: float,
           prf: float, bandwidth: float, waveform: str, snr: float, verbose: bool):
    """
    Generate a simple CRSD file with default targets
    
    \b
    Example:
        crsd-generate simple test.crsd
        crsd-generate simple test.crsd --pulses 256 --samples 4096
        crsd-generate simple test.crsd --bandwidth 20 --snr 25
    """
    
    try:
        # Create default scene
        waveform_type = WaveformType(waveform)
        
        # Default targets
        targets = [
            RadarTarget(range_m=1500, doppler_hz=120, rcs_dbsm=10, label="Near Target"),
            RadarTarget(range_m=3000, doppler_hz=-250, rcs_dbsm=15, label="Far Target"),
            RadarTarget(range_m=5000, doppler_hz=450, rcs_dbsm=10, label="Weak Target"),
        ]
        
        config = SceneConfig(
            num_pulses=pulses,
            samples_per_pulse=samples,
            sample_rate_hz=sample_rate * 1e6,
            prf_hz=prf * 1e3,
            bandwidth_hz=bandwidth * 1e6,
            waveform_type=waveform_type,
            snr_db=snr,
            targets=targets,
            output_file=output,
            verbose=verbose,
        )
        
        # Generate
        if not verbose:
            console.print(f"\n[cyan]Generating:[/cyan] {output}")
        
        generator = CRSDGenerator(config)
        report = generator.generate()
        
        # Display report
        if not verbose:
            _display_report(report)
        else:
            console.print("\n" + report.summary())
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}", style="bold red")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.argument("output", type=click.Path())
@click.option("--pulses", "-p", default=256, help="Number of pulses")
@click.option("--samples", "-s", default=4096, help="Samples per pulse")
@click.option("--num-targets", "-n", default=10, help="Number of targets")
@click.option("--enable-clutter", is_flag=True, help="Enable clutter")
@click.option("--clutter-cnr", default=30.0, help="Clutter CNR in dB")
@click.option("--bandwidth", "-b", default=20.0, help="Bandwidth in MHz")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def complex_scene(output: str, pulses: int, samples: int, num_targets: int,
                  enable_clutter: bool, clutter_cnr: float, bandwidth: float, verbose: bool):
    """
    Generate a complex radar scene with multiple targets and clutter
    
    \b
    Example:
        crsd-generate complex test.crsd --num-targets 20
        crsd-generate complex test.crsd --enable-clutter --clutter-cnr 25
        crsd-generate complex test.crsd -n 50 --enable-clutter -v
    """
    
    try:
        import numpy as np
        
        # Generate random targets
        rng = np.random.default_rng(42)
        targets = []
        
        for i in range(num_targets):
            range_m = rng.uniform(500, 8000)
            doppler_hz = rng.uniform(-500, 500)
            rcs_dbsm = rng.uniform(10, 30)
            
            # Random target model
            model_choice = rng.choice([TargetModel.POINT, TargetModel.SWERLING_1, TargetModel.SWERLING_2])
            
            targets.append(RadarTarget(
                range_m=range_m,
                doppler_hz=doppler_hz,
                rcs_dbsm=rcs_dbsm,
                model=model_choice,
                label=f"Target_{i+1}"
            ))
        
        # Clutter configuration
        clutter_config = None
        if enable_clutter:
            clutter_config = ClutterConfig(
                enabled=True,
                model=ClutterModel.K_DISTRIBUTION,
                cnr_db=clutter_cnr,
                range_extent_m=(0, 10000),
                correlation_range_m=30.0,
                doppler_spread_hz=50.0,
            )
        
        config = SceneConfig(
            num_pulses=pulses,
            samples_per_pulse=samples,
            sample_rate_hz=100e6,
            prf_hz=10e3,
            bandwidth_hz=bandwidth * 1e6,
            waveform_type=WaveformType.LFM,
            snr_db=15.0,
            targets=targets,
            clutter=clutter_config,
            output_file=output,
            verbose=verbose,
        )
        
        # Generate
        if not verbose:
            console.print(f"\n[cyan]Generating complex scene:[/cyan] {output}")
            console.print(f"  Targets: {num_targets}, Clutter: {'enabled' if enable_clutter else 'disabled'}")
        
        generator = CRSDGenerator(config)
        report = generator.generate()
        
        # Display report
        if not verbose:
            _display_report(report)
        else:
            console.print("\n" + report.summary())
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}", style="bold red")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.argument("output", type=click.Path())
@click.option("--config", "-c", type=click.Path(exists=True), help="JSON config file")
def custom(output: str, config: str):
    """
    Generate CRSD from a custom JSON configuration file
    
    \b
    Example:
        crsd-generate custom test.crsd --config scene.json
    """
    
    try:
        import json
        
        if not config:
            console.print("[red]Error:[/red] --config is required", style="bold red")
            sys.exit(1)
        
        # Load config
        with open(config) as f:
            cfg_dict = json.load(f)
        
        # Parse targets
        targets = []
        for tgt_dict in cfg_dict.get("targets", []):
            targets.append(RadarTarget(
                range_m=tgt_dict["range_m"],
                doppler_hz=tgt_dict.get("doppler_hz", 0.0),
                rcs_dbsm=tgt_dict.get("rcs_dbsm", 10.0),
                model=TargetModel(tgt_dict.get("model", "point")),
                label=tgt_dict.get("label", ""),
            ))
        
        # Parse clutter
        clutter_cfg = None
        if "clutter" in cfg_dict and cfg_dict["clutter"].get("enabled"):
            clutter_cfg = ClutterConfig(
                enabled=True,
                model=ClutterModel(cfg_dict["clutter"].get("model", "rayleigh")),
                cnr_db=cfg_dict["clutter"].get("cnr_db", 20.0),
            )
        
        # Build scene config
        scene_cfg = SceneConfig(
            num_pulses=cfg_dict.get("num_pulses", 128),
            samples_per_pulse=cfg_dict.get("samples_per_pulse", 2048),
            sample_rate_hz=cfg_dict.get("sample_rate_hz", 100e6),
            prf_hz=cfg_dict.get("prf_hz", 10e3),
            bandwidth_hz=cfg_dict.get("bandwidth_hz", 10e6),
            waveform_type=WaveformType(cfg_dict.get("waveform_type", "lfm")),
            snr_db=cfg_dict.get("snr_db", 20.0),
            targets=targets,
            clutter=clutter_cfg,
            output_file=output,
            verbose=cfg_dict.get("verbose", False),
        )
        
        # Generate
        console.print(f"\n[cyan]Generating from config:[/cyan] {config}")
        
        generator = CRSDGenerator(scene_cfg)
        report = generator.generate()
        
        console.print("\n" + report.summary())
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}", style="bold red")
        console.print_exception()
        sys.exit(1)


def _display_report(report):
    """Display generation report with rich formatting"""
    
    # Summary table
    summary_table = Table(title="Generation Summary", box=box.ROUNDED, show_header=False)
    summary_table.add_column("Property", style="cyan")
    summary_table.add_column("Value", style="white")
    
    summary_table.add_row("Output File", report.output_path)
    summary_table.add_row("File Size", f"{report.file_size_bytes / 1024 / 1024:.2f} MB")
    summary_table.add_row("Generation Time", f"{report.generation_time_s:.2f} s")
    summary_table.add_row("Targets", str(report.num_targets))
    if report.num_clutter_patches > 0:
        summary_table.add_row("Clutter Patches", str(report.num_clutter_patches))
    
    console.print()
    console.print(summary_table)
    
    # Performance table
    perf_table = Table(title="Signal Characteristics", box=box.ROUNDED, show_header=False)
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", style="white")
    
    perf_table.add_row("Peak SNR", f"{report.peak_snr_db:.1f} dB")
    perf_table.add_row("Mean SNR", f"{report.mean_snr_db:.1f} dB")
    if report.cnr_db:
        perf_table.add_row("CNR", f"{report.cnr_db:.1f} dB")
    perf_table.add_row("Range Resolution", f"{report.range_resolution_m:.2f} m")
    perf_table.add_row("Doppler Resolution", f"{report.doppler_resolution_hz:.2f} Hz")
    perf_table.add_row("Time-Bandwidth Product", f"{report.time_bandwidth_product:.1f}")
    perf_table.add_row("Compression Gain", f"{report.compression_gain_db:.1f} dB")
    
    console.print()
    console.print(perf_table)
    
    # Targets (if not too many)
    if report.num_targets <= 10 and report.target_labels:
        targets_table = Table(title="Target Details", box=box.ROUNDED)
        targets_table.add_column("#", style="dim")
        targets_table.add_column("Label", style="cyan")
        targets_table.add_column("Range (m)", style="white")
        targets_table.add_column("Doppler (Hz)", style="white")
        targets_table.add_column("SNR (dB)", style="green")
        
        for i, (label, rng, dop, snr) in enumerate(zip(
            report.target_labels, report.target_ranges_m,
            report.target_dopplers_hz, report.target_snrs_db
        ), 1):
            targets_table.add_row(
                str(i),
                label,
                f"{rng:.1f}",
                f"{dop:+.1f}",
                f"{snr:.1f}"
            )
        
        console.print()
        console.print(targets_table)
    
    console.print()


if __name__ == "__main__":
    cli()
