"""
CLI entry point for CRSD Inspector
"""
import sys
import argparse
from pathlib import Path


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog="crsd-inspector",
        description="CRSD file analysis and generation toolkit"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate example CRSD files")
    gen_parser.add_argument(
        "--output-dir",
        type=str,
        default="./examples",
        help="Output directory for generated files (default: ./examples)"
    )
    
    # App command (default)
    app_parser = subparsers.add_parser("app", help="Launch Streamlit app (default)")
    
    args = parser.parse_args()
    
    # Default to app if no command specified
    if args.command is None or args.command == "app":
        launch_app()
    elif args.command == "generate":
        generate_examples(args.output_dir)


def launch_app():
    """Launch the Streamlit app"""
    from streamlit.web import cli as stcli
    import os
    
    # Get path to app.py
    app_path = Path(__file__).parent / "app.py"
    
    # Launch streamlit with the app
    sys.argv = ["streamlit", "run", str(app_path)]
    sys.exit(stcli.main())


def generate_examples(output_dir: str):
    """Generate example CRSD files"""
    from . import generate
    
    # Update output directory in the generator
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating example CRSD files to: {output_path.absolute()}")
    print("=" * 80)
    
    # Create scenes with updated output paths
    scenes = []
    for i, scene_config in enumerate(generate.create_example_scenes(), 1):
        scene_config.output_file = str(output_path / f"example_{i}.crsd")
        scenes.append(scene_config)
    
    # Generate each scene
    for i, scene in enumerate(scenes, 1):
        print(f"\n{'='*80}")
        print(f"Example {i} of {len(scenes)}")
        print(f"{'='*80}")
        
        try:
            generator = generate.CRSDGenerator(scene)
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


if __name__ == "__main__":
    main()
