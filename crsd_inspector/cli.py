"""
CRSD Inspector Command Line Interface
"""

import os
import sys
import glob
import json
import click
import importlib.util
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def discover_workflows():
    """Discover available workflow modules"""
    workflows = {}
    workflows_dir = os.path.join(os.path.dirname(__file__), "workflows")
    
    if not os.path.isdir(workflows_dir):
        return workflows
    
    # Find all Python files in workflows directory
    workflow_files = glob.glob(os.path.join(workflows_dir, "*.py"))
    
    for filepath in workflow_files:
        filename = os.path.basename(filepath)
        if filename.startswith("_") or filename == "workflow.py":
            continue
        
        module_name = filename[:-3]  # Remove .py
        
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check for required functions/variables
            if hasattr(module, 'run_workflow') and hasattr(module, 'PARAMS'):
                # Get workflow metadata
                if hasattr(module, 'workflow'):
                    workflow_obj = getattr(module, 'workflow')
                    workflow_name = workflow_obj.name
                    workflow_desc = workflow_obj.description
                else:
                    workflow_name = getattr(module, 'WORKFLOW_NAME', module_name)
                    workflow_desc = getattr(module, 'WORKFLOW_DESCRIPTION', '')
                
                workflows[module_name] = {
                    'module': module,
                    'name': workflow_name,
                    'description': workflow_desc,
                    'filepath': filepath,
                    'params': module.PARAMS
                }
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Failed to load workflow {filename}: {e}")
    
    return workflows


def load_crsd_file(filepath):
    """Load CRSD file using sarkit"""
    try:
        import sarkit.crsd as crsd
        
        print(f"\nLoading: {filepath}")
        
        channel_data = {}
        tx_wfm = None
        channel_ids = []
        sample_rate_hz = 100e6  # Default
        
        # Load CRSD file with sarkit
        with open(filepath, 'rb') as f:
            reader = crsd.Reader(f)
            
            # Get channel IDs from metadata
            root = reader.metadata.xmltree.getroot()
            channels = root.findall('.//{http://api.nsgreg.nga.mil/schema/crsd/1.0}Channel')
            channel_ids = [ch.find('{http://api.nsgreg.nga.mil/schema/crsd/1.0}ChId').text 
                          for ch in channels] if channels else []
            
            # Load ALL channels
            if channel_ids:
                for ch_id in channel_ids:
                    channel_data[ch_id] = reader.read_signal(ch_id)
            else:
                # Fallback for files without channel metadata
                if hasattr(reader, 'read_signal_block'):
                    channel_data["CHAN1"] = reader.read_signal_block(0, 0, 256, 256)
                    channel_ids = ["CHAN1"]
            
            # Try to load TX waveform
            try:
                tx_wfm_array = reader.read_support_array("TX_WFM")
                tx_wfm = np.asarray(tx_wfm_array[0, :], dtype=np.complex64)
            except:
                pass
            
            # Extract radar parameters from metadata
            try:
                radar_params = root.find('.//{http://api.nsgreg.nga.mil/schema/crsd/1.0}RadarParameters')
                if radar_params is not None:
                    sample_rate = radar_params.find('.//{http://api.nsgreg.nga.mil/schema/crsd/1.0}SampleRate')
                    if sample_rate is not None:
                        sample_rate_hz = float(sample_rate.text)
            except:
                pass
            
            # Get file header KVPs if available
            file_header_kvps = {}
            try:
                if hasattr(reader, 'file_header'):
                    file_header_kvps = reader.file_header.kvps
            except:
                pass
        
        # Build metadata
        metadata = {
            'sample_rate_hz': sample_rate_hz,
            'file_header_kvps': file_header_kvps
        }
        
        print(f"  Channels: {channel_ids}")
        print(f"  Sample rate: {metadata['sample_rate_hz']/1e6:.1f} MHz")
        
        return channel_data, metadata, tx_wfm
        
    except Exception as e:
        print(f"Error loading CRSD file: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)


def generate_static_html(results, output_dir, workflow_name):
    """Generate a standalone static HTML report from workflow results"""
    from datetime import datetime
    import plotly.io as pio
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build HTML content
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        f"    <title>{workflow_name} - CRSD Inspector</title>",
        "    <meta charset='utf-8'>",
        "    <script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script>",
        "    <style>",
        "        body {",
        "            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;",
        "            max-width: 1400px;",
        "            margin: 0 auto;",
        "            padding: 20px;",
        "            background: #f5f5f5;",
        "        }",
        "        .header {",
        "            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);",
        "            color: white;",
        "            padding: 30px;",
        "            border-radius: 8px;",
        "            margin-bottom: 20px;",
        "        }",
        "        .header h1 { margin: 0; font-size: 32px; }",
        "        .header p { margin: 10px 0 0 0; opacity: 0.9; }",
        "        .section {",
        "            background: white;",
        "            padding: 20px;",
        "            margin-bottom: 20px;",
        "            border-radius: 8px;",
        "            box-shadow: 0 2px 4px rgba(0,0,0,0.1);",
        "        }",
        "        .section h2 {",
        "            margin-top: 0;",
        "            color: #333;",
        "            border-bottom: 2px solid #667eea;",
        "            padding-bottom: 10px;",
        "        }",
        "        table {",
        "            width: 100%;",
        "            border-collapse: collapse;",
        "            margin: 15px 0;",
        "        }",
        "        th, td {",
        "            text-align: left;",
        "            padding: 12px;",
        "            border-bottom: 1px solid #ddd;",
        "        }",
        "        th {",
        "            background: #f8f9fa;",
        "            font-weight: 600;",
        "            color: #495057;",
        "        }",
        "        tr:hover { background: #f8f9fa; }",
        "        .text-content {",
        "            line-height: 1.6;",
        "            color: #333;",
        "        }",
        "        .plot-container {",
        "            margin: 20px 0;",
        "        }",
        "        .footer {",
        "            text-align: center;",
        "            color: #666;",
        "            padding: 20px;",
        "            margin-top: 40px;",
        "            font-size: 14px;",
        "        }",
        "    </style>",
        "</head>",
        "<body>",
        "    <div class='header'>",
        f"        <h1>{workflow_name}</h1>",
        f"        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        "    </div>",
    ]
    
    # Process results
    if 'results' in results:
        items = results['results']
    else:
        # Legacy format - convert to new format
        items = []
        if 'text' in results:
            for text_item in results['text']:
                items.append({'type': 'text', 'content': text_item})
        if 'tables' in results:
            for table_item in results['tables']:
                items.append({'type': 'table', 'title': table_item.get('title', 'Table'), 'data': table_item.get('data', {})})
        if 'plots' in results:
            for plot_item in results['plots']:
                items.append({'type': 'plot', 'figure': plot_item})
    
    # Render items in order
    plot_counter = 0
    for item in items:
        item_type = item.get('type')
        
        if item_type == 'text':
            html_parts.append("    <div class='section'>")
            html_parts.append("        <div class='text-content'>")
            content = item.get('content', [])
            if isinstance(content, str):
                content = [content]
            for line in content:
                html_parts.append(f"            <p>{line}</p>")
            html_parts.append("        </div>")
            html_parts.append("    </div>")
        
        elif item_type == 'table':
            title = item.get('title', 'Table')
            data = item.get('data', {})
            html_parts.append("    <div class='section'>")
            html_parts.append(f"        <h2>{title}</h2>")
            html_parts.append("        <table>")
            
            # Table headers
            if data:
                headers = [k for k in data.keys()]
                html_parts.append("            <tr>")
                for header in headers:
                    html_parts.append(f"                <th>{header}</th>")
                html_parts.append("            </tr>")
                
                # Table rows
                num_rows = len(data[headers[0]]) if headers else 0
                for i in range(num_rows):
                    html_parts.append("            <tr>")
                    for header in headers:
                        value = data[header][i]
                        html_parts.append(f"                <td>{value}</td>")
                    html_parts.append("            </tr>")
            
            html_parts.append("        </table>")
            html_parts.append("    </div>")
        
        elif item_type == 'plot':
            figure = item.get('figure')
            if figure is not None:
                plot_counter += 1
                div_id = f"plot-{plot_counter}"
                
                # Convert figure to JSON
                plot_json = pio.to_json(figure)
                
                html_parts.append("    <div class='section'>")
                html_parts.append("        <div class='plot-container'>")
                html_parts.append(f"            <div id='{div_id}'></div>")
                html_parts.append("            <script>")
                html_parts.append(f"                var plotData = {plot_json};")
                html_parts.append(f"                Plotly.newPlot('{div_id}', plotData.data, plotData.layout, {{responsive: true}});")
                html_parts.append("            </script>")
                html_parts.append("        </div>")
                html_parts.append("    </div>")
    
    # Footer
    html_parts.extend([
        "    <div class='footer'>",
        "        <p>Generated by CRSD Inspector</p>",
        "    </div>",
        "</body>",
        "</html>"
    ])
    
    # Write HTML file
    html_path = output_dir / "index.html"
    with open(html_path, 'w') as f:
        f.write('\n'.join(html_parts))
    
    console.print(f"\n[green]✓[/green] Report generated: {html_path}")
    return html_path


@click.group()
@click.version_option(version="0.1.0", prog_name="crsd-inspector")
def cli():
    """
    CRSD Inspector - Analyze CRSD radar data files
    
    Process CRSD files with various workflows and generate static HTML reports.
    """
    pass


@cli.command()
def list():
    """List available workflows and their parameters"""
    workflows = discover_workflows()
    
    if not workflows:
        console.print("[yellow]No workflows found[/yellow]")
        return
    
    console.print(f"\n[bold cyan]Available Workflows ({len(workflows)}):[/bold cyan]\n")
    
    for module_name, wf_info in workflows.items():
        console.print(f"[bold]{module_name}[/bold]")
        console.print(f"  Name: {wf_info['name']}")
        console.print(f"  Description: {wf_info['description']}")
        
        # Show parameters
        params = wf_info['params']
        if params:
            console.print("  Parameters:")
            for param_name, param_spec in params.items():
                param_type = param_spec.get('type', 'unknown')
                default = param_spec.get('default', 'N/A')
                label = param_spec.get('label', param_name)
                console.print(f"    --{param_name.replace('_', '-')} ({param_type})")
                console.print(f"      {label}")
                console.print(f"      Default: {default}")
        console.print()


@cli.command()
@click.argument('workflow', type=str)
@click.argument('crsd_file', type=click.Path(exists=True))
@click.option('--channel', '-c', default='0', help='Channel ID to process')
@click.option('--output', '-o', default='output', help='Output directory for HTML report')
@click.option('--params', '-p', help='JSON string or file path with workflow parameters')
def run(workflow, crsd_file, channel, output, params):
    """
    Run a workflow on a CRSD file and generate static HTML output
    
    \b
    Example:
        crsd-inspector run pulse_extraction examples/example_4.crsd --channel 0 --output results/
        crsd-inspector run pulse_extraction examples/example_4.crsd --params '{"min_prf_hz": 1000, "max_prf_hz": 2000}'
        crsd-inspector run pulse_extraction examples/example_4.crsd --params params.json
    """
    
    # Discover workflows
    workflows = discover_workflows()
    
    if workflow not in workflows:
        console.print(f"[red]Error:[/red] Workflow '{workflow}' not found")
        console.print(f"\nAvailable workflows: {', '.join(workflows.keys())}")
        console.print("\nUse 'crsd-inspector list' to see all workflows and their parameters")
        sys.exit(1)
    
    wf_info = workflows[workflow]
    wf_module = wf_info['module']
    wf_name = wf_info['name']
    
    # Parse parameters
    workflow_params = {}
    if params:
        try:
            # Check if it's a file path
            if os.path.isfile(params):
                with open(params, 'r') as f:
                    workflow_params = json.load(f)
            else:
                # Parse as JSON string
                workflow_params = json.loads(params)
        except Exception as e:
            console.print(f"[red]Error parsing parameters:[/red] {e}")
            sys.exit(1)
    
    # Load CRSD file
    channel_data, metadata, tx_wfm = load_crsd_file(crsd_file)
    
    # Validate channel
    if channel not in channel_data:
        console.print(f"[red]Error:[/red] Channel '{channel}' not found")
        console.print(f"Available channels: {[k for k in channel_data.keys()]}")
        sys.exit(1)
    
    signal_data = channel_data[channel]
    
    # Add TX waveform to metadata
    if tx_wfm is not None:
        metadata['tx_wfm'] = tx_wfm
    
    # Add channel data and selected channel to metadata
    metadata['channel_data'] = channel_data
    metadata['selected_channel'] = channel
    
    # Merge workflow parameters into metadata
    metadata.update(workflow_params)
    
    # Execute workflow
    print(f"\nRunning workflow: {wf_name}")
    print(f"Channel: {channel}")
    
    try:
        results = wf_module.run_workflow(signal_data=signal_data, metadata=metadata)
        
        # Generate static HTML
        html_path = generate_static_html(results, output, wf_name)
        
        # Print summary
        console.print(f"\n[bold green]✓ Workflow completed successfully[/bold green]")
        console.print(f"\nOpen in browser: file://{html_path.absolute()}")
        
    except Exception as e:
        console.print(f"\n[red]Error executing workflow:[/red] {e}")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


def main():
    """Entry point for crsd-inspector CLI"""
    cli()


if __name__ == '__main__':
    main()
