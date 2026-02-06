"""
CRSD Inspector - Dash Application
Three-panel layout: File/Workflow Selection | Options | Results
"""

import os
import glob
from pathlib import Path
import numpy as np

import dash
from dash import dcc, html, Input, Output, State, ALL, ctx, dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

import sarkit.crsd as crsd

# Import workflows
from crsd_inspector.workflows import signal_analysis, range_doppler, pulse_extraction


# Initialize Dash app with dark Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)

# Custom CSS for sleek appearance
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #060606;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            }
            .card {
                background-color: #1a1a1a;
                border: 1px solid #2a2a2a;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            }
            .card-header {
                background-color: #222222;
                border-bottom: 1px solid #2a2a2a;
                border-radius: 12px 12px 0 0 !important;
                padding: 1rem 1.25rem;
            }
            .card-header h4 {
                color: #00d9ff;
                font-weight: 600;
                margin: 0;
                font-size: 1.1rem;
                letter-spacing: 0.5px;
            }
            .card-body {
                background-color: #1a1a1a;
                border-radius: 0 0 12px 12px;
            }
            h1 {
                color: #00d9ff;
                font-weight: 700;
                letter-spacing: 1px;
                text-shadow: 0 0 10px rgba(0, 217, 255, 0.3);
            }
            .Select-control, .dropdown {
                background-color: #2a2a2a !important;
                border-color: #3a3a3a !important;
                color: #e0e0e0 !important;
            }
            .Select-menu-outer {
                background-color: #2a2a2a !important;
                border-color: #3a3a3a !important;
            }
            .Select-option {
                background-color: #2a2a2a !important;
                color: #e0e0e0 !important;
            }
            .Select-option:hover {
                background-color: #3a3a3a !important;
            }
            label {
                color: #b0b0b0;
                font-weight: 500;
                font-size: 0.9rem;
            }
            .btn-primary {
                background: linear-gradient(135deg, #00d9ff 0%, #0099cc 100%);
                border: none;
                border-radius: 8px;
                font-weight: 600;
                box-shadow: 0 2px 8px rgba(0, 217, 255, 0.3);
                transition: all 0.3s ease;
            }
            .btn-primary:hover {
                background: linear-gradient(135deg, #00ffff 0%, #00b8e6 100%);
                box-shadow: 0 4px 12px rgba(0, 217, 255, 0.5);
                transform: translateY(-1px);
            }
            .alert-success {
                background-color: #1a3a2a;
                border-color: #2a5a3a;
                color: #66ff99;
            }
            .alert-danger {
                background-color: #3a1a1a;
                border-color: #5a2a2a;
                color: #ff6666;
            }
            .text-muted {
                color: #808080 !important;
            }
            hr {
                border-color: #2a2a2a;
                opacity: 0.5;
            }
            .table {
                color: #e0e0e0;
            }
            .table-striped tbody tr:nth-of-type(odd) {
                background-color: #1a1a1a;
            }
            .table-striped tbody tr:nth-of-type(even) {
                background-color: #222222;
            }
            .table-hover tbody tr:hover {
                background-color: #2a2a2a;
            }
            .table thead th {
                border-color: #3a3a3a;
                color: #00d9ff;
                font-weight: 600;
            }
            .table td, .table th {
                border-color: #2a2a2a;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Discover workflows
WORKFLOWS = {
    "Signal Analysis": signal_analysis,
    "Range-Doppler Processing": range_doppler,
    "Pulse Extraction": pulse_extraction,
}

# App layout - 3 column design
app.layout = dbc.Container([
    dbc.Row([
        html.H3("CRSD Inspector", className="text-center my-1", style={'marginTop': '0.5rem', 'marginBottom': '0.5rem'}),
    ]),
    
    # Top Row: File Selection and Workflow Options
    dbc.Row([
        # Panel 1: File and Workflow Selection
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("File & Workflow Selection")),
                dbc.CardBody([
                    # Directory browser
                    html.Label("Directory", className="fw-bold"),
                    dbc.InputGroup([
                        dbc.Input(
                            id='directory-input',
                            type='text',
                            value='examples',
                            placeholder="Path to CRSD files..."
                        ),
                        dbc.Button("Scan", id="scan-button", color="secondary", n_clicks=0),
                    ], className="mb-3"),
                    
                    html.Hr(),
                    
                    # File selection
                    html.Label("CRSD File", className="fw-bold"),
                    dcc.Dropdown(
                        id='file-dropdown',
                        placeholder="Select a CRSD file...",
                        className="mb-3"
                    ),
                    
                    html.Hr(),
                    
                    # Workflow selection
                    html.Label("Workflow", className="fw-bold"),
                    dcc.Dropdown(
                        id='workflow-dropdown',
                        options=[{"label": name, "value": name} for name in WORKFLOWS.keys()],
                        placeholder="Select a workflow...",
                        className="mb-3"
                    ),
                    
                    # Workflow description
                    html.Div(id='workflow-description', className="text-muted small"),
                ])
            ], className="h-100"),
        ], width=6),
        
        # Panel 2: Workflow Options
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Workflow Options")),
                dbc.CardBody([
                    html.Div(id='workflow-options'),
                    html.Hr(),
                    dbc.Button(
                        "Execute Workflow",
                        id="execute-button",
                        color="primary",
                        className="w-100",
                        disabled=True
                    ),
                    html.Div(id='status-message', className="mt-3"),
                ])
            ], className="h-100"),
        ], width=6),
    ], className="g-3 mb-3"),
    
    # Bottom Row: Results Full Width
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Results")),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading",
                        type="default",
                        children=html.Div(id='results-panel')
                    )
                ])
            ]),
        ], width=12),
    ], className="g-3")
], fluid=True, className="p-3")


# Callback to populate file dropdown from directory
@app.callback(
    Output('file-dropdown', 'options'),
    Input('scan-button', 'n_clicks'),
    State('directory-input', 'value'),
    prevent_initial_call=False
)
def populate_files(n_clicks, directory):
    """Scan for CRSD files in specified directory"""
    if not directory:
        directory = 'examples'
    
    # Handle relative paths
    if not os.path.isabs(directory):
        directory = Path(__file__).parent / directory
    else:
        directory = Path(directory)
    
    if not directory.exists():
        return []
    
    crsd_files = glob.glob(str(directory / "*.crsd"))
    
    if not crsd_files:
        return []
    
    return [{"label": os.path.basename(f), "value": f} for f in sorted(crsd_files)]


# Callback to update workflow description
@app.callback(
    Output('workflow-description', 'children'),
    Input('workflow-dropdown', 'value')
)
def update_workflow_description(workflow_name):
    """Display workflow description"""
    if not workflow_name:
        return ""
    
    workflow_module = WORKFLOWS.get(workflow_name)
    if workflow_module and hasattr(workflow_module, 'workflow'):
        return workflow_module.workflow.description
    return ""


# Callback to enable execute button
@app.callback(
    Output('execute-button', 'disabled'),
    Input('file-dropdown', 'value'),
    Input('workflow-dropdown', 'value')
)
def toggle_execute_button(file_path, workflow_name):
    """Enable button only when both file and workflow are selected"""
    return not (file_path and workflow_name)


# Callback to generate workflow options panel
@app.callback(
    Output('workflow-options', 'children'),
    Input('file-dropdown', 'value'),
    Input('workflow-dropdown', 'value')
)
def update_options_panel(file_path, workflow_name):
    """Generate options based on loaded file and selected workflow"""
    if not file_path:
        return html.Div("Select a CRSD file to see options", className="text-muted")
    
    try:
        # Load file metadata
        with open(file_path, 'rb') as f:
            reader = crsd.Reader(f)
            root = reader.metadata.xmltree.getroot()
            
            # Get channel IDs
            channel_ids = []
            channels = root.findall('.//{http://api.nsgreg.nga.mil/schema/crsd/1.0}Channel')
            for channel in channels:
                ch_id_elem = channel.find('{http://api.nsgreg.nga.mil/schema/crsd/1.0}ChId')
                if ch_id_elem is not None:
                    channel_ids.append(ch_id_elem.text)
        
        options = [
            html.Label("Channel", className="fw-bold mb-2"),
            dcc.Dropdown(
                id='channel-dropdown',
                options=[{"label": ch_id, "value": ch_id} for ch_id in channel_ids],
                value=channel_ids[0] if channel_ids else None,
                className="mb-3"
            ),
        ]
        
        # Add workflow-specific options from PARAMS
        if workflow_name:
            workflow_module = WORKFLOWS.get(workflow_name)
            if workflow_module and hasattr(workflow_module, 'PARAMS'):
                options.append(html.Hr())
                options.append(html.H6("Workflow Parameters", className="mb-3"))
                
                for param_name, param_config in workflow_module.PARAMS.items():
                    options.append(html.Label(param_config['label'], className="fw-bold mb-2"))
                    
                    param_type = param_config.get('type', 'number')
                    
                    if param_type == 'dropdown':
                        options.append(dcc.Dropdown(
                            id={'type': 'workflow-param', 'name': param_name},
                            options=param_config.get('options', []),
                            value=param_config.get('default'),
                            className="mb-3"
                        ))
                    elif param_type == 'number':
                        options.append(dcc.Input(
                            id={'type': 'workflow-param', 'name': param_name},
                            type='number',
                            value=param_config.get('default'),
                            min=param_config.get('min'),
                            max=param_config.get('max'),
                            step=param_config.get('step'),
                            className="form-control mb-3"
                        ))
                    elif param_type == 'text':
                        options.append(dcc.Input(
                            id={'type': 'workflow-param', 'name': param_name},
                            type='text',
                            value=param_config.get('default', ''),
                            className="form-control mb-3"
                        ))
        
        # File info
        file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
        options.append(html.Hr())
        options.append(html.Div([
            html.P([html.Strong("File Info:"), html.Br(),
                   f"Channels: {len(channel_ids)}", html.Br(),
                   f"Size: {file_size:.1f} MB"])
        ], className="text-muted small"))
        
        return options
        
    except Exception as e:
        return html.Div(f"Error loading file: {str(e)}", className="text-danger")


# Main callback to execute workflow
@app.callback(
    Output('results-panel', 'children'),
    Output('status-message', 'children'),
    Input('execute-button', 'n_clicks'),
    State('file-dropdown', 'value'),
    State('workflow-dropdown', 'value'),
    State('channel-dropdown', 'value'),
    State({'type': 'workflow-param', 'name': ALL}, 'value'),
    prevent_initial_call=True
)
def execute_workflow(n_clicks, file_path, workflow_name, channel_id, param_values):
    """Execute selected workflow on selected channel"""
    if not all([file_path, workflow_name, channel_id]):
        return html.Div("Missing selections", className="text-warning"), ""
    
    # Extract workflow parameters from pattern-matched inputs
    workflow_params = {}
    if param_values:
        param_ids = [p['id']['name'] for p in ctx.states_list[-1]]
        workflow_params = dict(zip(param_ids, param_values))
    
    try:
        # Load CRSD file
        with open(file_path, 'rb') as f:
            reader = crsd.Reader(f)
            root = reader.metadata.xmltree.getroot()
            
            # Get all available channels from XML metadata
            channel_ids = []
            channels = root.findall('.//{http://api.nsgreg.nga.mil/schema/crsd/1.0}Channel')
            for channel in channels:
                ch_id_elem = channel.find('{http://api.nsgreg.nga.mil/schema/crsd/1.0}ChId')
                if ch_id_elem is not None:
                    channel_ids.append(ch_id_elem.text)
            
            # Load all channel data for variants (enables caching/memoization)
            all_channel_data = {}
            for ch_id in channel_ids:
                all_channel_data[ch_id] = reader.read_signal(ch_id)
            
            # Use selected channel
            signal_data = all_channel_data[channel_id]
            
            # Get TX waveform (if available)
            tx_wfm = None
            try:
                tx_wfm_array = reader.read_support_array("TX_WFM")
                tx_wfm = np.asarray(tx_wfm_array[0, :], dtype=np.complex64)
            except:
                pass
            
            # Extract radar parameters
            sample_rate_hz = 100e6  # Default
            prf_hz = 1000.0  # Default
            try:
                radar_params = root.find('.//{http://api.nsgreg.nga.mil/schema/crsd/1.0}RadarParameters')
                if radar_params is not None:
                    sample_rate = radar_params.find('.//{http://api.nsgreg.nga.mil/schema/crsd/1.0}SampleRate')
                    if sample_rate is not None:
                        sample_rate_hz = float(sample_rate.text)
                    
                    prf = radar_params.find('.//{http://api.nsgreg.nga.mil/schema/crsd/1.0}PRF')
                    if prf is not None:
                        prf_hz = float(prf.text)
            except:
                pass
            
            # Build metadata dict with all channel data for variants
            file_metadata = {
                'tx_wfm': tx_wfm,
                'selected_channel': channel_id,
                'all_channels': channel_ids,
                'signal_data_all_channels': all_channel_data,
                'sample_rate_hz': sample_rate_hz,
                'prf_hz': prf_hz,
            }
            
            # Add workflow-specific parameters from store
            if workflow_params:
                if 'range-window-type' in workflow_params:
                    file_metadata['range_window'] = workflow_params['range-window-type']
                if 'doppler-window-type' in workflow_params:
                    file_metadata['doppler_window'] = workflow_params['doppler-window-type']
        
        # Execute workflow
        workflow_module = WORKFLOWS[workflow_name]
        results = workflow_module.run_workflow(signal_data, metadata=file_metadata)
        
        # Render results
        if "results" in results:
            rendered = render_results(results["results"])
            status = dbc.Alert(
                f"✓ {workflow_name} completed successfully",
                color="success",
                dismissable=True
            )
            return rendered, status
        else:
            return html.Div("No results returned", className="text-warning"), ""
            
    except Exception as e:
        error_msg = dbc.Alert(
            [html.H5("Execution Error"), html.P(str(e))],
            color="danger"
        )
        return html.Div("See error message"), error_msg


def render_results(results_list):
    """Render workflow results (text, tables, plots)"""
    rendered = []
    
    for i, item in enumerate(results_list):
        item_type = item.get("type")
        
        if item_type == "text":
            for text in item.get("content", []):
                rendered.append(html.P(text, className="mb-2"))
            rendered.append(html.Hr())
            
        elif item_type == "table":
            title = item.get("title", "Table")
            data = item.get("data", {})
            
            # Convert dict to table rows
            if data:
                columns = list(data.keys())
                num_rows = len(data[columns[0]])
                
                table_header = html.Thead(html.Tr([html.Th(col) for col in columns]))
                table_rows = []
                for row_idx in range(num_rows):
                    table_rows.append(
                        html.Tr([html.Td(data[col][row_idx]) for col in columns])
                    )
                table_body = html.Tbody(table_rows)
                
                rendered.append(html.H5(title, className="mt-4"))
                rendered.append(
                    dbc.Table(
                        [table_header, table_body],
                        bordered=True,
                        hover=True,
                        responsive=True,
                        striped=True,
                        className="mb-4"
                    )
                )
            
        elif item_type == "plot":
            figure = item.get("figure")
            if figure:
                rendered.append(
                    dcc.Graph(
                        figure=figure,
                        id={"type": "result-graph", "index": i},
                        config={'displayModeBar': True},
                        className="mb-4"
                    )
                )
    
    if not rendered:
        return html.Div("No results to display", className="text-muted")
    
    return html.Div(rendered)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("CRSD Inspector - Dash Application")
    print("="*60)
    print("\nStarting server at http://127.0.0.1:8050")
    print("Press Ctrl+C to stop\n")
    app.run(debug=True, port=8050)
