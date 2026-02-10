"""
Pulse Extraction Viewer for CRSD files
Generates static HTML dashboard
"""

import sys
import numpy as np
import sarkit.crsd as crsd
from crsd_inspector.workflows import pulse_extraction
import staticdash

def load_and_run(filename='examples/example_5.crsd'):
    """Load CRSD file and run pulse extraction workflow"""
    
    # Load CRSD file
    print(f"Loading {filename}...")
    with open(filename, 'rb') as f:
        reader = crsd.Reader(f)
        root = reader.metadata.xmltree.getroot()
        
        # Get channel IDs
        channel_ids = []
        channels = root.findall('.//{http://api.nsgreg.nga.mil/schema/crsd/1.0}Channel')
        for channel in channels:
            ch_id_elem = channel.find('{http://api.nsgreg.nga.mil/schema/crsd/1.0}ChId')
            if ch_id_elem is not None:
                channel_ids.append(ch_id_elem.text)
        
        # Use first channel
        channel_id = channel_ids[0] if channel_ids else None
        signal_data = reader.read_signal(channel_id)
        
        # Get TX waveform
        tx_wfm = None
        try:
            tx_wfm_array = reader.read_support_array("TX_WFM")
            tx_wfm = np.asarray(tx_wfm_array[0, :], dtype=np.complex64)
        except:
            pass
        
        # Extract sample rate
        sample_rate_hz = 100e6  # Default
        try:
            radar_params = root.find('.//{http://api.nsgreg.nga.mil/schema/crsd/1.0}RadarParameters')
            if radar_params is not None:
                sample_rate = radar_params.find('.//{http://api.nsgreg.nga.mil/schema/crsd/1.0}SampleRate')
                if sample_rate is not None:
                    sample_rate_hz = float(sample_rate.text)
        except:
            pass
        
        # Extract file header KVPs for ground truth
        file_header_kvps = {}
        if hasattr(reader.metadata, 'file_header_part') and hasattr(reader.metadata.file_header_part, 'additional_kvps'):
            file_header_kvps = reader.metadata.file_header_part.additional_kvps
        
        # Try to extract PRF bounds from PPP data
        min_prf_hz = 800.0  # Default
        max_prf_hz = 1200.0  # Default
        try:
            ppp = reader.read_ppps('TX1')
            if ppp is not None and len(ppp) > 1 and 'TxTime' in ppp.dtype.names:
                pulse_times = ppp['TxTime']
                pris = np.diff(pulse_times)
                prfs = 1.0 / pris
                # Use actual PRF range with some margin
                min_prf_hz = float(prfs.min() * 0.9)  # 10% margin
                max_prf_hz = float(prfs.max() * 1.1)  # 10% margin
                print(f"Detected PRF range: {min_prf_hz:.1f} - {max_prf_hz:.1f} Hz")
        except Exception as e:
            print(f"Could not extract PRF from PPP: {e}")
            print(f"Using default PRF range: {min_prf_hz:.1f} - {max_prf_hz:.1f} Hz")
    
    print(f"Signal shape: {signal_data.shape}")
    print(f"Sample rate: {sample_rate_hz/1e6:.1f} MHz")
    print(f"TX waveform length: {len(tx_wfm)} samples")
    
    # Prepare metadata
    metadata = {
        'sample_rate_hz': sample_rate_hz,
        'tx_wfm': tx_wfm,
        'window_type': 'hamming',
        'file_header_kvps': file_header_kvps,
        'min_prf_hz': min_prf_hz,
        'max_prf_hz': max_prf_hz,
    }
    
    # Run workflow
    print("\nRunning pulse extraction workflow...")
    workflow = pulse_extraction.run_workflow(signal_data, metadata)
    
    return workflow


if __name__ == '__main__':
    # Get filename from command line or use default
    filename = sys.argv[1] if len(sys.argv) > 1 else 'examples/example_5.crsd'
    
    # Run the workflow
    workflow = load_and_run(filename)
    
    # Extract filename for title
    import os
    basename = os.path.basename(filename)
    
    # Create staticdash Dashboard
    print("\nGenerating static HTML dashboard...")
    dashboard = staticdash.Dashboard(f"Pulse Extraction - {basename}")
    page = staticdash.Page(slug="results", title="Results")
    
    # Add workflow results to page
    results = workflow.get('results', [])
    
    for item in results:
        if item['type'] == 'text':
            content = item['content']
            if isinstance(content, list):
                # Join list items with HTML breaks
                text_html = '<br>'.join(content)
                page.add_text(text_html)
            else:
                page.add_header(content, level=2)
        
        elif item['type'] == 'plot':
            fig = item['figure']
            page.add_plot(fig)
    
    dashboard.add_page(page)
    
    # Publish to output directory
    output_dir = 'pulse_extraction_output'
    dashboard.publish(output_dir)
    
    print(f"\n{'='*60}")
    print(f"Dashboard published to: {output_dir}/index.html")
    print(f"Open it: open {output_dir}/index.html")
    print(f"{'='*60}")

