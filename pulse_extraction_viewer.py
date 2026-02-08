"""
Pulse Extraction Viewer for example_4.crsd
Generates static HTML dashboard
"""

import numpy as np
import sarkit.crsd as crsd
from crsd_inspector.workflows import pulse_extraction
import staticdash

def load_and_run():
    """Load example_4.crsd and run pulse extraction workflow"""
    
    # Load CRSD file
    print("Loading example_4.crsd...")
    with open('examples/example_4.crsd', 'rb') as f:
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
    
    print(f"Signal shape: {signal_data.shape}")
    print(f"Sample rate: {sample_rate_hz/1e6:.1f} MHz")
    print(f"TX waveform length: {len(tx_wfm)} samples")
    
    # Extract file header KVPs for ground truth
    file_header_kvps = {}
    if hasattr(reader.metadata, 'file_header_part') and hasattr(reader.metadata.file_header_part, 'additional_kvps'):
        file_header_kvps = reader.metadata.file_header_part.additional_kvps
    
    # Prepare metadata
    metadata = {
        'sample_rate_hz': sample_rate_hz,
        'tx_wfm': tx_wfm,
        'window_type': 'hamming',
        'file_header_kvps': file_header_kvps,
    }
    
    # Run workflow
    print("\nRunning pulse extraction workflow...")
    workflow = pulse_extraction.run_workflow(signal_data, metadata)
    
    return workflow


if __name__ == '__main__':
    # Run the workflow
    workflow = load_and_run()
    
    # Create staticdash Dashboard
    print("\nGenerating static HTML dashboard...")
    dashboard = staticdash.Dashboard("Pulse Extraction - example_4.crsd")
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

