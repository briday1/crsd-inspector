from sarkit import crsd
import numpy as np
from scipy.signal import correlate

# Read the continuous CRSD
with open('/Users/brian.day/git/crsd-inspector/examples/example_continuous.crsd', 'rb') as f:
    reader = crsd.Reader(f)
    
    root = reader.metadata.xmltree.getroot()
    channels = root.findall('.//{http://api.nsgreg.nga.mil/schema/crsd/1.0}Channel')
    channel_ids = [ch.find('{http://api.nsgreg.nga.mil/schema/crsd/1.0}ChId').text 
                  for ch in channels] if channels else []
    
    data = reader.read_signal(channel_ids[0])
    tx_wfm_array = reader.read_support_array("TX_WFM")
    tx_wfm = np.asarray(tx_wfm_array[0, :], dtype=np.complex64)
    
    signal = data.ravel()
    
    print('Testing rising edge detection on matched filter output...')
    print(f'Signal length: {len(signal):,} samples')
    print(f'TX waveform length: {len(tx_wfm)} samples')
    
    # Apply window to reference
    window = np.hamming(len(tx_wfm))
    ref_wfm = tx_wfm * window
    
    # Matched filter
    print('\nApplying matched filter...')
    mf_output = correlate(signal, np.conj(ref_wfm), mode='same')
    mf_power = np.abs(mf_output) ** 2
    mf_power_db = 10 * np.log10(mf_power + 1e-12)
    
    print(f'MF power:')
    print(f'  Peak: {np.max(mf_power_db):.1f} dB')
    print(f'  Mean: {np.mean(mf_power_db):.1f} dB')
    
    # Find rising edges
    sample_rate = 100e6
    max_prf = 10000
    min_distance_samples = int(sample_rate / max_prf * 0.8)
    
    peak_power_db = np.max(mf_power_db)
    threshold_db = peak_power_db - 20
    threshold_linear = 10 ** (threshold_db / 10)
    
    print(f'\nRising edge detection:')
    print(f'  Threshold: {threshold_db:.1f} dB ({threshold_linear:.2e})')
    print(f'  Min distance: {min_distance_samples:,} samples')
    
    # Find regions above threshold
    above_threshold = mf_power > threshold_linear
    num_above = np.sum(above_threshold)
    print(f'  Samples above threshold: {num_above:,} ({100*num_above/len(mf_power):.2f}%)')
    
    # Find rising edges
    transitions = np.diff(above_threshold.astype(int))
    rising_edges = np.where(transitions == 1)[0] + 1
    falling_edges = np.where(transitions == -1)[0] + 1
    
    print(f'  Raw rising edges: {len(rising_edges)}')
    print(f'  Raw falling edges: {len(falling_edges)}')
    
    if len(rising_edges) > 0:
        print(f'  First 10 rising edges: {rising_edges[:10]}')
        
        # Filter edges too close together
        valid_edges = [rising_edges[0]]
        for edge in rising_edges[1:]:
            if edge - valid_edges[-1] >= min_distance_samples:
                valid_edges.append(edge)
        
        print(f'\nFiltered rising edges: {len(valid_edges)}')
        print(f'  First 10 locations: {valid_edges[:10]}')
        
        if len(valid_edges) > 1:
            spacings = np.diff(valid_edges)
            print(f'  Spacings: min={np.min(spacings):,}, max={np.max(spacings):,}, mean={np.mean(spacings):.0f}')
            print(f'  Estimated PRF: {sample_rate/np.mean(spacings):.1f} Hz')
            print(f'  Expected PRF: 1000 Hz')
            print(f'  Expected spacing: 100,000 samples')
