from sarkit import crsd
import numpy as np
from scipy.signal import correlate, find_peaks

# Read the continuous CRSD
with open('/Users/brian.day/git/crsd-inspector/examples/example_continuous.crsd', 'rb') as f:
    reader = crsd.Reader(f)
    
    # Get channel and data
    root = reader.metadata.xmltree.getroot()
    channels = root.findall('.//{http://api.nsgreg.nga.mil/schema/crsd/1.0}Channel')
    channel_ids = [ch.find('{http://api.nsgreg.nga.mil/schema/crsd/1.0}ChId').text 
                  for ch in channels] if channels else []
    
    data = reader.read_signal(channel_ids[0])
    
    # Get TX waveform
    tx_wfm_array = reader.read_support_array("TX_WFM")
    tx_wfm = np.asarray(tx_wfm_array[0, :], dtype=np.complex64)
    
    print(f'Data shape: {data.shape}')
    print(f'TX waveform shape: {tx_wfm.shape}')
    print(f'TX waveform length: {len(tx_wfm)} samples')
    
    # Apply Hamming window to reference
    ref_wfm = tx_wfm.copy()
    window = np.hamming(len(ref_wfm))
    ref_wfm = ref_wfm * window
    
    signal = data.ravel()
    
    # Matched filter
    print('\nPerforming matched filtering...')
    mf_output = correlate(signal, np.conj(ref_wfm), mode='same')
    mf_power = np.abs(mf_output) ** 2
    
    print(f'MF output shape: {mf_output.shape}')
    print(f'MF power max: {np.max(mf_power):.2e}')
    print(f'MF power mean: {np.mean(mf_power):.2e}')
    print(f'MF power min: {np.min(mf_power):.2e}')
    
    # Find peaks
    sample_rate = 100e6
    max_prf = 10000
    min_distance_samples = int(sample_rate / max_prf * 0.8)
    
    # Convert to dB
    mf_power_db = 10 * np.log10(mf_power + 1e-12)
    peak_power_db = np.max(mf_power_db)
    threshold_db = peak_power_db - 20
    threshold_linear = 10 ** (threshold_db / 10)
    
    print(f'\nPeak finding:')
    print(f'  Peak power: {peak_power_db:.1f} dB')
    print(f'  Threshold: {threshold_db:.1f} dB')
    print(f'  Min distance: {min_distance_samples} samples')
    
    # Find peaks
    peak_indices, properties = find_peaks(mf_power, height=threshold_linear, distance=min_distance_samples)
    
    print(f'  Number of peaks found: {len(peak_indices)}')
    if len(peak_indices) > 0:
        print(f'  First 10 peak locations: {peak_indices[:10]}')
        print(f'  Peak values (dB): {10*np.log10(mf_power[peak_indices[:10]] + 1e-12)}')
        if len(peak_indices) > 1:
            spacings = np.diff(peak_indices)
            print(f'  Peak spacings: min={np.min(spacings)}, max={np.max(spacings)}, mean={np.mean(spacings):.0f}')
            print(f'  Estimated PRF: {sample_rate/np.mean(spacings):.1f} Hz')
