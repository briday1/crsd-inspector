from sarkit import crsd
import numpy as np
from scipy.signal import correlate, find_peaks

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
    
    print('Testing integrated matched filter for pulse detection...')
    print(f'Signal length: {len(signal):,} samples')
    print(f'TX waveform length: {len(tx_wfm)} samples')
    
    # Apply window to reference
    window = np.hamming(len(tx_wfm))
    ref_wfm = tx_wfm * window
    
    # Matched filter
    print('\nApplying matched filter...')
    mf_output = correlate(signal, np.conj(ref_wfm), mode='same')
    mf_power = np.abs(mf_output) ** 2
    
    # Integrate over range bins
    integration_window = len(ref_wfm) * 4  # 4x waveform length = 2000 samples
    print(f'Integration window: {integration_window} samples')
    
    window_kernel = np.ones(integration_window) / integration_window
    integrated_power = np.convolve(mf_power, window_kernel, mode='same')
    integrated_power_db = 10 * np.log10(integrated_power + 1e-12)
    
    print(f'\nIntegrated power:')
    print(f'  Peak: {np.max(integrated_power_db):.1f} dB')
    print(f'  Mean: {np.mean(integrated_power_db):.1f} dB')
    
    # Find peaks
    sample_rate = 100e6
    max_prf = 10000
    min_distance_samples = int(sample_rate / max_prf * 0.8)
    
    peak_power_db = np.max(integrated_power_db)
    threshold_db = peak_power_db - 20
    threshold_linear = 10 ** (threshold_db / 10)
    
    print(f'\nPeak finding:')
    print(f'  Threshold: {threshold_db:.1f} dB')
    print(f'  Min distance: {min_distance_samples:,} samples')
    
    peak_indices, _ = find_peaks(integrated_power, height=threshold_linear, distance=min_distance_samples)
    
    print(f'  Number of peaks found: {len(peak_indices)}')
    if len(peak_indices) > 0:
        print(f'  First 10 peak locations: {peak_indices[:10]}')
        if len(peak_indices) > 1:
            spacings = np.diff(peak_indices)
            print(f'  Peak spacings: min={np.min(spacings):,}, max={np.max(spacings):,}, mean={np.mean(spacings):.0f}')
            print(f'  Estimated PRF: {sample_rate/np.mean(spacings):.1f} Hz')
            print(f'  Expected PRF: 1000 Hz')
            print(f'  Expected spacing: 100,000 samples')
