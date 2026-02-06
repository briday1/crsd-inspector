from sarkit import crsd
import numpy as np
from scipy.signal import correlate

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
    print(f'TX waveform length: {len(tx_wfm)} samples')
    print(f'TX waveform energy: {np.sum(np.abs(tx_wfm)**2):.2e}')
    print(f'TX waveform normalized: {np.sqrt(np.mean(np.abs(tx_wfm)**2)):.2e}')
    
    signal = data.ravel()
    
    # Method 1: Current approach (no normalization)
    print('\n=== Method 1: No normalization ===')
    mf1 = correlate(signal, np.conj(tx_wfm), mode='same')
    mf1_power_db = 10 * np.log10(np.abs(mf1)**2 + 1e-12)
    print(f'Peak MF output: {np.max(mf1_power_db):.1f} dB')
    
    # Method 2: Normalize reference to unit energy
    print('\n=== Method 2: Normalize reference to unit energy ===')
    ref_normalized = tx_wfm / np.sqrt(np.sum(np.abs(tx_wfm)**2))
    mf2 = correlate(signal, np.conj(ref_normalized), mode='same')
    mf2_power_db = 10 * np.log10(np.abs(mf2)**2 + 1e-12)
    print(f'Peak MF output: {np.max(mf2_power_db):.1f} dB')
    
    # Method 3: Proper matched filter with normalization
    print('\n=== Method 3: Matched filter normalized by reference energy ===')
    ref_energy = np.sum(np.abs(tx_wfm)**2)
    mf3 = correlate(signal, np.conj(tx_wfm), mode='same') / ref_energy
    mf3_power_db = 10 * np.log10(np.abs(mf3)**2 + 1e-12)
    print(f'Peak MF output: {np.max(mf3_power_db):.1f} dB')
    
    # Method 4: Just look at raw signal power where TX waveform is
    print('\n=== Raw signal check ===')
    raw_power = np.abs(signal)**2
    raw_power_db = 10 * np.log10(raw_power + 1e-12)
    print(f'Peak raw power: {np.max(raw_power_db):.1f} dB')
    
    # Look at first pulse (should be around sample 0-512)
    first_pulse = signal[:1000]
    print(f'\nFirst 1000 samples:')
    print(f'  Max amplitude: {np.max(np.abs(first_pulse)):.2e}')
    print(f'  Max power (dB): {10*np.log10(np.max(np.abs(first_pulse)**2)):.1f}')
    
    # Find where signal is non-zero
    nonzero_mask = np.abs(signal) > 1e-10
    nonzero_indices = np.where(nonzero_mask)[0]
    if len(nonzero_indices) > 100:
        print(f'\nNon-zero signal spans: {nonzero_indices[0]} to {nonzero_indices[-1]}')
        print(f'First non-zero region: {nonzero_indices[:20]}')
