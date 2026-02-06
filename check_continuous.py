from sarkit import crsd
import numpy as np
from scipy.signal import correlate

# Read the continuous CRSD
with open('/Users/brian.day/git/crsd-inspector/examples/example_continuous.crsd', 'rb') as f:
    reader = crsd.Reader(f)
    
    # Get channel IDs
    root = reader.metadata.xmltree.getroot()
    channels = root.findall('.//{http://api.nsgreg.nga.mil/schema/crsd/1.0}Channel')
    channel_ids = [ch.find('{http://api.nsgreg.nga.mil/schema/crsd/1.0}ChId').text 
                  for ch in channels] if channels else []
    
    print(f'Channel IDs: {channel_ids}')
    
    if channel_ids:
        data = reader.read_signal(channel_ids[0])
    else:
        print('No channels found!')
        exit(1)
    
    # Get TX waveform
    try:
        tx_wfm_array = reader.read_support_array("TX_WFM")
        tx_wfm = np.asarray(tx_wfm_array[0, :], dtype=np.complex64)
        print(f'\nTX waveform: {len(tx_wfm)} samples')
    except Exception as e:
        print(f'Failed to load TX waveform: {e}')
        tx_wfm = None
print(f'Data shape: {data.shape}')
print(f'Total samples: {data.size:,}')

signal_data = data.ravel()

# Compute matched filter output if we have reference
sample_rate = 100e6
if tx_wfm is not None:
    print('\n=== MATCHED FILTER OUTPUT ===')
    mf_output = correlate(signal_data, np.conj(tx_wfm), mode='same')
    mf_power = np.abs(mf_output) ** 2
    
    print(f'MF Power stats:')
    print(f'  Max: {np.max(mf_power):.2e}')
    print(f'  Mean: {np.mean(mf_power):.2e}')
    
    # Find strong peaks in matched filter output
    strong_threshold = np.max(mf_power) * 0.1
    strong_indices = np.where(mf_power > strong_threshold)[0]
    if len(strong_indices) > 0:
        print(f'\nStrong MF peaks:')
        print(f'  Number of samples above threshold: {len(strong_indices):,}')
        # Look at actual peak locations  
        local_maxima = []
        min_distance = 50000  # About half a PRI at 1000 Hz
        last_peak = -min_distance
        for i in range(1, len(mf_power)-1):
            if mf_power[i] > mf_power[i-1] and mf_power[i] > mf_power[i+1]:
                if mf_power[i] > strong_threshold and i - last_peak > min_distance:
                    local_maxima.append(i)
                    last_peak = i
        
        print(f'  Number of local maxima (pulses): {len(local_maxima)}')
        if len(local_maxima) > 1:
            spacings = np.diff(local_maxima)
            print(f'  Pulse spacings (samples): min={np.min(spacings)}, max={np.max(spacings)}, mean={np.mean(spacings):.0f}')
            print(f'  Estimated PRF: {sample_rate/np.mean(spacings):.1f} Hz')
            print(f'  First 10 pulse locations: {local_maxima[:10]}')
    
    # Energy detector on MF output
    sample_rate = 100e6
    max_prf = 10000
    window_samples = int(sample_rate / max_prf / 4)
    window = np.ones(window_samples) / window_samples
    smoothed_mf_power = np.convolve(mf_power, window, mode='same')
    smoothed_mf_power_db = 10 * np.log10(smoothed_mf_power + 1e-12)
    
    print(f'\nEnergy detector on MF output:')
    print(f'  Window size: {window_samples:,} samples')
    print(f'  Peak smoothed power: {np.max(smoothed_mf_power_db):.1f} dB')

# Original raw power for comparison
print('\n=== RAW POWER (NO MATCHED FILTER) ===')
power = np.abs(signal_data) ** 2
print(f'\nPower stats:')
print(f'  Max: {np.max(power):.2e}')
print(f'  Mean: {np.mean(power):.2e}')
print(f'  Nonzero samples: {np.count_nonzero(power > 1e-20):,}')

# Find where signal is strong
strong_threshold = np.max(power) * 0.1
strong_indices = np.where(power > strong_threshold)[0]
if len(strong_indices) > 0:
    print(f'\nStrong signal regions:')
    print(f'  First 10 indices: {strong_indices[:10]}')
    diffs = np.diff(strong_indices[:200])
    print(f'  Min spacing: {np.min(diffs)}')
    print(f'  Max spacing: {np.max(diffs)}')
    # Find gaps (large jumps)
    gaps = np.where(diffs > 1000)[0]
    if len(gaps) > 0:
        print(f'  Number of gaps: {len(gaps)}')
        print(f'  First few gap sizes: {diffs[gaps[:5]]}')

# Check what happens with energy detection
sample_rate = 100e6
max_prf = 10000
window_samples = int(sample_rate / max_prf / 4)
print(f'\nEnergy detector parameters:')
print(f'  Window size: {window_samples:,} samples')
print(f'  Window duration: {window_samples/sample_rate*1e3:.3f} ms')

# Compute smoothed power
window = np.ones(window_samples) / window_samples
convolved_power = np.convolve(power, window, mode='same')
smoothed_power_db = 10 * np.log10(convolved_power + 1e-12)
threshold_db = np.max(smoothed_power_db) - 20
threshold_linear = 10 ** (threshold_db / 10)

print(f'  Peak smoothed power: {np.max(smoothed_power_db):.1f} dB')
print(f'  Threshold: {threshold_db:.1f} dB')

above_threshold = smoothed_power_db > threshold_db
num_above = np.sum(above_threshold)
print(f'  Samples above threshold: {num_above:,}')

