from sarkit import crsd
import numpy as np
from scipy.signal import correlate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    
    # Matched filter on first pulse only
    print('Analyzing first pulse (samples 0-99999)...')
    first_pulse = signal[:100000]
    
    # Matched filter
    mf_output = correlate(first_pulse, np.conj(tx_wfm), mode='same')
    mf_power = np.abs(mf_output)**2
    mf_power_db = 10 * np.log10(mf_power + 1e-12)
    
    print(f'MF output on first pulse:')
    print(f'  Peak: {np.max(mf_power_db):.1f} dB')
    print(f'  Mean: {np.mean(mf_power_db):.1f} dB')
    print(f'  Peak location: sample {np.argmax(mf_power_db)}')
    
    # Expected target delays (from generator: 5km and 8km at c=3e8)
    c = 3e8
    fs = 100e6
    target1_range = 5000  # meters
    target2_range = 8000  # meters
    target1_delay_samples = int(2 * target1_range / c * fs)
    target2_delay_samples = int(2 * target2_range / c * fs)
    
    print(f'\nExpected target delays:')
    print(f'  Target 1 (5km): {target1_delay_samples} samples = {target1_delay_samples/fs*1e6:.1f} µs')
    print(f'  Target 2 (8km): {target2_delay_samples} samples = {target2_delay_samples/fs*1e6:.1f} µs')
    
    # Look at MF output around expected delays
    window = 200
    for tgt_name, tgt_delay in [('5km', target1_delay_samples), ('8km', target2_delay_samples)]:
        start = max(0, tgt_delay - window)
        end = min(len(mf_power_db), tgt_delay + window)
        local_peak = np.max(mf_power_db[start:end])
        local_peak_idx = start + np.argmax(mf_power_db[start:end])
        print(f'  {tgt_name}: peak {local_peak:.1f} dB at sample {local_peak_idx} (expected {tgt_delay})')
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Full pulse range profile
    range_samples = np.arange(len(mf_power_db))
    range_m = range_samples / fs * c / 2
    ax1.plot(range_m/1000, mf_power_db)
    ax1.axvline(5, color='r', linestyle='--', label='Target 1 (5km)')
    ax1.axvline(8, color='g', linestyle='--', label='Target 2 (8km)')
    ax1.set_xlabel('Range (km)')
    ax1.set_ylabel('Matched Filter Output (dB)')
    ax1.set_title('First Pulse - Matched Filter Range Profile')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 15)
    
    # Zoomed around targets
    ax2.plot(range_m/1000, mf_power_db)
    ax2.axvline(5, color='r', linestyle='--', label='Target 1 (5km)')
    ax2.axvline(8, color='g', linestyle='--', label='Target 2 (8km)')
    ax2.set_xlabel('Range (km)')
    ax2.set_ylabel('Matched Filter Output (dB)')
    ax2.set_title('Zoomed View - Target Region')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(3, 10)
    ax2.set_ylim(np.max(mf_power_db) - 40, np.max(mf_power_db) + 5)
    
    plt.tight_layout()
    plt.savefig('/Users/brian.day/git/crsd-inspector/mf_range_profile.png', dpi=150)
    print(f'\nPlot saved to mf_range_profile.png')
