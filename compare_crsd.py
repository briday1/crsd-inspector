from sarkit import crsd
import numpy as np

print("=== example_2.crsd ===")
with open('/Users/brian.day/git/crsd-inspector/examples/example_2.crsd', 'rb') as f:
    reader = crsd.Reader(f)
    root = reader.metadata.xmltree.getroot()
    
    channels = root.findall('.//{http://api.nsgreg.nga.mil/schema/crsd/1.0}Channel')
    channel_ids = [ch.find('{http://api.nsgreg.nga.mil/schema/crsd/1.0}ChId').text 
                  for ch in channels] if channels else []
    
    data = reader.read_signal(channel_ids[0])
    print(f'Data shape: {data.shape}')
    print(f'Format: {data.shape[0]} vectors x {data.shape[1]} samples')
    
    # Check power
    power = np.abs(data) ** 2
    power_db = 10 * np.log10(power + 1e-12)
    print(f'Power: max={np.max(power_db):.1f} dB, mean={np.mean(power_db):.1f} dB')
    
    # Check if TX waveform exists
    try:
        tx_wfm_array = reader.read_support_array("TX_WFM")
        print(f'TX waveform: {tx_wfm_array.shape}')
    except:
        print('No TX waveform')

print("\n=== example_continuous.crsd ===")
with open('/Users/brian.day/git/crsd-inspector/examples/example_continuous.crsd', 'rb') as f:
    reader = crsd.Reader(f)
    root = reader.metadata.xmltree.getroot()
    
    channels = root.findall('.//{http://api.nsgreg.nga.mil/schema/crsd/1.0}Channel')
    channel_ids = [ch.find('{http://api.nsgreg.nga.mil/schema/crsd/1.0}ChId').text 
                  for ch in channels] if channels else []
    
    data = reader.read_signal(channel_ids[0])
    print(f'Data shape: {data.shape}')
    print(f'Format: {data.shape[0]} vectors x {data.shape[1]} samples')
    
    # Check power
    power = np.abs(data) ** 2
    power_db = 10 * np.log10(power + 1e-12)
    print(f'Power: max={np.max(power_db):.1f} dB, mean={np.mean(power_db):.1f} dB')
    
    # Check fraction of non-zero samples
    nonzero = np.sum(power > 1e-20)
    print(f'Non-zero samples: {nonzero:,} / {data.size:,} ({100*nonzero/data.size:.1f}%)')
    
    # Check if TX waveform exists
    try:
        tx_wfm_array = reader.read_support_array("TX_WFM")
        print(f'TX waveform: {tx_wfm_array.shape}')
    except:
        print('No TX waveform')
