#!/usr/bin/env python3
"""Analyze PRF sequence in example_6.crsd"""

import numpy as np
import sarkit.crsd as crsd

with open('examples/example_6.crsd', 'rb') as f:
    reader = crsd.Reader(f)
    
    # Read PPP data - try TX1 first
    try:
        ppp = reader.read_ppps('TX1')
    except Exception as e:
        print(f"Could not read PPP for TX1: {e}")
        ppp = None
    
    if ppp is None or len(ppp) == 0:
        print("Error: Could not read pulse timing data")
        exit(1)
    
    # Extract pulse times
    if 'TxTime' not in ppp.dtype.names:
        print(f"Available fields: {ppp.dtype.names}")
        print("Error: TxTime field not found")
        exit(1)
    
    pulse_times = ppp['TxTime']
    
    # Calculate PRIs and PRFs
    pris = np.diff(pulse_times)
    prfs = 1.0 / pris
    
    print('First 40 pulses:')
    print('Pulse | PRI (μs) | PRF (Hz) | Type')
    print('-' * 50)
    for i in range(min(40, len(pris))):
        # Identify type
        prf = prfs[i]
        if abs(prf - 1500) < 1.0:
            ptype = 'High-1'
        elif abs(prf - 1600) < 1.0:
            ptype = 'High-2'
        elif abs(prf - 2000) < 1.0:
            ptype = 'Low-1'
        elif abs(prf - 2100) < 1.0:
            ptype = 'Low-2'
        else:
            ptype = 'Random'
        
        print(f'{i+1:3d}   | {pris[i]*1e6:8.2f} | {prfs[i]:8.1f} | {ptype}')
    
    print(f'\n{"="*50}')
    print('PRF Statistics:')
    print(f'  Min PRF: {prfs.min():.1f} Hz')
    print(f'  Max PRF: {prfs.max():.1f} Hz')
    print(f'  Mean PRF: {prfs.mean():.1f} Hz')
    print(f'  Std Dev: {prfs.std():.1f} Hz')
    
    # Count expected vs actual
    expected_prfs = [1500, 1600, 2000, 2100]
    print(f'\nExpected PRF counts:')
    for prf in expected_prfs:
        count = np.sum(np.abs(prfs - prf) < 1.0)
        print(f'  {prf} Hz: {count} pulses')
    
    random_count = np.sum((prfs < 1450) | (prfs > 2150))
    print(f'  Random (outside main set): {random_count} pulses')
    
    # Check sequence pattern
    print(f'\n{"="*50}')
    print('Sequence Analysis (20-pulse cycles):')
    for cycle in range(min(3, len(prfs) // 20)):
        start = cycle * 20
        end = start + 20
        cycle_prfs = prfs[start:end]
        print(f'\nCycle {cycle+1} (pulses {start+1}-{end}):')
        for i, prf in enumerate(cycle_prfs):
            if abs(prf - 1500) < 1.0:
                ptype = 'H1'
            elif abs(prf - 1600) < 1.0:
                ptype = 'H2'
            elif abs(prf - 2000) < 1.0:
                ptype = 'L1'
            elif abs(prf - 2100) < 1.0:
                ptype = 'L2'
            else:
                ptype = 'R'
            print(f'  P{start+i+1:3d}: {prf:7.1f} Hz ({ptype})', end='')
            if (i+1) % 4 == 0:
                print()
