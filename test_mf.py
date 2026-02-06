import numpy as np
from scipy.signal import correlate

# Test matched filtering
print('Testing matched filter correlation...')

# Create a simple chirp
N = 500
t = np.arange(N) / 100e6
chirp = np.exp(1j * 2 * np.pi * 10e6 * t**2)

# Create signal with chirp at position 1000
signal = np.zeros(10000, dtype=complex)
signal[1000:1000+N] = chirp * 10  # Amplitude 10

print(f'Signal shape: {signal.shape}')
print(f'Reference shape: {chirp.shape}')
print(f'Signal max at: {np.argmax(np.abs(signal))}')

# Matched filter - method 1: correlate with conjugate
mf1 = correlate(signal, np.conj(chirp), mode='same')
print(f'\nMethod 1 - correlate(signal, conj(ref)):')
print(f'  MF output max at: {np.argmax(np.abs(mf1))}')
print(f'  MF output max value: {np.max(np.abs(mf1)):.2e}')

# Matched filter - method 2: correlate without conjugate (scipy does it)
mf2 = correlate(signal, chirp, mode='same')
print(f'\nMethod 2 - correlate(signal, ref):')
print(f'  MF output max at: {np.argmax(np.abs(mf2))}')
print(f'  MF output max value: {np.max(np.abs(mf2)):.2e}')

# Matched filter - method 3: convolve with time-reversed conjugate
mf3 = np.convolve(signal, np.conj(chirp[::-1]), mode='same')
print(f'\nMethod 3 - convolve(signal, conj(ref[::-1])):')
print(f'  MF output max at: {np.argmax(np.abs(mf3))}')
print(f'  MF output max value: {np.max(np.abs(mf3)):.2e}')

print('\n=== Expected: peak at sample 1000+N/2 = 1250 ===')
