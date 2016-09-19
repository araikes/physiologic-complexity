from __future__ import unicode_literals
import itertools
import numpy as np

def power_spectrum(ts, Fs, norm):
    if norm:
        ts = (ts - np.mean(ts))/np.std(ts, ddof=1)
    N = len(ts)
    fftx = np.fft.fft(ts)

    N2 = int(N/2)-1
    result = np.zeros((2, N2))

    amp = np.abs(fftx)[1:int(N/2)]
    freq = [i/N2 for i in range(N2)]
    freq = [i*(Fs/2) for i in freq]

    power = amp**2
    result = np.zeros((2, len(freq)))
    result[0] = freq
    result[1] = power

    return result

def average_power(ts, max_freq, bin_size, norm=False):



