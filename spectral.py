from __future__ import unicode_literals
import itertools
import numpy as np

def power_spectrum(ts, Fs, norm):
    if norm:
        ts = (ts - np.mean(ts))/np.std(ts)

    N = len(ts)
    n = np.int(N/2)

    ts_fft = np.abs(np.fft.fft(ts))[1:N]

    ts_pow = (ts_fft**2)[1:n+1]
    ts_freq = (np.arange(0, N-1, 1)/(N*1/Fs))[1:n+1]

    return ts_freq, ts_pow

def average_power(ts, Fs, max_freq, bin_size, norm):
    N = len(ts)

    frequencies, power = power_spectrum(ts, Fs, norm, one_sided)

    bins = np.arange(0,max_freq + 1,bin_size)
    avg_power = np.zeros((1,len(bins)-1))


    for i in range(len(bins)-1):
        avg_power[0,i] = sum(power[np.int(np.floor(bins[i]*N/Fs)):np.int(np.floor(bins[i+1]*N/Fs))])

    return avg_power