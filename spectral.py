from __future__ import unicode_literals
import itertools
import numpy as np

def power_spectrum(ts, Fs, norm, one_sided = True):
    if norm:
        ts = (ts - np.mean(ts))/np.std(ts, ddof=1)
    N = len(ts)
    dt = 1/Fs

    ts_fft = np.abs(np.fft.fft(ts))**2

    if one_sided:
        ts_fft = ts_fft[0:np.int(np.floor(N/2))]

    mean_square_power = ts_fft/N
    PSD = dt * mean_square_power

    if one_sided:
        PSD[1:-1] = 2*PSD[1:-1]

    return PSD

def average_power(ts, Fs, max_freq, bin_size, norm, one_sided = True):
    N = len(ts)
    df = Fs/N

    power_analysis = power_spectrum(ts, Fs, norm, one_sided)

    bins = np.arange(0,max_freq + 1,bin_size)
    avg_power = np.zeros((1,len(bins)-1))


    for i in range(len(bins)-1):
        avg_power[0,i] = df*sum(power_analysis[np.int(np.floor(bins[i]*N/Fs)):np.int(np.floor(bins[i+1]*N/Fs))])

    return avg_power


def average_power(ts, Fs, max_freq, bin_size, norm):
    power_analysis = power_spectrum(ts, Fs, norm)

    bins = np.delete(np.arange(0, max_freq + 1, bin_size), 0)
    bin_index = power_analysis[0].searchsorted(bins)
    binned_spectrum = np.split(power_analysis[1], bin_index)

    avg_power = [np.mean(binned_spectrum[i]) for i, val in enumerate(bins)]

    return avg_power




Band = [0,1,2,3,4,5,6,7,8,9,10,11,12]
C  = np.fft.fft(x)
C = abs(C)
Power = np.zeros(len(Band)-1)


for Freq_Index in range(len(Band)-1):
    Freq = float(Band[Freq_Index])
    Next = float(Band[Freq_Index+1])
    print(Freq)
    Power[Freq_Index] = 2*sum(Pxx[np.floor(Freq*N/Fs):np.floor(Next*N/Fs)])/N
Power_ratio = Power/sum(Power)

Power5 = np.zeros(len(Band)-1)
for Freq_Index in range(len(Band)-1):
    Freq = float(Band[Freq_Index])
    Next = float(Band[Freq_Index+1])
    Power6[Freq_Index] = np.trapz(test_c[1,np.floor(Freq/Fs*N):np.floor(Next/Fs*N)])
Power5_ratio = Power5/sum(Power5)