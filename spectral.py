from __future__ import unicode_literals
import numpy as np


def power_spectrum(ts, Fs, norm):
    """

    Args:
        ts: The time-series values
        Fs: Sampling rate
        norm: A logical flag to center and standardize the time series prior to computing the FFT

    Returns:
        ts_freq: The frequencies associated with the FFT
        ts_pow: The power at each frequency

    """
    # Center and standardize time series
    if norm:
        ts = (ts - np.mean(ts)) / np.std(ts)

    N = len(ts)
    n = np.int(N / 2)

    # Use FFT to compute magnitude
    ts_fft = np.abs(np.fft.fft(ts))[1:N]

    # Compute power as the square of the magnitude and retrieve frequencies
    # Retain only the frequencies from 0 to Nyquist frequency (right-side of frequency spectrum
    ts_pow = (ts_fft ** 2)[1:n + 1]
    ts_freq = (np.arange(0, N - 1, 1) / (N * 1 / Fs))[1:n + 1]

    return ts_freq, ts_pow


def average_power(ts, Fs, bin_ends, norm):
    """

    Args:
        ts: The time-series
        Fs: The sampling rate
        bin_ends: A list of endpoints between which the average power is computed. DO NOT include 0
        norm: A logical flag to center and standardize the time series prior to computing the FFT

    Returns:
        avg_power: A list the length of bin_ends with the average power between the endpoints

    """
    # Get power spectrum
    frequencies, power = power_spectrum(ts, Fs, norm)

    # Compute indices of frequency bin endpoints
    bin_indices = sum([[i for i, x in enumerate(frequencies) if x == k] for k in bin_ends], [])

    # Normalize the power to max power in the considered range
    power = power / max(power[0:bin_indices[-1]])

    # Prepare bin_indices for use in computing average power.
    # To do this, we add 1 to the each value and prepend a 0 to the front of the list.
    # This way, we compute the sum power from bin_index[i] to (bin_index[i+1] - 1)
    bin_indices = [x + 1 for x in bin_indices]
    bin_indices.insert(0, 0)

    avg_power = [sum(power[bin_indices[i]:bin_indices[i + 1]]) for i, x in enumerate(bin_ends)]

    return avg_power
