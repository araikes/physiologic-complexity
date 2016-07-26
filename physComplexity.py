from __future__ import unicode_literals

import itertools
import numpy as np
from math import factorial

def _course_grain(time_series, scale):
    data_points = len(time_series)
    data = np.zeros(int(data_points / scale) + (data_points % scale > 0))

    for i in range(int(data_points / scale) + (data_points % scale > 0)):
        data[i] = np.mean(time_series[i * scale: (i + 1) * scale])

    return data

def _moving_average(time_series, scale):
    data_points = len(time_series)
    data = np.zeros(data_points - scale + 1)

    for i in range(data_points - scale + 1):
        data[i] = np.mean(time_series[i:i + scale])

    return data

def _rms(time_series, bin_size):
	n_bins = int(len(time_series)/bin_size)
	trend_lines = _detrended_bins(time_series, bin_size)
	rmse = np.sqrt(np.mean((time_series[0:n_bins*bin_size] - trend_lines[0:n_bins*bin_size])**2))
	
	return rmse
	
def _detrended_bins(time_series, bin_size):
	detrend_line = np.zeros(len(time_series))
	
	for i in range(0, len(time_series), bin_size):
		xval = list(range(i, i + bin_size))
		detrend_slope = np.polyfit(xval, time_series[xval], 1)
		detrend_line[xval] = np.polyval(detrend_slope, xval)
	
	return detrend_line

def sample_entropy(data, m, r, delay):
    data_points = len(data)
    m_matches = 0
    m_plus_one_matches = 0

    for i in range(data_points - (m + 1) * delay):
        for j in range(i + delay, data_points - m * delay, 1):
            if (abs(data[i] - data[j]) < r and
                        abs(data[i + delay] - data[j + delay]) < r):
                m_matches += 1

                if abs(data[i + m * delay] - data[j + m * delay]) < r:
                    m_plus_one_matches += 1

    entropy = -np.log(m_plus_one_matches / m_matches)

    return entropy

def multiscale_entropy(time_series, scale, r):
    tol = r * np.std(time_series, ddof = 1)
    scale_values = [x + 1 for x in range(scale)]

    mse = [0] * scale

    for element in scale_values:
        course_data = _course_grain(time_series=time_series, scale = element)
        mse[element -1] = sample_entropy(data = course_data, m = 2, r = tol, delay = 1)

    return mse

def modified_multiscale_entropy(time_series, scale, r):
    tol = r * np.std(time_series, ddof = 1)
    scale_values = [x + 1 for x in range(scale)]
    
    mmse = [0]*scale

    for element in scale_values:
        averaged_data = _moving_average(time_series, element)
        mmse[element-1] = sample_entropy(data = averaged_data, m = 2, r = tol, delay = element)

    return mmse


def perm_entropy_norm(time_series, embed_dimension, delay):
    data_points = len(time_series)
    perm_list = np.array(list(itertools.permutations(range(embed_dimension))))

    c = [0] * len(perm_list)

    for i in range(data_points - delay * (embed_dimension - 1)):
        sorted_index_list = np.argsort(time_series[i:i + delay * embed_dimension:delay])

        for j in range(len(perm_list)):
            if abs(np.subtract(perm_list[j], sorted_index_list)).any == 0:
                c[j] += 1

    c = [element for element in c if element != 0]
    p = np.array(c) / float(sum(c))

    pe = -sum(p * np.log2(p))
    pe_norm = pe / np.log2(factorial(embed_dimension))

    entropy = [embed_dimension, delay, pe, pe_norm]
    return entropy

def dfa(time_series, bin_range):
	integrated_ts = np.cumsum(time_series - np.mean(time_series))
	
	bins = np.arange(bin_range[0], bin_range[1] + 1, 1)
	fluctuations = np.zeros(len(bin_size))
	
	for n, bin_size in enumerate(bins):
		fluctuations[n] = _rms(integrated_ts, bin_size)
	
	alpha = np.polyfit(np.log2(bins), np.log2(fluctuations), 1)
	
	return alpha