import numpy as np
import matplotlib.pyplot as plt


def _rms(time_series, bin_size):
    """Computes the root mean square error across data segments

    Args:
        time_series: The time series
        bin_size: The number of non-overlapping points to be detrended and for which rmse will be calculated

    Returns:
        rmse - The root means squared error of the time series when detrended with non-overlapping windows of length
            bin_size

    """
    n_bins = int(len(time_series) / bin_size)
    trend_lines = _detrended_bins(time_series, bin_size)
    rmse = np.sqrt(np.mean((time_series[0:n_bins * bin_size] - trend_lines[0:n_bins * bin_size]) ** 2))

    return rmse


def _detrended_bins(time_series, bin_size):
    """Computes the linear trend in the time series for each non-overlapping window of length bin_size

    Args:
        time_series: The time series
        bin_size: The number of non-overlapping points to be detrended

    Returns:
        detrend_line - The linear trend for each window of length bin_size to be removed

    """
    detrend_line = np.zeros(len(time_series))

    for i in range(0, len(time_series), bin_size):
        if i + bin_size <= len(time_series):
            xval = list(range(i, i + bin_size))
            detrend_slope = np.polyfit(xval, time_series[xval], 1)
            detrend_line[xval] = np.polyval(detrend_slope, xval)

    return detrend_line


def dfa(time_series, bin_range, plot_dfa=True):
    """Computes the detrended fluctuation analysis (DFA) for a given time series.

     For a given time series, DFA is the root mean squared error between the time series and local linear trends in
     non-overlapping windows of length x. This procedure is repeated for multiple window lengths. The result of this
     process yields a slope when plotting the rmse against bin size in the :math:`log_2` scale.

    Args:
        time_series: The time series
        bin_range: A list of two values containing the smallest and the largest bin sizes of interest
        plot_dfa (optional): Default is true. When true, the :math:`log_2` plot of bin size and rmse is produced.

    Returns:
        alpha - The slope of the line fitting the bin sizes and rmse in the :math:`log_2` scale.

    """
    integrated_ts = np.cumsum(time_series - np.mean(time_series))

    bins = list(range(bin_range[0], bin_range[1] + 1, 1))
    fluctuations = np.zeros(len(bins))

    for n, bin_size in enumerate(bins):
        fluctuations[n] = _rms(integrated_ts, bin_size)

    alpha = np.polyfit(np.log2(bins), np.log2(fluctuations), 1)[0]

    if plot_dfa:
        fig, ax = plt.subplots()
        plt.plot(bins, fluctuations)
        ax.set_xscale('log', basex=2)
        ax.set_yscale('log', basey=2)
        plt.show()

    return alpha
