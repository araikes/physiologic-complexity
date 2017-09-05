import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)


def _detrending_coeff(win_len, order):
    """ Computes the detrending coefficient matrix to apply to the time series.


    Args:
        win_len: The number of points to include
        order: The polynomial order

    Returns:
        a: A win_length x order matrix
        coeff_output: The coefficients for each order from 0 to order

    """
    coeff_output = np.zeros((win_len, win_len), dtype = np.float64)
    if win_len % 2 == 0:
        n = int(win_len / 2)
        x = list(range(-n, n))
    else:
        n = int((win_len - 1) / 2)
        x = list(range(-n, n + 1))
    a = np.ones((win_len, order + 1))
    a[:, 1] = x

    for j in range(2, order + 1):
        a[:, j] = np.power(x, j)

    coeff_output = np.linalg.inv(a.T @ a) @ a.T

    return a, coeff_output


def _integrate_data(data):
    """ Creates an integrated time series. This is used when the original signal is white noise. The integration is the cumulative sum of the time series at each point.

    Args:
        data: The original time series

    Returns:
        integrated_data: The integrated time series.
    """
    data = data - np.mean(data)

    integrated_data = np.cumsum(data)

    return integrated_data


def _first_segment(coeff, seg_len, nonoverlap_len, fit_order, w, data):
    """ Computes the trend in the first data segment.

    Args:
        coeff: The trend-line coefficients
        seg_len: The number of points for detrending
        nonoverlap_len: The number of points to the left and right of the knitting region
        fit_order: The polynomial order for detrending
        w: The percentage of the segment for each point
        data: The first set of points of length seg_len in the time series

    Returns:
        record_x: The x-values of the trend
        record_y: The y-values of the trend

    """
    data_len = len(data)

    xi_left = list(range(0, seg_len))
    left_seg = data[xi_left]
    left_trend = coeff @ left_seg

    # Mid segment
    if seg_len + nonoverlap_len > data_len:
        xi_mid = list(range(nonoverlap_len, data_len))
        mid_seg = data[xi_mid]

        if len(mid_seg) < seg_len:
            if len(mid_seg) <= fit_order:
                a_tmp, coeff_tmp = _detrending_coeff(len(mid_seg), len(mid_seg) - 1)
                a_coeff_tmp = a_tmp @ coeff_tmp
                mid_trend = a_coeff_tmp @ mid_seg
            else:
                a_tmp, coeff_tmp = _detrending_coeff(len(mid_seg), fit_order)
                a_coeff_tmp = a_tmp @ coeff_tmp
                mid_trend = a_coeff_tmp @ mid_seg
        else:
            mid_trend = coeff @ mid_seg

        xx1 = left_trend[int((seg_len - 1) / 2):seg_len]
        xx2 = mid_trend[0:int((seg_len + 1) / 2)]
        xx_left = np.multiply(xx1, [1 - x for x in w]) + np.multiply(xx2, w)

        record_x = xi_left[0:nonoverlap_len]
        record_y = left_trend[0:nonoverlap_len]

        mid_start_index = xi_mid.index(next(i for i in xi_mid if xi_left[-1] == i - 1))
        try:
            mid_start_index
        except NameError:
            mid_start_index = None

        if mid_start_index is None:
            record_x.extend(xi_left[int((max(xi_left) + 3) / 2):])
            record_y.extend([xx_left[1:]])
        else:
            record_x.extend(xi_left[int((max(xi_left) + 1) / 2):] + xi_mid[mid_start_index:])
            record_y.extend([xx_left[0:],
                             mid_trend[int((seg_len + 3) / 2):]])

            return record_x, record_y
    else:
        xi_mid = list(range(nonoverlap_len, seg_len + nonoverlap_len))
        mid_seg = data[xi_mid]
        mid_trend = coeff @ mid_seg

    # Right Trend
    if 2 * (seg_len - 1) + 1 > data_len:
        xi_right = list(range(seg_len, data_len))
        right_seg = data[xi_right]

        if len(right_seg) < seg_len:
            coeff_tmp, a_tmp = _detrending_coeff(len(right_seg), fit_order)
            a_coeff_tmp = a_tmp @ coeff_tmp
            right_trend = a_coeff_tmp @ right_seg
        else:
            right_trend = coeff @ right_seg

        xx1 = left_trend[int((seg_len - 1) / 2):seg_len]
        xx2 = mid_trend[0:int((seg_len + 1) / 2)]
        xx_left = np.multiply(xx1, [1 - x for x in w]) + np.multiply(xx2, w).tolist()

        xx1 = mid_trend[int((seg_len - 1) / 2):seg_len]
        xx2 = right_trend[0:int((seg_len + 1) / 2)]
        xx_right = np.multiply(xx1, [1 - x for x in w]) + np.multiply(xx2, w).tolist()

        record_x = xi_left[0:nonoverlap_len]
        record_y = left_trend[0:nonoverlap_len]

        record_x.extend(xi_left[int(len(xi_left) / 2):] +
                        xi_mid[int((len(xi_mid) + 1) / 2):])

        record_y.extend(xx_left[0:] + xx_right[1:])

        right_start_index = xi_right.index(next(i for i in xi_right if xi_mid[-1] == i - 1))
        record_x.extend(xi_right[right_start_index:])
        record_y.extend(right_trend[right_start_index:])

        return record_x, record_y
    else:
        xi_right = list(range(seg_len - 1, (2 * (seg_len - 1) + 1)))
        right_seg = data[xi_right]
        right_trend = coeff @ right_seg

        xx1 = left_trend[int((seg_len - 1) / 2):seg_len]
        xx2 = mid_trend[0:int((seg_len + 1) / 2)]
        xx_left = (np.multiply(xx1, [1 - x for x in w]) + np.multiply(xx2, w)).tolist()

        xx1 = mid_trend[int((seg_len - 1) / 2):seg_len]
        xx2 = right_trend[0:int((seg_len + 1) / 2)]
        xx_right = (np.multiply(xx1, [1 - x for x in w]) + np.multiply(xx2, w)).tolist()

        record_x = xi_left[0:nonoverlap_len]
        record_x.extend(xi_left[int((len(xi_left)) / 2):] +
                        xi_mid[int((len(xi_mid) + 1) / 2):])

        record_y = (left_trend[0:nonoverlap_len]).tolist()
        record_y.extend(xx_left[0:] + xx_right[1:])

    return record_x, record_y


def _mid_segments(coeff, seg_index, seg_len, nonoverlap_len, w, data):
    """ Computes the trend for the middle data segments

    Args:
        coeff: The trend-line coefficients
        seg_len: The number of points for detrending
        nonoverlap_len: The number of points to the left and right of the knitting region
        fit_order: The polynomial order for detrending
        w: The percentage of the segment for each point
        data: The each set of points of length seg_len in the time series that are the first or last segments

    Returns:
        record_x: The x-values of the trend
        record_y: The y-values of the trend

    """
    xi_left = list(range((seg_index * (seg_len - 1)), ((seg_index + 1) * (seg_len - 1) + 1)))
    left_seg = data[xi_left]
    left_trend = coeff @ left_seg

    xi_mid = list(
        range(seg_index * (seg_len - 1) + nonoverlap_len, (seg_index + 1) * (seg_len - 1) + 1 + nonoverlap_len))
    mid_seg = data[xi_mid]
    mid_trend = coeff @ mid_seg

    xi_right = list(range((seg_index + 1) * (seg_len - 1), (seg_index + 2) * (seg_len - 1) + 1))
    right_seg = data[xi_right]
    right_trend = coeff @ right_seg

    xx1 = left_trend[list(range(int((seg_len - 1) / 2), seg_len))]
    xx2 = mid_trend[list(range(0, int((seg_len - 1) / 2) + 1))]
    xx_left = np.multiply(xx1, [1 - x for x in w]) + np.multiply(xx2, w)

    xx1 = mid_trend[list(range(int((seg_len - 1) / 2), seg_len))]
    xx2 = right_trend[list(range(0, int((seg_len - 1) / 2) + 1))]
    xx_right = np.multiply(xx1, [1 - x for x in w]) + np.multiply(xx2, w)

    record_x = xi_left[int((len(xi_left) + 2) / 2):] + xi_mid[int((len(xi_mid) + 1) / 2):]
    record_y = xx_left[1:].tolist() + xx_right[1:].tolist()

    return record_x, record_y


def _last_segment(coeff, seg_index, seg_len, nonoverlap_len, fit_order, w, data):
    """ Computes the trend for the last data segment

    Args:
        coeff: The trend-line coefficients
        seg_len: The number of points for detrending
        nonoverlap_len: The number of points to the left and right of the knitting region
        fit_order: The polynomial order for detrending
        w: The percentage of the segment for each point
        data: The last set of points of length seg_len in the time series

    Returns:
        record_x: The x-values of the trend
        record_y: The y-values of the trend

    """
    xi_left = list(range(seg_index * (seg_len - 1), (seg_index + 1) * (seg_len - 1) + 1))
    left_seg = data[xi_left]
    left_trend = coeff @ left_seg

    if (seg_index + 1) * (seg_len - 1) + 1 + nonoverlap_len > len(data):
        xi_mid = list(range(seg_index * (seg_len - 1) + nonoverlap_len, len(data)))
        mid_seg = data[xi_mid]

        if len(mid_seg) < seg_len:
            if len(mid_seg) <= fit_order:
                a_tmp, coeff_tmp = _detrending_coeff(len(mid_seg), len(mid_seg) - 1)
                a_coeff_tmp = a_tmp @ coeff_tmp
                mid_trend = a_coeff_tmp @ mid_seg
            else:
                a_tmp, coeff_tmp = _detrending_coeff(len(mid_seg), fit_order)
                a_coeff_tmp = a_tmp @ coeff_tmp
                mid_trend = a_coeff_tmp @ mid_seg

        else:
            mid_trend = coeff @ mid_seg

        xx1 = left_trend[list(range(int((seg_len - 1) / 2), seg_len))]
        xx2 = mid_trend[list(range(0, int((seg_len - 1) / 2 + 1)))]
        xx_left = np.multiply(xx1, [1 - x for x in w]) + np.multiply(xx2, w)

        mid_start_index = xi_left.index(next(i for i in xi_left if xi_mid[0] == i - 1))
        try:
            mid_start_index
        except NameError:
            mid_start_index = None

        if mid_start_index is None:
            record_x = xi_left[int((len(xi_left) + 2) / 2):]
            record_y = xx_left[1:]
        else:
            mid_trend = mid_trend.tolist()
            record_x = xi_left[int((len(xi_left) + 2) / 2):] + xi_mid[mid_start_index:]
            record_y = xx_left[1:].tolist() + mid_trend[int((seg_len + 2) / 2):]

        return record_x, record_y
    else:
        xi_mid = list(
            range(seg_index * (seg_len - 1) + nonoverlap_len, (seg_index + 1) * (seg_len - 1) + 1 + nonoverlap_len))
        mid_seg = data[xi_mid]
        mid_trend = coeff @ mid_seg

    xi_right = list(range((seg_index + 1) * (seg_len - 1), len(data)))
    right_seg = data[xi_right]

    if len(right_seg) < seg_len:
        if len(right_seg) <= fit_order:
            a_tmp, coeff_tmp = _detrending_coeff(len(right_seg), len(right_seg) - 1)
            a_coeff_tmp = a_tmp @ coeff_tmp
            right_trend = a_coeff_tmp @ right_seg
        else:
            a_tmp, coeff_tmp = _detrending_coeff(len(right_seg), fit_order)
            a_coeff_tmp = a_tmp @ coeff_tmp
            right_trend = a_coeff_tmp @ right_seg
    else:
        right_trend = coeff @ right_seg

    xx1 = left_trend[int((seg_len - 1) / 2):seg_len]
    xx2 = mid_trend[0:int((seg_len - 1) / 2 + 1)]
    xx_left = np.multiply(xx1, [1 - x for x in w]) + np.multiply(xx2, w)

    xx1 = mid_trend[int((seg_len - 1) / 2):seg_len]
    xx2 = right_trend[0:int((seg_len - 1) / 2 + 1)]
    xx_right = np.multiply(xx1, [1 - x for x in w]) + np.multiply(xx2, w)

    right_start_index = xi_mid.index(next(i for i in xi_mid if xi_right[0] == i - 1))

    record_x = xi_left[int((len(xi_left) + 2) / 2):] + \
               xi_mid[int((len(xi_mid) + 1) / 2):] + \
               xi_right[right_start_index:]

    record_y = xx_left[1:].tolist() + xx_right[1:].tolist()
    record_y.extend(right_trend[right_start_index:])

    return record_x, record_y


def detrending_method(data, seg_len, fit_order):
    # Persistent variable definitions
    """ Detrends the time series over segments of length seg_len using a fit_order polynomial

    Args:
        data: The time series
        seg_len: The number of points in each detrending segment
        fit_order: The polynomial order for detrending

    Returns:
        detrended_data: The detrended data
        trend: The trend line

    """
    data_len = len(data)
    nonoverlap_len = int((seg_len - 1) / 2)
    w = [x / nonoverlap_len for x in list(range(0, nonoverlap_len + 1))]
    a, coeff = _detrending_coeff(seg_len, fit_order)
    a_coeff = a @ coeff

    for seg_index in list(range(0, int(np.floor(int(data_len - 1) / (seg_len - 1))))):
        if seg_index == 0:
            index, trend = _first_segment(a_coeff, seg_len, nonoverlap_len, fit_order, w, data)

        elif 0 < seg_index < int(np.floor(int(data_len - 1) / (seg_len - 1))) - 1:
            index_tmp, trend_tmp = _mid_segments(a_coeff, seg_index, seg_len, nonoverlap_len, w, data)
            index.extend(index_tmp)
            trend.extend(trend_tmp)

        else:
            index_tmp, trend_tmp = _last_segment(a_coeff, seg_index, seg_len, nonoverlap_len, fit_order, w,
                                                 data)
            index.extend(index_tmp)
            trend.extend(trend_tmp)

    detrended_data = data - trend

    return detrended_data, trend


def multi_detrending(data, step_size, q, order, random_walk=True, plot_trend=False):
    """ Applies the adaptive fractal detrending method to data.

    Args:
        data: The original time series
        step_size: The resolution for determining the number of points in each segment. step_size = 1 is generally sufficient.
        q: The q-spectrum. When using a list, the value results in a multifractal formulation
        order: The order of the polynomial for detrending
        random_walk: A logical flag indicating whether the time series is a random walk or white noise. If False, then the time series is first integrated. Default is True.
        plot_trend: A logical flag indicating whether to plot the results. Default is False.

    Returns:
        detrended_data: The detrended time series
        trend: The trend at each x-value
        result: The fractal scaling coefficient

    References:
        Gao J, Hu J, Tung W. Facilitating joint chaos and fractal analysis of biosignals through nonlinear adaptive filtering. PloS one. 2011;6(9):e24331.


    """
    if isinstance(q, int):
        q = [q]

    if not random_walk:
        data = _integrate_data(data)

    data_len = len(data)
    max_seg_index = int(np.log2(data_len))

    segments = [int(2 ** i + 1) for i in range(1, max_seg_index, step_size)]
    segments = [i + 1 if i % 2 == 0 else i for i in segments]

    result = np.zeros((len(q) + 1, int(np.floor(int((max_seg_index - 2) / step_size) + 1))))
    detrended_data = np.zeros((max_seg_index, data_len))
    trend = np.zeros((max_seg_index, data_len))

    detrended_data[0:, ] = data
    trend[0:, ] = data

    for index, seg_len in enumerate(segments):
        detrended_data[index + 1:, ], trend[index + 1:,] = detrending_method(data, seg_len, order)

        result[0, index] = seg_len
        for i in list(range(0, len(q))):
            result[i + 1, index] = sum(abs(detrended_data[index + 1]) * q[i]) / (len(detrended_data[index + 1]) - 1) \
                                                                                ** (1 / q[i])

    if plot_trend:
        plt.plot(trend[0], label = 'Original Data')
        for i in range(1, max_seg_index):
            print(segments[i-1])
            plt.plot(trend[i], label = str(segments[i-1]))
        plt.legend(loc='best')

    return detrended_data, trend, result
