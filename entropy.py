from __future__ import unicode_literals
import itertools
import numpy as np
from math import factorial


def _course_grain(time_series, tau, method):
    """Creates non-overlapping windows of length tau and averages the values.

    Args:
        time_series: The time series data to be course-grain averaged
        tau (int): The scale factor
        method (str): The method for course-graining. This may be "mean" for average or "var" for variance. "var" is
            only used when called from generalized.

    Returns:
        data - an ndarray of values containing the average values in each window

    """

    data_points = len(time_series)
    data = np.zeros(int(data_points / tau))

    if method == "mean":
        for i in range(int(data_points / tau)):
            data[i] = np.mean(time_series[i * tau: (i + 1) * tau])
    else:
        for i in range(int(data_points / tau)):
            data[i] = np.var(time_series[i * tau: (i + 1) * tau], ddof=1)

    return data


def _moving_average(time_series, tau):
    """ Creates overlapping windows of length tau and averages the values.

    Args:
        time_series: The time series to be averaged
        tau (int): The scale factor

    Returns:
        data - an ndarray containing the average values in each window.

    """

    data_points = len(time_series)
    data = np.zeros(data_points - tau + 1)

    for i in range(data_points - tau + 1):
        data[i] = np.mean(time_series[i:i + tau])

    return data


def sample_entropy(data, m, r, delay):
    r"""Computes sample entropy for a time series.

    For a given time series, first, construct template vectors of length m as
    :math:`x_i^m(\\tau) = \{x_i, x_{(i + \delta)} ... x_{(i + (m-1)\delta)}\}, \qquad 1 \leq i \leq N - m\delta`

    Second, calculate the Euclidean distance between each pair of template vectors such that:
    :math:`d_{ij}^m = \|(x_i^m(\delta) - x_j^m(\delta))\|_{\infty}, \qquad 1 \leq i,j \leq N -m\delta, \qquad  j > i +
    \delta`

    n(m, \delta, r) is the total number of vectors pairs such that :math:`d_{ij}^m(\delta) \leq r`. This process is
    repeated for m + 1.

    Sample entropy is then computed as:
    :math:`SampEn(x, m, \delta, r) = -ln(\frac{n(m + 1, \delta, r)}{n(m, \delta, r)})`


    Args:
        data: The time series
        m (int): The length of template vectors to be compared
        r: The tolerance band around the vector endpoints
        delay: The number of time series values between template vector endpoints.
            A delay value of one is a unity delay and denotes consecutive time series points.

    Returns:
        entropy - The sample entropy of the time series.

    References:
        Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis using approximate entropy
        and sample entropy. American Journal of Physiology-Heart and Circulatory Physiology, 278(6), H2039–H2049.

        Govindan, R. B., Wilson, J. D., Eswaran, H., Lowery, C. L., & Preißl, H. (2007). Revisiting sample entropy
        analysis. Physica A: Statistical Mechanics and Its Applications, 376, 158–164.
        http://doi.org/10.1016/j.physa.2006.10.077
    """
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


def multiscale_entropy(time_series, tau, r, status):
    """ Computes the entropy of a time series at multiple time scales using a course-grain averaging method.

    To accomplish this, the sample entropy of the time series is computed. The series is then course-grain averaged by
    values from :math:`2 \leq i \leq \\tau` and sample entropy is recomputed.

    Args:
        time_series: The time series
        tau (int): The maximum scale value
        r: The percentage, in decimal form, of the time series standard deviation to form tolerance bands for sample
            entropy.
        status: When true, print a progress indicator after each tau value has run.

    Returns:
        mse - An ndarray of length tau with the sample entropy at each scale value.

    References:
        Costa, M., Goldberger, A. L., & Peng, C.-K. (2002). Multiscale entropy analysis of complex physiologic time
        series. Physical Review Letters, 89(6), 068102.

    """
    tol = r * np.std(time_series, ddof=1)
    scale_values = [x + 1 for x in range(tau)]

    mse = np.zeros(tau)

    for i, scale_val in enumerate(scale_values):
        course_data = _course_grain(time_series=time_series, tau=scale_val, method="mean")
        mse[i] = sample_entropy(data=course_data, m=2, r=tol, delay=1)

        if status:
            print('Scale value ', scale_val, 'completed')

    return mse


def comp_multiscale_entropy(time_series, tau, r, status):
    """Computes composite multiscale entropy from a time series.

    The time series is course-grain averaged is averaged over windows of length :math:`\\tau_i`. For each :math:`\\tau_i`,
    the course-graining is computed on the original signal and at offsets from the first data value ranging from
    :math:`1 \leq k \leq \\tau_i`. This yields :math:`\\tau_i` course-grained time series for each :math:`\\tau_i`.
    Sample entropy is calculated for all :math:`\\tau_i` time series and averaged.

    Args:
        time_series: The time series
        tau (int): The maximum scale value
        r: The percentage, in decimal form, of the time series standard deviation to form tolerance bands for sample
            entropy.
        status: When true, print a progress indicator after each tau value has run.

    Returns:
        cmse - An ndarray of length tau of the the average sample entropy at each tau value.

    References:
        Wu, S.-D., Wu, C.-W., Lin, S.-G., Wang, C.-C., & Lee, K.-Y. (2013). Time Series Analysis Using Composite
        Multiscale Entropy. Entropy, 15(3), 1069–1084. http://doi.org/10.3390/e15031069
    """
    tol = r * np.std(time_series, ddof=1)
    scale_values = [x + 1 for x in range(tau)]

    csme = np.zeros(tau)

    for i, scale_val in enumerate(scale_values):
        for j in range(i):
            course_data = _course_grain(time_series[j:len(time_series)], scale_val, method="mean")
            csme[i] += sample_entropy(data=course_data, m=2, r=tol, delay=1) / i

        if status:
            print('Scale value ', scale_val, 'completed')

    return csme


def generalized_multiscale_entropy(time_series, tau, r, status=True):
    """Computes generalized multiscale entropy.

    For this method of multiscale entropy, the course-grain variance is computed for the time series in non-overlapping
    windows of length :math:`\\tau_i`. Sample entropy is then computed on each course-grained variance series.

    Args:
        time_series: The time series
        tau (int): The maximum scale value
        r: The percentage, in decimal form, of the time series standard deviation to form tolerance bands for sample
            entropy.
        status: When true, print a progress indicator after each tau value has run.

    Returns:
        gmse - An ndarray of length tau sample entropy of the variance at each value of tau.

    References:
       Costa, M., & Goldberger, A. (2015). Generalized Multiscale Entropy Analysis: Application to Quantifying the
       Complex Volatility of Human Heartbeat Time Series. Entropy, 17(3), 1197–1203. http://doi.org/10.3390/e17031197
    """
    tol = r * np.std(time_series, ddof=1)
    scale_values = [x + 1 for x in range(tau)]

    gmse = np.zeros(tau)

    for i, scale_val in enumerate(scale_values):
        course_data = _course_grain(time_series, scale_val, method="var")
        gmse[i] = sample_entropy(data=course_data, m=2, r=tol, delay=1)

        if status:
            print('Scale value ', scale_val, 'completed')

    return gmse


def modified_multiscale_entropy(time_series, tau, r, status=True):
    """Computes modified multiscale entropy.

    This function is designed for short time series. For each value of tau, the data is averaged in overlapping moving
    windows of length tau. Then, for each moving-averaged time series, the sample entropy is computed with
    :math:`\delta = \\tau_i`.

    Note that this procedure is time-consuming due to the number of template vectors per tau value time series.

    Args:
        time_series: The time series of interest
        tau: The maximum scale value
        r: The percentage, in decimal form, of the time series standard deviation to form tolerance bands for sample
            entropy.
        status: When true, print a progress indicator after each tau value has run.

    Returns:
        mmse - An ndarray of length tau with the sample entropy for each value of :math:`\\tau_i`.

    References:
        Wu, S.-D., Wu, C.-W., Lee, K.-Y., & Lin, S.-G. (2013). Modified multiscale entropy for short-term time series
        analysis. Physica A, 392(23), 5865–5873. http://doi.org/10.1016/j.physa.2013.07.075
    """
    tol = r * np.std(time_series, ddof=1)
    scale_values = [x + 1 for x in range(tau)]

    mmse = np.zeros(tau)

    for i, scale_val in enumerate(scale_values):
        averaged_data = _moving_average(time_series, scale_val)
        mmse[i] = sample_entropy(data=averaged_data, m=2, r=tol, delay=scale_val)

        if status:
            print('Scale value ', scale_val, 'completed')

    return mmse


def perm_entropy_norm(time_series, embed_dimension, delay):
    """Computes permutuation and normalized permutation entropy for a given time series.

    Args:
        time_series: The time series
        embed_dimension: The maximum of number of points to consider
        delay: The delay between points in the time series when forming vectors of length embed_dimension

    Returns:
        entropy - A list including the embedding dimension, the delay, the permuation entropy, and normalized
            permutation entropy

    References:
        Bandt, C., & Pompe, B. (2002). Permutation entropy: a natural complexity measure for time series. Physical
        Review Letters, 88(17), 174102.
    """
    data_points = len(time_series)
    perm_list = np.array(list(itertools.permutations(range(embed_dimension))))

    c = np.zeros(len(perm_list))

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
