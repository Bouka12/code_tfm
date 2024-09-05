# select_intervals.py
# select intervals is adapted to select intervals for the interval-method combination considering the number of representation.

import numpy as np

def select_intervals_dr_cif(X, n_representations, random_state=None):
    """
    IS0: select_intervals_dr_cif
    Select random intervals from time series data as has been implemented in the Dr-CIF classifier.

    Parameters
    ----------
    X : np.ndarray
        Time series data of shape (n_samples, n_timepoints).
    n_representations : int, default=3
        Number of representations (e.g., original series, first-order differences, periodogram).
    random_state : int or None, default=None
        Seed for the random number generator.

    Returns
    -------
    selected_intervals : list of lists of tuples
        List of lists where each sublist contains tuples of intervals (start, end) for each representation.
    """
    n_samples, n_timepoints = X.shape
    n_intervals = int(np.sqrt(n_timepoints)/n_representations) + 4  # adaptive interval count
    min_interval_length = 3
    max_interval_length = int(0.5 * n_timepoints)  # 50% of the series length
    
    if random_state is not None:
        np.random.seed(random_state)

    intervals = []
    for _ in range(n_intervals):
        interval_length = np.random.randint(min_interval_length, max_interval_length + 1)
        start = np.random.randint(0, n_timepoints - interval_length + 1)
        end = start + interval_length
        intervals.append((start, end))
    return intervals

def select_quant_intervals(X, interval_depth=6, random_state=None):
    """
    IS2: select_quant_intervals
    Select intervals for the QUANT classifier.

    Parameters
    ----------
    X : np.ndarray
        Time series data of shape (n_samples, n_channels, n_timepoints).
    interval_depth : int, default=6
        The depth to stop extracting intervals at. The number of intervals at each level
        is `2 ** level` (starting at 0) for each level.
    random_state : int or None, default=None
        Seed for the random number generator.

    Returns
    -------
    selected_intervals : list of lists of tuples
        List of intervals (start, end) for each depth level.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_timepoints = X.shape
    selected_intervals = []

    for level in range(interval_depth + 1):
        interval_length = n_timepoints // (2 ** level)
        intervals = []
        start = 0
        
        while start + interval_length <= n_timepoints:
            end = start + interval_length
            intervals.append((start, end))
            start += interval_length // 2  # Shift by half the interval length
        
        selected_intervals.append(intervals)
    
    return selected_intervals


