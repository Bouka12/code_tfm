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


import numpy as np

def select_quant_intervals(X, depth, min_interval_size=3):
    """
    Select intervals for QUANT with a minimum interval size constraint.

    Parameters
    ----------
    X : np.ndarray
        Time series data of shape (n_samples, n_timepoints).
    depth : int
        Depth for the interval selection.
    min_interval_size : int, default=3
        Minimum allowed interval size to avoid small, uninformative intervals.

    Returns
    -------
    np.ndarray
        Array of selected intervals (start, end).
    """
    input_length = X.shape[-1]
    
    # Calculate the exponent, limiting the depth based on the input length
    exponent = min(depth, int(np.log2(input_length)) + 1)
    
    intervals = []

    # Loop over powers of 2 to generate intervals
    for n in 2 ** np.arange(exponent):
        # Generate indices using linspace and convert them to integers
        indices = np.linspace(0, input_length, n + 1, dtype=int)

        # Create intervals as pairs of start and end indices
        intervals_n = np.stack((indices[:-1], indices[1:]), axis=1)
        
        # Filter intervals based on the minimum interval size
        intervals_n = intervals_n[(intervals_n[:, 1] - intervals_n[:, 0]) >= min_interval_size]
        
        if len(intervals_n) > 0:
            intervals.append(intervals_n)

        # If n > 1 and the median difference is greater than 1, apply shifting logic
        if n > 1 and np.median(np.diff(intervals_n, axis=1)) > 1:
            shift = int(np.ceil(input_length / n / 2))
            shifted_intervals = intervals_n[:-1] + shift
            
            # Ensure shifted intervals stay within bounds
            shifted_intervals = shifted_intervals[shifted_intervals[:, 1] <= input_length]
            
            # Filter shifted intervals by minimum size
            shifted_intervals = shifted_intervals[(shifted_intervals[:, 1] - shifted_intervals[:, 0]) >= min_interval_size]
            
            if len(shifted_intervals) > 0:
                intervals.append(shifted_intervals)

    # Concatenate all intervals into one array
    if len(intervals) > 0:
        return np.vstack(intervals)
    else:
        return np.empty((0, 2), dtype=int)  # Return an empty array if no valid intervals


'''
def select_quant_intervals(X, interval_depth=6, random_state=None):
    """
    IS2: select_quant_intervals
    Select intervals for the QUANT classifier.

    Parameters
    ----------
    X : np.ndarray
        Time series data of shape (n_samples, n_timepoints).
    interval_depth : int, default=6
        The depth to stop extracting intervals at. The number of intervals at each level
        is `2 ** level` (starting at 0) for each level.
    random_state : int or None, default=None
        Seed for the random number generator.

    Returns
    -------
    intervals : np.ndarray
        Array of intervals (start, end) for all depth levels.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_timepoints = X.shape

    # Correctly calculate total number of intervals
    total_intervals = 0
    for level in range(interval_depth + 1):
        interval_length = n_timepoints // (2 ** level)
        num_intervals = 0
        start = 0
        while start + interval_length <= n_timepoints:
            num_intervals += 1
            start += interval_length // 2  # Shift by half the interval length
        total_intervals += num_intervals

    # Preallocate a numpy array to store (start, end) interval pairs
    intervals = np.zeros((total_intervals, 2), dtype=int)
    
    index = 0  # To track the position in the preallocated array

    for level in range(interval_depth + 1):
        interval_length = n_timepoints // (2 ** level)
        start = 0
        
        # Efficiently create intervals without exceeding array bounds
        while start + interval_length <= n_timepoints:
            end = start + interval_length
            intervals[index] = [start, end]
            index += 1
            start += interval_length // 2  # Shift by half the interval length
    
    return intervals
'''

'''
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
    # Flatten the list
    flattened_list_intervals = [item for sublist in selected_intervals for item in sublist]
    
    return flattened_list_intervals
'''

def get_intervals(X, y, n_representations, interval_depth, select_intervals_func, random_state):
    if select_intervals_func == select_intervals_dr_cif:
        intervals = select_intervals_dr_cif(X, n_representations, random_state)
    elif select_intervals_func == select_quant_intervals:
        intervals = select_quant_intervals(X, interval_depth)
    return intervals


