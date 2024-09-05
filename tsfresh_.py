import numpy as np
import pandas as pd
from tsfresh import extract_features
from aeon.transformations.collection.feature_based import TSFreshFeatureExtractor

def _from_3d_numpy_to_long(arr):
    """
    Convert a 3D numpy array (univariate time series) into long format for tsfresh.

    Parameters
    ----------
    arr : numpy.ndarray
        A 3D numpy array of shape (n_cases, 1, n_timepoints).

    Returns
    -------
    df : pandas.DataFrame
        DataFrame in long format expected by tsfresh with columns:
        'index', 'time_index', 'dimension', 'value'.
    """
    n_cases, n_channels, n_timepoints = arr.shape

    # Reshape into 2D for easier processing, ignoring the channel as it's univariate (1 channel)
    df = pd.DataFrame(arr.reshape(n_cases * n_timepoints, 1), columns=["value"])
    df["case_index"] = np.repeat(np.arange(n_cases), n_timepoints)
    df["time_index"] = np.tile(np.arange(n_timepoints), n_cases)
    df["dimension"] = "dim_0"  # Only one dimension for univariate data

    # Return DataFrame with correct order for tsfresh
    return df[["case_index", "time_index", "dimension", "value"]].rename(
        columns={"case_index": "index"}
    )
# Define the function for extracting interval features
def extract_interval_features(X, intervals):
    """
    Extract features for each interval in the time series.

    Parameters
    ----------
    X : numpy.ndarray
        A 3D numpy array (n_cases, n_channels, n_timepoints) representing the time series.
    intervals : list of tuples
        A list of intervals, where each interval is a tuple (start, end).
    tsfresh_extractor : TSFreshFeatureExtractor
        A TSFreshFeatureExtractor object configured for feature extraction.

    Returns
    -------
    interval_features : pandas.DataFrame
        A DataFrame containing the extracted features for each interval.
    """
    n_cases, n_channels, n_timepoints = X.shape
    interval_features_list = []

    for (start, end) in intervals:
        # Select the interval slice for each case and channel
        X_interval = X[:, :, start:end]
        
        # Reshape the interval to the long format expected by TSFresh
        Xt = _from_3d_numpy_to_long(X_interval)
        # Initialize the TSFreshFeatureExtractor
        tsfresh_extractor = TSFreshFeatureExtractor(default_fc_parameters="efficient")
        
        # Extract features using the tsfresh extractor
        Xt_features = tsfresh_extractor._transform(X_interval)
        
        # Add the features for this interval to the list
        interval_features_list.append(Xt_features)

    # Concatenate features from all intervals
    interval_features = pd.concat(interval_features_list, axis=1)
    
    return interval_features

'''
n_cases, n_channels, n_timepoints = 10, 1, 100  # Example data dimensions
X = np.random.rand(n_cases, n_channels, n_timepoints)  # Randomly generated time series data

from select_intervals import select_intervals_dr_cif
X_ =X.reshape(X.shape[0], X.shape[2])
intervals = select_intervals_dr_cif(X_, 1, 2025)
print(f"shape of intervals selected by cif for each time series: ", len(intervals))
# Define the intervals you want to use
#intervals = [(0, 20), (20, 50), (50, 80)]  # Example intervals

# Initialize the TSFreshFeatureExtractor
tsfresh_extractor = TSFreshFeatureExtractor(default_fc_parameters="efficient")

# Extract features for the defined intervals
#interval_features = extract_interval_features(X, intervals, tsfresh_extractor)
#print(interval_features.shape)
'''