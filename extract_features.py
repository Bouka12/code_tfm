import numpy as np
import pycatch22
from pycatch22 import catch22_all 
from scipy.stats import iqr

def get_slope(interval):
    """
    Calculates the slope of the linear fit for the interval.

    Parameters:
    sub_interval (numpy.ndarray): 1D array representing the time series data.

    Returns:
    float: The slope value.
    """
    n = len(interval)
    p = np.arange(1, n + 1)
    p_mean = np.mean(p)
    interval_mean = np.mean(interval)
    
    # Calculate the numerator and denominator for the slope
    num = np.dot(p - p_mean, interval - interval_mean)  # Numerator
    den = np.dot(p - p_mean, p - p_mean)  # Denominator
    
    # Slope is simply num / den without the arctangent
    slope = num / den
    
    return slope

# 7 summary statistics
def calculate_summary_statistics(series):
    """
    7 features: the mean, standard deviation(std), median, interquartile range (iqr), minimum(min), maximum  (max), and slope
    """
    stats = []
    mean = np.mean(series)
    std_dev= np.std(series)
    min= np.min(series)
    max= np.max(series)
    median= np.median(series)
    slope = get_slope(series)
    iqr_ = iqr(series)
    
    stats.extend([mean, std_dev, max, min, median, slope, iqr_])
    return np.array(stats)

# 6 summary statistics
def calculate_6summary_statistics(series):
    """
    6 features: the mean, standard deviation(std), median, interquartile range (iqr), minimum(min), maximum  (max), and slope
    """
    stats = []
    mean = np.mean(series)
    std_dev= np.std(series)
    min= np.min(series)
    max= np.max(series)
    median= np.median(series)
    slope = get_slope(series)
    #iqr_ = iqr(series)
    
    stats.extend([mean, std_dev, max, min, median, slope ])
    return np.array(stats)


# Catch22
"""
            ["DN_HistogramMode_5", "DN_HistogramMode_10", "CO_f1ecac","CO_FirstMin_ac",
            "CO_HistogramAMI_even_2_5", "CO_trev_1_num", "MD_hrv_classic_pnn40",
            "SB_BinaryStats_mean_longstretch1", "SB_TransitionMatrix_3ac_sumdiagcov",
            "PD_PeriodicityWang_th0_01", "CO_Embed2_Dist_tau_d_expfit_meandiff",
            "IN_AutoMutualInfoStats_40_gaussian_fmmi", "FC_LocalSimple_mean1_tauresrat",
            "DN_OutlierInclude_p_001_mdrmd", "DN_OutlierInclude_n_001_mdrmd",
            "SP_Summaries_welch_rect_area_5_1", "SB_BinaryStats_diff_longstretch0",
            "SB_MotifThree_quantile_hh", "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
            "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
            "SP_Summaries_welch_rect_centroid", "FC_LocalSimple_mean3_stderr"]
"""

def calculate_catch22_features(series):
    return catch22_all(series)['values']

# TSFresh
from tsfresh.feature_extraction import extract_features, EfficientFCParameters

# edit here to include all features 794
def calculate_tsfresh_features(series):
    tsfresh_f = extract_features(series, default_fc_parameters=EfficientFCParameters())
    #mean_abs_change = tsf_calcs.mean_abs_change(series)
    #mean_change = tsf_calcs.mean_change(series),
        # Add more TSFresh features as needed
    return tsfresh_f

# QUANTILES
def f_quantile(X, div=4): # check quantiles if correct
    n = X.shape[-1]

    if n == 1:
        return X
    else:
        num_quantiles = 1 + (n - 1) // div

        if num_quantiles == 1:
            return np.quantile(X, 0.5, axis=-1)[..., np.newaxis]
        else:
            quantiles = np.quantile(X, np.linspace(0, 1, num_quantiles), axis=-1)
            #quantiles_squeezed = np.squeeze(quantiles, axis=2)  # Removes channel dimension if it's 1
            return quantiles.T


# Interval features sets: 

def calculate_if1(X):
    features_list = []
    
    # Loop over each series in the interval
    for series in X:
        # Compute 6 summary statistics for each time series
        stats = calculate_6summary_statistics(series)
        features_list.append(stats)

    quantiles = f_quantile(X)
    return np.hstack([np.array(features_list), quantiles])

def calculate_if2(X):
    features_list = []
    C22 = []
    # Loop over each series in the interval
    for series in X:
        # Compute 6 summary statistics for each time series
        stats = calculate_summary_statistics(series)
        features_list.append(stats)
        c22 = calculate_catch22_features(series)
        C22.append(c22)
    #catch22 = calculate_catch22_features(X)
    return np.hstack([np.array(features_list), np.array(C22)])

def calculate_if3(X):
    quantiles = f_quantile(X)
    catch22 = calculate_catch22_features(X)
    return np.hstack([quantiles,catch22])

def calculate_if4(X):
    stats = calculate_6summary_statistics(X)
    quantiles = f_quantile(X)
    catch22 = calculate_catch22_features(X)
    return np.hstack([stats,quantiles,catch22])

def calculate_if5(X):
    return f_quantile(X)

def calculate_if6(X):
    return calculate_tsfresh_features(X)

##
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
    n_cases,  n_timepoints = arr.shape

    # Reshape into 2D for easier processing, ignoring the channel as it's univariate (1 channel)
    df = pd.DataFrame(arr.reshape(n_cases * n_timepoints, 1), columns=["value"])
    df["case_index"] = np.repeat(np.arange(n_cases), n_timepoints)
    df["time_index"] = np.tile(np.arange(n_timepoints), n_cases)
    df["dimension"] = "dim_0"  # Only one dimension for univariate data

    # Return DataFrame with correct order for tsfresh
    return df[["case_index", "time_index", "dimension", "value"]].rename(
        columns={"case_index": "index"}
    )

def extract_interval_features_func(X, extract_features_func, intervals):
    """
    Extract features for each interval in the time series and return them as a DataFrame.

    Parameters
    ----------
    X : numpy.ndarray
        A 3D numpy array (n_cases, n_channels, n_timepoints) representing the time series.
    extract_features_func : function
        Function to extract features from the time series intervals.
    intervals : list of tuples
        A list of intervals, where each interval is a tuple (start, end).

    Returns
    -------
    interval_features : pandas.DataFrame
        A DataFrame containing the extracted features for each interval.
    """
    n_cases, n_timepoints = X.shape
    interval_features_list = []
    feature_names = []  # For storing the feature names

    for (start, end) in intervals:
        # Select the interval slice for each channel (univariate data)
        X_interval = X[:, start:end]

        # Extract features for this interval
        print(f"the shape of the interval before feature extraction= {X_interval.shape}")
        case_features = extract_features_func(X_interval) # 
        print(f"case features are of shape: {case_features.shape}")
        
        # Add the features for this interval to the list
        interval_features_list.append(np.array(case_features))
        
    # Concatenate all features from all intervals into a single array (axis=0 to stack intervals)
    interval_features = np.hstack(interval_features_list)
    # (162, 0, 34) = (number_features, 0, number of samples)
    
    return interval_features

