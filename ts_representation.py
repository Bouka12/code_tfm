# ts_representation.py

import numpy as np
from statsmodels.regression.linear_model import burg
import pyfftw
from scipy.signal import periodogram

def tsrep0(X):
    """ 
    TSRep0: Base Time Series
    """
    return X

def first_diff(X):
    """
    Compute the first difference of the input array X.

    Parameters
    ----------
    X : numpy.ndarray
        The input array for which the first difference will be computed.

    Returns
    -------
    first_diff : numpy.ndarray
        The first difference of the input array.
    """
    first_diff = np.diff(X, axis=-1)  # Use np.diff along the last axis (timepoints)
    return first_diff



def second_diff(X):
    second_diff = X.diff().diff().dropna() 
    return second_diff


def getPeriodogramRepr(X):
    nfeats = X.shape[1]
    fft_object = pyfftw.builders.fft(X)
    per_X = np.abs(fft_object())
    return per_X[:,:int(nfeats/2)]


def ar_coefs(X):
    X_transform = []
    lags = int(12*(X.shape[1]/100.)**(1/4.))
    for i in range(X.shape[0]):
        coefs,_ = burg(X[i,:],order=lags)
        X_transform.append(coefs)
    return np.array(X_transform)