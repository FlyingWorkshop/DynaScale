import numpy as np
import fathon
from fathon import fathonUtils as fu

## FROM FATHON ##

def dfa(data, windows=[[10,2000]], detailed = False):
    """
    Detrended Fluctuation Analysis: Given a trajectory and window range(s), compute its Hurst exponent(s)
    Args:
        data (matrix): D x T vector of data points
        windows (list of lists): each element in the format [start,end]
        detailed (bool): returns slope (aka exponent), intercept, window sizes, fluation of each window
    Returns:
        float/list: Hurst exponent(s)
    """
    # zero-mean cumulative sum
    data = fu.toAggregated(data)
    # initialize dfa object
    pydfa = fathon.DFA(data)
    # compute fluctuation function and Hurst exponent
    wins = fu.linRangeByStep(windows[0][0], windows[0][1])
    n, F = pydfa.computeFlucVec(wins, revSeg=True, polOrd=3)
    H, H_intercept = pydfa.fitFlucVec()

    # if multiple windows are provided, returns list of exponent, one for each window
    if len(windows) > 1:
        limits_list = np.array(windows, dtype=int)
        list_H, list_H_intercept = pydfa.multiFitFlucVec(limits_list)
        if detailed == True:
            return list_H, list_H_intercept, n, F
        return list_H
    
    # in the case that there was just one window, returns just that one exponent
    if detailed == True:
        return H, H_intercept, n, F
    return H

def multi_dfa(data, detailed = False):
    """
    MultiFractal Detrended Fluctuation Analysis: Computation of the fluctuations in each window for each q-order
    Args:
        data (matrix): D x T vector of data points
        detailed (bool): returns slope (aka exponent), intercept, window sizes, fluation of each window
    Returns:
        list: Hurst exponent(s)
    """
    # zero-mean cumulative sum
    data = fu.toAggregated(data)
    # initialize mfdfa object
    pymfdfa = fathon.MFDFA(data)
    # compute fluctuation function and generalized Hurst exponents
    wins = fu.linRangeByStep(10, 2000)
    n, F = pymfdfa.computeFlucVec(wins, np.arange(-3, 4, 0.1), revSeg=True, polOrd=1)
    list_H, list_H_intercept = pymfdfa.fitFlucVec()

    if detailed == True:
        return list_H, list_H_intercept, n, F
    return list_H

def mass_exponents(data):
    """
    Computation of the mass exponents.
    Args:
        data (matrix): D x T vector of data points
    Returns:
        list: mass exponents
    """
    # zero-mean cumulative sum
    data = fu.toAggregated(data)
    # initialize mfdfa object
    pymfdfa = fathon.MFDFA(data)
    # compute fluctuation function and generalized Hurst exponents
    wins = fu.linRangeByStep(10, 2000)
    n, F = pymfdfa.computeFlucVec(wins, np.arange(-3, 4, 0.1), revSeg=True, polOrd=1)
    list_H, list_H_intercept = pymfdfa.fitFlucVec()

    tau = pymfdfa.computeMassExponents()
    return tau

def multifractal_spectrum(data, detailed = False):
    """
    Computation of the multifractal spectrum
    Args:
        data (matrix): D x T vector of data points
        detailed (bool): returns multifractal spectrum and singularity strengths
    Returns:
        list: multifractal spectrum
    """
    # zero-mean cumulative sum
    data = fu.toAggregated(data)
    # initialize mfdfa object
    pymfdfa = fathon.MFDFA(data)
    # compute fluctuation function and generalized Hurst exponents
    wins = fu.linRangeByStep(10, 2000)
    n, F = pymfdfa.computeFlucVec(wins, np.arange(-3, 4, 0.1), revSeg=True, polOrd=1)
    list_H, list_H_intercept = pymfdfa.fitFlucVec()
    alpha, mfSpect = pymfdfa.computeMultifractalSpectrum()

    if detailed == True:
        return mfSpect, alpha
    return mfSpect