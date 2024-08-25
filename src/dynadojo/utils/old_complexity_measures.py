import neurokit2
### Multiscale Entropy (Outdated)
#def mse_mv(traj, return_info=False, gilpin=False):
#    """
#    Generate an estimate of the multivariate multiscale entropy. The current version 
#    computes the entropy separately for each channel and then averages. It therefore 
#    represents an upper-bound on the true multivariate multiscale entropy
#
#    Args:
#        traj (ndarray): a trajectory of shape (n_timesteps, n_channels)
#
#    Returns:
#        mmse (float): the multivariate multiscale entropy
#
#    TODO:
#        Implement algorithm from Ahmed and Mandic PRE 2011
#    """
#    mmse_opts = {"composite": True, "fuzzy": True}
#
#    # For univariate data, just calculates once
#    if len(traj.shape) == 1:
#        mmse, info = neurokit2.entropy_multiscale(traj, dimension=2, **mmse_opts)
#        return mmse, info
#    
#    # traj has shape T by D
#    traj = standardize_ts(traj) 
#    all_mse = list()
#    all_info = []
#
#    # Now D by T, where sol_coord is one dimension across all timepoints
#    for sol_coord in traj.T:
#        all_mse.append(
#            neurokit2.entropy_multiscale(sol_coord, dimension=2, **mmse_opts)[0]
#        )
#        all_info.append(
#            neurokit2.entropy_multiscale(sol_coord, dimension=2, **mmse_opts)[1]["Value"]
#        )
#        
#    if return_info == True:
#        # Additionally returns a dataframe containing all SampEn values across all dimensions and coarse grainings
#        return np.sum(all_mse), pd.DataFrame(all_info)
#    
#    if gilpin == True:
#        # If we want to return Gilpin's original version
#        return np.median(all_mse)
#    
#    return np.sum(all_mse)

### Correlation dimension (Outdated)
#def gp_dim(data, y_data=None, rvals=None, nmax=100):
#    """
#    Estimate the Grassberger-Procaccia dimension for a numpy array using the 
#    empirical correlation integral.
#
#    Args:
#        data (np.array): T x D, where T is the number of datapoints/timepoints, and D
#            is the number of features/dimensions
#        y_data (np.array, Optional): A second dataset of shape T2 x D, for 
#            computing cross-correlation.
#        rvals (np.array): A list of radii
#        nmax (int): The number of points at which to evaluate the correlation integral
#
#    Returns:
#        rvals (np.array): The discrete bins at which the correlation integral is 
#            estimated
#        corr_sum (np.array): The estimates of the correlation integral at each bin
#
#    """
#    data = np.asarray(data)
#
#    # Makes a copy of original data for self correlation
#    if y_data is None:
#        y_data = data.copy()
#
#    if rvals is None:
#        std = np.std(data)
#        rvals = np.logspace(np.log10(0.1 * std), np.log10(0.5 * std), nmax)
#
#    n = len(data)
#    
#    dists = cdist(data, y_data)
#    rvals = dists.ravel()
#
#    ## Truncate the distance distribution to the linear scaling range
#    std = np.std(data)
#    rvals = rvals[rvals > 0]
#    rvals = rvals[rvals > np.percentile(rvals, 5)]
#    rvals = rvals[rvals < np.percentile(rvals, 50)]
#    
#    return estimate_powerlaw(rvals)