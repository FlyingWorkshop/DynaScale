from juliacall import Main as jl
import importlib
import juliapkg
import numpy as np

from dynadojo.systems.gilpin_flows import GilpinFlowsSystem

# ## Neccesary installations, just need to run once
# juliapkg.add("DynamicalSystems", "61744808-ddfa-5f27-97ff-6e42cc95d634")
# juliapkg.resolve()
# juliapkg.status(target=None)

jl.seval("using DynamicalSystems")

## FROM CHAOSTOOLS.JL ##

# Using Self-Defined Python System
def custom_find_lyapunov(rhs, u0, p, timesteps, Δt, max=False):
    """
    Compute the Lyapunov spectrum of a customized, self-defined system
    Args:
        rhs (string): right hand side equations governing system
        u0 (array): inital condition
        p (array): system parameters in order prescribed by rhs
        timesteps (int): the length of  trajectory used to calulate Lyapunov exponents
        Δt (float): time between timesteps
    Returns:
        λs (array): Lyapunov spectrum
    Example Usage:
    >>> def new_lorenz(u, p, t):
    >>>     σ = p[0]
    >>>     ρ = p[1]
    >>>     β = p[2]
    >>>     du1 = σ * (u[1] - u[0])
    >>>     du2 = u[0] * (ρ - u[2]) - u[1]
    >>>     du3 = u[0] * u[1] - β * u[2]
    >>>     return jl.SVector[3](du1, du2, du3)
    >>> lu0 = [-9.7869288, -15.03852, 20.533978]
    >>> lp = [10, 28, 8/3]
    >>> print(jl_custom_find_lyapunov(new_lorenz, lu0, lp, 1000, 0.0003002100350058257))
    """
    ds = jl.ContinuousDynamicalSystem(rhs, u0, p)
    if max == True:
        max_λ = jl.lyapunov(ds, timesteps, Δt = Δt)
        return max_λ
    return jl.lyapunovspectrum(ds, timesteps, Δt = Δt)

# Using Gilpin Defined System
def find_lyapunov(system_name, timesteps, u0=None, p=None, Δt=None, max=False):
    """
    Given a dynamical system, compute its spectrum of Lyapunov exponents using Julia's DynamicalSystems package
    Args:
        system_name (string): name of Gilpin class for desired dynamical system
        timesteps (int): the length of trajectory used to calulate Lyapunov exponents
        u0 (array): inital condition
        p (array): system parameters in alphabetical order
        Ttr (float): time between timesteps
    Returns:
        λs (array): Lyapunov spectrum
    Note:
        Only works for Gilpin flows which have _rhs defined in flows.py
    """
    # Creates an instance of Gilpin class, from which to pull attributes/methods
    module = importlib.import_module('dysts.flows')
    SystemClass = getattr(module, system_name)
    system = SystemClass()
    
    # Defaults to predefined values if not specified
    if u0 == None:
        u0 = system.ic
    if p == None:
        p = []
        for key in system.params:
            p.append(system.params[key])
    if Δt == None:
        Δt = system.dt

    # Wrapper for Gilpin's rhs to work with Julia's rhs
    def juliafied_gilpin(u, p, t):
        result = system._rhs(*u, t, *p)
        return jl.SVector[len(result)](*result)
    
    # creates a ds class used to calcuate Lyapunov in Julia
    ds = jl.ContinuousDynamicalSystem(juliafied_gilpin, u0, p)

    if max == True:
        max_λ = jl.lyapunov(ds, timesteps, Δt = Δt)
        return max_λ
    
    λs = jl.lyapunovspectrum(ds, timesteps, Δt = Δt)
    return λs

def local_lyapunov(system_name, n, u0=None, p=None, S = 10):
    """
    Given a dynamical system and a number of intial points, compute local exponential growth rate
    Args:
        system_name (string): name of Gilpin class for desired dynamical system
        n (int): number of intial conditions from which to permute
        S (optional): number of permutations
    Returns:
        array: A list of local exponential growth rates
    Note:
        Only works for Gilpin flows which have _rhs defined in flows.py
    """
    if n < 2:
        raise ValueError(f"n must be greater than one")

    # Makes n initial conditions
    system = GilpinFlowsSystem(latent_dim=3, embed_dim=3, system_name=system_name, seed=1)
    x0 = system.make_init_conds(n)

    x0 = jl.SVector[len(x0)](*x0)

    # Creates an instance of Gilpin class, from which to pull attributes/methods
    module = importlib.import_module('dysts.flows')
    SystemClass = getattr(module, system_name)
    system = SystemClass()
    
    # Defaults to predefined values if not specified
    if u0 == None:
        u0 = system.ic
    if p == None:
        p = []
        for key in system.params:
            p.append(system.params[key])

    # Wrapper for Gilpin's rhs to work with Julia's rhs
    def juliafied_gilpin(u, p, t):
        result = system._rhs(*u, t, *p)
        return jl.SVector[len(result)](*result)
    
    # creates a ds class used to calcuate Lyapunov in Julia
    ds = jl.ContinuousDynamicalSystem(juliafied_gilpin, u0, p)

    # calculates local exponential growth rate(s) of perturbations
    return jl.local_growth_rates(ds, x0, S = S)

def naive_lyapunov(x, dt):
    """
    Given a trajectory and the dt used to create it, approximate the maximum Lyapunov exponent
        by estimating the slope of E(k)
    Args:
        x (matrix): D x T vector of data points
        dt (float): time between each data point
    Returns:
        float: Maximum Lyapunov exponent
    note:
        WARNING: kind of broken, a lot of uninformed design choices
    """
    # determines how many k points we want to calculate divergence
    k_min = 1
    k_max = int(np.floor(len(x) * 0.80)) # recommendation: k_max ≤ length(R) - delay - embedding_dim - 1
    ks = list(range(k_min, k_max, 10))
    # creates E(k), where E is the y-axis and k is the x-axis
    E = jl.lyapunov_from_data(jl.StateSpaceSet(x), ks)
    E = [e for e in E]
    k = [k * dt for k in ks]
    # find the slope of the largest linear reigon of E(k)
    λ = jl.slopefit(jl.SVector[len(k)](*k), jl.SVector[len(E)](*E), jl.LargestLinearRegion())
    return λ

## FROM COMPLEXITYMEASURES.JL ##
def perm_en(x, τ = 1, base = 2):
    """
    Given a trajectory, compute its permuation entropy
    Args:
        x (matrix): D x T vector of data points
        base (int): log base for caluclating entropy
        τ (int): time delay
    Returns:
        float: Permuation entropy
    """
    m = len(x[0]) # dimension of data
    x = jl.StateSpaceSet(x) # convert to right data type for julia function, multivariate equivalent
    est = jl.OrdinalPatterns(m = m, τ = τ)
    return jl.information(jl.Shannon(base), est, x)

IMPLEMENTATIONS = ["KozachenkoLeonenko", "Kraskov", "Zhu", "ZhuSingh", "Gao", "Goria", "LeonenkoProzantoSavani"]

def shannon_en(x, implementation = "Kraskov", w = 0, k = 1, base = 2):
    """
    Given a trajectory, compute an estimate for Shannon differential information using nearest neighbors
        calculated using specified implementation method
    Args:
        x (matrix): D x T vector of data points
        implementation (string): method of implementation
        w (int): Theiler window, which determines if temporal neighbors are excluded during neighbor searches
        k (int): which specifies k-th nearest neighbor searches method
        base (int): log base to evaluate Shannon entropy
    Returns:
        float: Shannon differential information of a multi-dimensional trajectory
    """

    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"{implementation} is not a valid implementation")
    
    if implementation == "LeonenkoProzantoSavani":
        k = 2

    if implementation == "KozachenkoLeonenko":
        est = jl.KozachenkoLeonenko(w = w)
    else:
        est = eval(f"jl.{implementation}(jl.Shannon({base}), k = k, w = w)")

    return jl.information(est, jl.StateSpaceSet(x))

## FROM FRACTALDIMENSIONS.JL ##

def generalized_dim(x, molteno = False, q = 1, base = 2, k0 = 10):
    """
    Given a trajectory, compute an estimate for the qth order generalized dimension of x
        by calculating its histogram-based Rényi entropy
    Args:
        x (matrix): D x T vector of data points
        q (int): 0 is  "capacity" or "box-counting" dimension, 1 is the "information" dimension
        base (int): log base to evaluate Rényi entropy
        molteno (bool): If True, return an estimate of the generalized_dim of X using the molteno algorithm
    Returns:
        float: qth order generalized dimension of x
    """
    if molteno == True:
        return jl.molteno_dim(jl.StateSpaceSet(x), k0, q = q, base = base)
    else:
        return jl.generalized_dim(jl.StateSpaceSet(x), q = q, base = base)

METHODS = ["grassberger_proccacia", "boxassisted_correlation", "fixedmass_correlation", "takens_best_estimate"]

def corr_dim(x, method = "grassberger_proccacia"):
    """
    Given a trajectory, compute its correlation dimension
    Args:
        x (matrix): D x T vector of data points
        method (string): method of calculating correlation dimension
    Returns:
        float: Correlation dimension of x
    """
    if method not in METHODS:
        raise ValueError(f"{method} is not a valid method")
    
    x = jl.StateSpaceSet(x)

    if method == "takens_best_estimate":
        std = np.std(x)
        corr_dim = eval(f"jl.{method}_dim(x, {std/4})")
    else:
        corr_dim = eval(f"jl.{method}_dim(x)")
    return corr_dim

def ky_dim(λs):
    """
    Given a Lyapunov spectrum, compute its Kaplan Yorke dimension
    Args:
        λs (list): Lyapunov spectrum
    Returns:
        float: Kaplan Yorke dimension
    """
    return jl.kaplanyorke_dim(jl.SVector[len(λs)](*λs))

# ## TEST DATA ##
# test_data_2d = np.loadtxt("tests/Henon_dim2_seed1_timestep1000_inDist.csv", delimiter=',')
# test_data_3d = np.loadtxt("tests/Lorenz_dim3_seed1_timestep1000_inDist.csv", delimiter=',')
# λs = [0.906, 0, -14.572]
# def new_lorenz(u, p, t):
#     σ = p[0]
#     ρ = p[1]
#     β = p[2]
#     du1 = σ * (u[1] - u[0])
#     du2 = u[0] * (ρ - u[2]) - u[1]
#     du3 = u[0] * u[1] - β * u[2]
#     return jl.SVector[3](du1, du2, du3)
# lu0 = [-9.7869288, -15.03852, 20.533978]
# lp = [10, 28, 8/3]
x = find_lyapunov("Lorenz", 1000)
x = [i for i in x]
print(x)