from juliacall import Main as jl
import juliapkg
import numpy as np

# ## Neccesary installations
# juliapkg.add("DynamicalSystems", "61744808-ddfa-5f27-97ff-6e42cc95d634")
# juliapkg.resolve()
# juliapkg.status(target=None)

jl.seval("using DynamicalSystems")

# ## Using Predefined Julia System ##
# lds = jl.Systems.lorenz()
# lλs = jl.lyapunovspectrum(lds, 1000, Ttr = 0.0003002100350058257)

# ## Using Self Defined Python System ##
# def new_lorenz(u, p, t):
#     σ = p[0]
#     ρ = p[1]
#     β = p[2]
#     
#     du1 = σ * (u[1] - u[0])
#     du2 = u[0] * (ρ - u[2]) - u[1]
#     du3 = u[0] * u[1] - β * u[2]
#     return jl.SVector[3](du1, du2, du3)
# lu0 = [-9.7869288, -15.03852, 20.533978]
# lp = [10, 28, 8/3]
# lds = jl.ContinuousDynamicalSystem(new_lorenz, lu0, lp)
# lλs = jl.lyapunovspectrum(lds, 1000, Ttr = 0.0003002100350058257)

# ## Using Gilpin Defined System ##

import importlib

def julia_find_lyapunov(system_name, timesteps, u0=None, p=None, Ttr=None):
    """
    Given a dynamical system, compute its spectrum of Lyapunov exponents using Julia's DynamicalSystems package
    Args:
        system_name (string): name of Gilpin class for desired dynamical system
        timesteps (int): the length of each trajectory used to calulate Lyapunov
            exponents
        u0 (array): inital condition
        p (array): system parameters in alphabetical order
        Ttr (float): time between timesteps

    Returns:
        λs (array): A list of computed Lyapunov exponents

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
    if Ttr == None:
        Ttr = system.dt

    # Wrapper for Gilpin's rhs to work with Julia's rhs
    def juliafied_gilpin(u, p, t):
        result = system._rhs(*u, t, *p)
        return jl.SVector[len(result)](*result)
    
    # creates a ds class used to calcuate Lyapunov in Julia
    ds = jl.ContinuousDynamicalSystem(juliafied_gilpin, u0, p)
    λs = jl.lyapunovspectrum(ds, timesteps, Ttr = Ttr)
    return λs

# Example Usage
print(julia_find_lyapunov("Lorenz", 1000))

# FAILED CODE
# from dynadojo.wrappers import SystemChecker
# from dynadojo.systems.gilpin_flows import GilpinFlowsSystem
# def gilpin_lorenz(u, p, t):
#     system = SystemChecker(GilpinFlowsSystem(latent_dim=3, embed_dim=3, system_name="Lorenz", seed=1))
#     model = system._system.system
#     out = model.rhs(np.array(u), t=0)
#     # WON'T work because this function is brought over and evaluated in Julia when passed as input to jl.ContinuousDynamicalSystem
#     # i.e. this function can not have any python unique code such as np.array
#     return jl.SVector[3](out[0], out[1], out[2])