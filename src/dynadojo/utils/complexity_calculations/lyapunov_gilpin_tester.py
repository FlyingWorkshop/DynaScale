from dynadojo.wrappers import SystemChecker
from dynadojo.systems.gilpin_flows import GilpinFlowsSystem
from dynadojo.utils.complexity_measures import find_lyapunov_exponents
import numpy as np

system = SystemChecker(GilpinFlowsSystem(latent_dim=3, embed_dim=3, system_name="Lorenz", seed=1))
unwrapped_system = system._system
model = unwrapped_system.system

model.beta = 8/3
model.rho = 28
model.sigma = 10
timesteps = 1000
x0 = np.array([[-9.7869288, -15.03852, 20.533978]])

xtpts, x = unwrapped_system.make_data(x0, timesteps=timesteps, return_times=True)
spectrum = find_lyapunov_exponents(x[0], xtpts, timesteps, model)

print("Lorenz:", spectrum)

"""

"parameters": {
            "beta": 2.667,
            "rho": 28,
            "sigma": 10
        }
"initial_conditions": [
            -9.7869288,
            -15.03852,
            20.533978
        ],
"dt": 0.0003002100350058257,

class Lorenz(DynSys):
@staticjit
def _rhs(x, y, z, t, beta, rho, sigma):
    xdot = sigma * y - sigma * x
    ydot = rho * x - x * z - y
    zdot = x * y - beta * z
    return xdot, ydot, zdot
@staticjit
def _jac(x, y, z, t, beta, rho, sigma):
    row1 = [-sigma, sigma, 0]
    row2 = [rho - z, -1, -x]
    row3 = [y, x, -beta]
    return row1, row2, row3

"""

system = SystemChecker(GilpinFlowsSystem(latent_dim=3, embed_dim=3, system_name="Rossler", seed=1))
unwrapped_system = system._system
model = unwrapped_system.system

model.beta = 8/3
model.rho = 28
model.sigma = 10
timesteps = 1000
x0 = np.array([[6.5134412, 6.5134412, 0.34164294]])

xtpts, x = unwrapped_system.make_data(x0, timesteps=timesteps, return_times=True)
spectrum = find_lyapunov_exponents(x[0], xtpts, timesteps, model)

print("Rossler:", spectrum)

"""

"parameters": {
            "a": 0.2,
            "b": 0.2,
            "c": 5.7
        },
"initial_conditions": [
            6.5134412,
            6.5134412,
            0.34164294
        ],
"dt": 0.001181916986164424,

class Rossler(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c):
        xdot = -y - z
        ydot = x + a * y
        zdot = b + z * x - c * z
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, b, c):
        row1 = [0, -1, -1]
        row2 = [1, a, 0]
        row3 = [z, 0, x - c]
        return row1, row2, row3

"""