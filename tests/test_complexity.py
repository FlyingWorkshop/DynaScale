import unittest
from parameterized import parameterized
from dynadojo.wrappers import SystemChecker
from dynadojo.systems.gilpin_flows import GilpinFlowsSystem
import numpy as np
import EntropyHub as EH
#from dynadojo.utils.julia_in_python import julia_find_lyapunov
from dynadojo.utils.complexity_measures import multi_en, corr_dim, find_lyapunov_exponents, kaplan_yorke_dimension, pesin

ALL_FLOWS = [
    "Lorenz",
    "Rossler"
]

ALL_MAPS = [
    "Henon",
    "Ikeda"
]

ALL_CANONCIAL = {
    "Lorenz": {
        "corr_dim": 2.06,
    },
    "Rossler": {
        "corr_dim": 2.01,
    },
    "Henon": {
        "corr_dim": 1.261
    },"Ikeda": {
        "corr_dim": 1.7
    }
}

flows = ALL_FLOWS  # To test multiple systems, add them to this list
maps = ALL_MAPS

# Creates lyapunov spectrum for Lorenz
system = SystemChecker(GilpinFlowsSystem(latent_dim=3, embed_dim=3, system_name="Lorenz", seed=1))
unwrapped_system = system._system
model = unwrapped_system.system

model.beta = 8/3
model.rho = 28
model.sigma = 10
timesteps = 2500
x0 = np.array([[-9.7869288, -15.03852, 20.533978]])

lspectrum_calc = find_lyapunov_exponents(unwrapped_system, 1000, x0)

# Creates lyapunov spectrum for Rossler
system = SystemChecker(GilpinFlowsSystem(latent_dim=3, embed_dim=3, system_name="Rossler", seed=1))
unwrapped_system = system._system
model = unwrapped_system.system

model.a = 0.2
model.b = 0.2
model.c = 5.7
timesteps = 1000
x0 = np.array([[6.5134412, 0.4772013, 0.34164294]])

rspectrum_calc = find_lyapunov_exponents(unwrapped_system, 1000, x0)

class TestCorrDim(unittest.TestCase):

    @parameterized.expand(flows)
    def test_corr_dim_flows(self, flows):
        data = np.loadtxt(f"tests/{flows}_dim3_seed1_timestep1000_inDist.csv", delimiter=',')
        corr_dim_calc = corr_dim(data)
        corr_dim_canon = ALL_CANONCIAL[flows]["corr_dim"]
        np.testing.assert_allclose(corr_dim_calc, corr_dim_canon, rtol=0.01)

    @parameterized.expand(maps)
    def test_corr_dim_maps(self, maps):
        data = np.loadtxt(f"tests/{maps}_dim2_seed1_timestep1000_inDist.csv", delimiter=',')
        corr_dim_calc = corr_dim(data)
        corr_dim_canon = ALL_CANONCIAL[maps]["corr_dim"]
        np.testing.assert_allclose(corr_dim_calc, corr_dim_canon, rtol=0.01)

class TestMultiEn(unittest.TestCase):

    def test_multi_en(self):
        data = EH.ExampleData('lorenz')
        CI_calc, MSx_calc = multi_en(data, Scales = 5, return_info=True)
        CI_canon = 0.04603960
        Msx_canon = [0,  0.00796833,  0.00926765,  0.01193731,  0.01686631]
        np.testing.assert_allclose(CI_calc, CI_canon, rtol=0.1)
        np.testing.assert_allclose(MSx_calc, Msx_canon, rtol=0.1)
""" These tests will seldom work, since the middle lypaunov exponent of a 3d flow 
in canon will always be exactly zero, and the calculation will be close but never
exactly zero.

class TestLyapunov(unittest.TestCase):

    def test_lorenz(self):
        spectrum_canon = [0.906, 0, -14.572]
#        spectrum_julia = julia_find_lyapunov("Lorenz", timesteps=timesteps, u0=[-9.7869288, -15.03852, 20.533978], p=[8/3, 28, 10])

        #np.testing.assert_allclose(lspectrum_calc, spectrum_julia, rtol=0.05)
        np.testing.assert_allclose(lspectrum_calc, spectrum_canon, rtol=0.05)

    def test_rossler(self):
        spectrum_canon = [0.0714, 0, -5.3943]
#        spectrum_julia = julia_find_lyapunov("Rossler", timesteps=timesteps, u0=[6.5134412, 0.4772013, 0.34164294], p=[0.2, 0.2, 5.7])

        #np.testing.assert_allclose(rspectrum_calc, spectrum_julia, rtol=0.05)
        np.testing.assert_allclose(rspectrum_calc, spectrum_canon, rtol=0.05)
"""
class TestKYDim(unittest.TestCase):

    def test_lorenz(self):
        ky_dim_calc = kaplan_yorke_dimension(lspectrum_calc)
        ky_dim_canon = 2.075158758095728
        np.testing.assert_allclose(ky_dim_calc, ky_dim_canon, rtol=0.01)

    def test_rossler(self):
        ky_dim_calc = kaplan_yorke_dimension(rspectrum_calc)
        ky_dim_canon = 2.0146095059018845
        np.testing.assert_allclose(ky_dim_calc, ky_dim_canon, rtol=0.01)

class TestPesinEn(unittest.TestCase):

    def test_lorenz(self):
        pesin_calc = pesin(lspectrum_calc)
        np.testing.assert_(pesin_calc >= 0)

    def test_rossler(self):
        pesin_calc = pesin(rspectrum_calc)
        np.testing.assert_(pesin_calc >= 0)