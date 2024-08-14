from dysts.base import BaseDyn, staticjit

class DynSys:
    def __init__(self, alpha):
        self.alpha = alpha

    def rhs(self, X, t):
        """The right hand side of a dynamical equation"""
        out = self._rhs(*X.T, t, alpha=self.alpha)
        return out
    def jac(self, X, t):
        """The Jacobian of the dynamical system"""
        out = self._jac(*X.T, t, alpha=self.alpha)
        return out

class LDS(DynSys):
    @staticjit
    def _rhs(x, t, alpha):
        xdot = alpha * x
        return xdot
    @staticjit
    def _jac(x, t, alpha):
        row1 = [alpha]
        return row1