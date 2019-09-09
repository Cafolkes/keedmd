from .affine_dynamics import AffineDynamics
from .scalar_dynamics import ScalarDynamics

class AffineResidualDynamics(AffineDynamics):
    def __init__(self, affine_dynamics, drift_res, act_res):
        self.dynamics = affine_dynamics
        self.drift_res = drift_res
        self.act_res = act_res

    def eval(self, x, t):
        return self.dynamics.eval(x, t)

    def drift(self, x, t):
        return self.dynamics.drift(x, t) + self.drift_res(x, t)

    def act(self, x, t):
        return self.dynamics.act(x, t) + self.act_res(x, t)

class ScalarResidualDynamics(AffineResidualDynamics, ScalarDynamics):
    def __init__(self, scalar_dynamics, drift_res, act_res):
        AffineResidualDynamics.__init__(self, scalar_dynamics, drift_res, act_res)