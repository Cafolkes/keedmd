from numpy import array, concatenate, cos, dot, reshape, sin, zeros

from core.dynamics import SystemDynamics


class ClosedSubspaceSys(SystemDynamics):
    def __init__(self, mu, lambd):
        """Dynamics for canoncial systems
        
        Arguments:
            SystemDynamics {dynamical system} -- dynamics
            mu {float} -- System parameter # 1
            lambda {float} -- System parameter # 2

        """
        SystemDynamics.__init__(self, 2, 0)
        self.mu = mu
        self.lambd = lambd

    def eval_dot(self, x, u, t):
        dx1 = self.mu*x[0]
        dx2 = self.lambd*(x[1] - x[0]**2)

        return array([dx1, dx2])