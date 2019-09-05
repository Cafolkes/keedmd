from .controller import Controller
from numpy import array, interp
class OpenLoopController(Controller):
    """Class for open loop action policies."""

    def __init__(self, dynamics, u_open_loop, t_open_loop):
        """Create a OpenLoopController object.

        Inputs:
        Dynamics, dynamics: Dynamics
        Control action sequence, u_open_loop: numpy array
        Time sequence: numpy array
        """

        Controller.__init__(self, dynamics)
        self.u_open_loop = u_open_loop
        self.t_open_loop = t_open_loop
        self.m = u_open_loop.shape[1]

    def eval(self, x, t):
        return array([interp(t, self.t_open_loop.flatten(), self.u_open_loop[:,ii].flatten()) for ii in range(self.m)]).squeeze()