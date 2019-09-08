from core.handlers import Handler
from core.controllers import AggregatedMpcController
from numpy import array, random

class SimulationHandler(Handler):
    def __init__(self,n,m,Nlift,Nep,w,initial_controller,pert_noise,dynamics, q_d, t_d):
        super().__init__(n,m,Nlift,Nep,w,initial_controller,pert_noise)
        self.dynamical_system = dynamics
        self.q_d = q_d
        self.t_d = t_d

    def run(self):
        controller_agg = AggregatedMpcController(self.dynamical_system, self.controller_list, self.weights, self.pert_noise)
        x0 = self.q_d[:,:1] + 0.05*random.randn(self.q_d.shape[0],1)
        xs, us = self.dynamical_system.simulate(x0.squeeze(), controller_agg, self.t_d)
        us_nom = array(controller_agg.u_pert_lst)

        return xs, self.q_d.transpose(), us, us_nom, self.t_d

    def process(self, X, Xd, U, Upert, t):
        Unom = U-Upert
        return X, Xd, U, Unom, t

    #TODO: What other functions must be overloaded to enable simulation with the aggregated controller?
    def get_ctrl(self, q, q_d):
        pass