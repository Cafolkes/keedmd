from core.handlers import Handler

class SimulationHandler(Handler):
    def __init__(self,n,m,Nlift,Nep,w,initial_controller,pert_noise,dynamics):
        super().__init__(n,m,Nlift,Nep,w,initial_controller,pert_noise)
        self.dynamical_system = dynamics

    def run(self):

        #self.dynamical_system.simulate(x_0, pd_controllers, t_eval)
        pass

    def process(self, X, Xd, U, Unom, t):
        pass

    #TODO: What other functions must be overloaded to enable simulation with the aggregated controller?

    def get_ctrl(self, q, q_d):
        pass