from numpy import zeros, random


from .controller import Controller

class AggregatedMpcController(Controller):

    def __init__(self, dynamics, controller_list, w, noise_var=0.):

        Controller.__init__(self, dynamics)
        self.weights = w
        self.controller_list = controller_list
        self.noise_var = noise_var
        self.Nctrl = len(controller_list)
        self.n = controller_list[0].nx
        self.m = controller_list[0].nu
        self.u_pert_lst = []

    def eval(self, x, t):
        u_seq_agg = zeros((self.m,self.controller_list[0]._osqp_N))
        u_nom = zeros((self.m,))
        for ii in range(self.Nctrl):
            u_nom += self.weights[ii]*self.controller_list[ii].eval(x, t) #TODO: Add u_seq_agg as input to MPC problem and adjust torque bounds based on previous output
            u_seq_agg += self.weights[ii]*self.controller_list[ii].get_control_prediction().transpose()

        u_pert = self.noise_var*random.randn(self.m)
        self.u_pert_lst.append(u_pert)

        return u_nom + u_pert
