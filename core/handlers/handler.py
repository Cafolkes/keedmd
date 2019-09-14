from numpy import array, zeros, append, delete

class Handler(object):
    """
    Class to handle Episodic Learning. This class is a template only.
    """
    def __init__(self, n, m, Nlift, Nep, w, initial_controller, pert_noise):
        self.n = n
        self.m = m
        self.Nlift = Nlift
        self.Nep = Nep
        self.X_agg = zeros((n,1))
        self.Xd_agg = zeros((n,1))
        self.Z_agg = zeros((Nlift,1))
        self.Zdot_agg = zeros((Nlift,1))
        self.U_agg = zeros((m,1))
        self.Unom_agg = zeros((m,1))
        self.t_agg = zeros((1, 1))
        self.controller_list = []
        self.weights = w
        self.pert_noise = pert_noise
        self.initial_controller = initial_controller


    def run(self):
        pass

    def process(self):
        pass

    def aggregate_data(self, X, Xd, U, Unom, t, edmd_object):
        X, Xd, Z, Zdot, U, Unom, t = edmd_object.process(array(X.transpose()), 
                                                         array(Xd.transpose()), 
                                                         array(U.transpose()), 
                                                         array(Unom.transpose()), 
                                                         array(t.transpose()))
        assert (X.shape[0] == self.X_agg.shape[0])
        assert (Xd.shape[0] == self.Xd_agg.shape[0])
        assert (U.shape[0] == self.U_agg.shape[0])
        assert (Unom.shape[0] == self.Unom_agg.shape[0])

        self.X_agg = append(self.X_agg, X, axis=1)
        self.Xd_agg = append(self.Xd_agg, Xd, axis=1)
        self.Z_agg = append(self.Z_agg, Z, axis=1)
        self.Zdot_agg = append(self.Zdot_agg, Zdot, axis=1)
        self.U_agg = append(self.U_agg, U, axis=1)
        self.Unom_agg = append(self.Unom_agg, Unom, axis=1)
        self.t_agg = append(self.t_agg, t, axis=1)

        if self.X_agg.shape[1] == X.shape[1]+1:
            self.X_agg = delete(self.X_agg, 0, axis=1)
            self.Xd_agg = delete(self.Xd_agg, 0, axis=1)
            self.Z_agg = delete(self.Z_agg, 0, axis=1)
            self.Zdot_agg = delete(self.Zdot_agg, 0, axis=1)
            self.U_agg = delete(self.U_agg, 0, axis=1)
            self.Unom_agg = delete(self.Unom_agg, 0, axis=1)
            self.t_agg = delete(self.t_agg, 0, axis=1)

    def aggregate_ctrl(self, controller):
        self.controller_list.append(controller)

    def get_ctrl(self, q, q_d):
        assert(q.shape[0] == self.n)
        assert(q_d.shape[0] == self.n)

        pass
