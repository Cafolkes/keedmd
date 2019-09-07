from numpy import empty, append

class Handler():
    def __init__(self, n, m, Nlift, Nep, w, initial_controller, pert_noise):
        self.n = n
        self.m = m
        self.Nlift = Nlift
        self.Nep = Nep
        self.X_agg = empty((1,n))
        self.Xd_agg = empty((1,n))
        self.Z_agg = empty((1,Nlift))
        self.Zdot_agg = empty((1,Nlift))
        self.U_agg = empty((1,m))
        self.Unom_agg = empty((1,m))
        self.t_agg = empty((1, 1))
        self.controller_list = []
        self.weights = w
        self.pert_noise = pert_noise

        self.controller_list.append(initial_controller)

    def run(self):
        pass

    def process(self):
        pass

    def aggregate_data(self, X, Xd, U, Unom, t, edmd_object):
        assert (X.shape[1] == self.X_agg.shape[1])
        assert (Xd.shape[1] == self.Xd_agg.shape[1])
        assert (U.shape[1] == self.U_agg.shape[1])
        assert (Unom.shape[1] == self.Unom_agg.shape[1])

        X, Xd, Z, Zdot, U, Unom, t = edmd_object.process(self, X, Xd, U, Unom, t)

        self.X_agg = append(self.X_agg, X, axis=0)
        self.Xd_agg = append(self.Xd_agg, Xd, axis=0)
        self.Z_agg = append(self.Z_agg, Z, axis=0)
        self.Zdot_agg = append(self.Zdot_agg, Zdot, axis=0)
        self.U_agg = append(self.X_agg, U, axis=0)
        self.Unom_agg = append(self.X_agg, Unom, axis=0)
        self.t_agg = append(self.X_agg, t, axis=0)

    def aggregate_ctrl(self, controller):
        self.controller_list.append(controller)

    def get_ctrl(self, q, q_d):
        assert(q.shape[0] == self.n)
        assert(q_d.shape[0] == self.n)

        pass
