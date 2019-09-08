from numpy import empty, append

class Handler():
    """
    Class to handle Episodic Learning. This class is a template only.
    """
    def __init__(self, ns, nu, Nlift, Nep, w, initial_controller, pert_noise):
        """
        Sizes:
        - Nlift: number of lifted dimensions
        - ns: number or original states
        - nu: number of control inputs

        Inputs:
        - ns: number or original states
        - nu: number of control inputs
        - Nlift: number of lifted dimensions
        - Nep: number of episodes
        - w: weight for each controller, numpy 1darray [Nep,]. All w sum 1.
        - initial_controller: controller object
        - pert_noise: added noise to the previous input to excit the system, float

        """

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
        """
        Call this function to generate data. 
        
        Te output should be fed into 'process' to obtain clean data. 
        """
        pass

    def process(self):
        """
        Call this function to clean up your data.
        It should be called after run and before fit
        """
        pass

    def fit(self):
        """
        Call this function to fit a new model using the new data.
        It should be called after process
        """
        pass

    def aggregate_data(self, X, Xd, U, Unom, t, edmd_object):
        """
        Aggregates data for episodic learning

        Inputs:
        - X: array ??
        - Xd: array array 
        - U: input, array [nu,N,Ntraj]
        - Unom:
        - t:
        - edmd_object: 

        """


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
