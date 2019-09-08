
# Python
from numpy import zeros, random

# ROS
import rospy

# Project
from main_controller_force import Robot
from dynamics.goto_optitrack import MavrosGOTOWaypoint
from dynamics.goto_land import land

from core.handlers import Handler

class DroneHandler(Handler):
    """
    Class to handle Episodic KEEMD using the Bintel Drone.
    """
    def __init__(self,ns,nu,Nlift,Nep,w):
        super().__init__(ns,nu,Nlift,Nep,w)

        
        #* Drone Parameters
        self.duration_low = 1.
        self.n_waypoints = 1
        self.controller_rate = 80

        #* Experiment Parameters
        self.p_init = np.array([0., 0., 1.5])
        self.p_final = np.array([0., 0., 0.])
        self.duration = 2

        #* Initialize robot
        self.bintel = Robot(self.controller_rate)
        self.go_waypoint = MavrosGOTOWaypoint()

        print("Starting the experiments..")
        for episode in range(Nep):
            self.episode = episode
            print("- Episode {}, collecting data:".format(episode))
            data = self.run()
            print("Fitting Data")
            self.fit()
            self.aggregate_ctrl(mpc_ep)

        
        print("Experiments finilized.")



    def run(self):
        bintel.gotopoint(self.p_init, self.p_final, self.duration)
        land()
        self.Xraw = bintel.X_agg
        self.process(self)

    def process(self):
        """
        Filter Xraw to X
        """
        self.X = self.Xraw

    def fit(self):
        # Fit KEEDM Model using the new data        
        eigenfunction_basis.fit_diffeomorphism_model(X=X, t=t, X_d=Xd, l2=l2_diffeomorphism,
                                                 jacobian_penalty=jacobian_penalty_diffeomorphism,
                                                 learning_rate=diff_learn_rate, learning_decay=diff_learn_rate_decay,
                                                 n_epochs=diff_n_epochs, train_frac=diff_train_frac,
                                                 batch_size=diff_batch_size)
        eigenfunction_basis.construct_basis(ub=upper_bounds, lb=lower_bounds)
        self.keedmd_model[self.episode] = Keedmd(eigenfunction_basis,n=n,l1=l1_keedmd,l2=l2_keedmd,episodic=True)


    def aggregate_ctrl(self):
        """
        Create a new controller with the new data
        """
        keedmd_sys[self.episode] = LinearSystemDynamics(A=keedmd_model.A, B=keedmd_model.B)
        keedmd_controller[self.episode] = MPCController(linear_dynamics=keedmd_sys[self.episode], 
                                                        N=int(MPC_horizon/dt),
                                                        dt=dt, 
                                                        umin=array([-umax]), 
                                                        umax=array([+umax]),
                                                        xmin=lower_bounds, 
                                                        xmax=upper_bounds, 
                                                        Q=Q, 
                                                        R=R, 
                                                        QN=QN, 
                                                        x0=zeros(n), 
                                                        xr=q_d_pred,
                                                        lifting=True,
                                                        edmd_object=self.keedmd_model[self.episode])


        self.aggregate_data(X,Xd,U,Unom,t,keedmd_ep)
        keedmd_ep.fit(handler.X_agg, handler.Xd_agg, handler.Z_agg, handler.Zdot_agg, handler.U_agg, handler.Unom_agg)

        
    def get_ctrl(self, q, q_d):
        assert(q.shape[0] == self.ns)
        assert(q_d.shape[0] == self.ns)
        u_nom = zeros((self.m,q_d.shape[0]))
        for ii in range(len(self.controller_list)):
            u_nom += self.w[ii]*self.controller_list[ii](q, q_d, u_nom)

        u = u_nom + self.pert_noise*random.randn((u_nom.shape))

        return u, u_nom


if __name__ == '__main__':
    try:
        ns = 2
        nu = 1 
        Nlift = 10
        Nep = 10 
        w = 1
        tester = DroneHandler()
    except rospy.ROSInterruptException:
        pass