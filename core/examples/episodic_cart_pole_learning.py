"""Cart Pendulum Example"""
from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title, ylim, xlabel, ylabel, \
    fill_between
from os import path
import sys
from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title, scatter, savefig
from numpy import arange, array, concatenate, cos, identity
from numpy import linspace, ones, sin, tanh, tile, zeros, pi, random, interp, dot, multiply, asarray
import numpy as np
# from numpy.random import uniform
from scipy.io import loadmat, savemat
from sys import argv
from core.systems import CartPole
from core.dynamics import LinearSystemDynamics
from core.controllers import PDController, OpenLoopController, MPCController
from core.learning_keedmd import KoopmanEigenfunctions, RBF, Edmd, Keedmd, plot_trajectory
import time

import random as veryrandom
import scipy.sparse as sparse
from core.handlers import SimulationHandler


class CartPoleTrajectory(CartPole):
    def __init__(self, robotic_dynamics, q_d, t_d):
        m_c, m_p, l, g = robotic_dynamics.params
        CartPole.__init__(self, m_c, m_p, l, g)
        self.robotic_dynamics = robotic_dynamics
        self.q_d = q_d
        self.t_d = t_d

    def eval(self, q, t):
        return q - self.desired_state(t)

    def desired_state(self, t):
        return [interp(t, self.t_d.flatten(), self.q_d[ii, :].flatten()) for ii in range(self.q_d.shape[0])]

    def drift(self, q, t):
        return self.robotic_dynamics.drift(q, t)

    def act(self, q, t):
        return self.robotic_dynamics.act(q, t)


# %% ===============================================   SET PARAMETERS    ===============================================

# Define true system
system_true = CartPole(m_c=.5, m_p=.2, l=.4)
n, m = 4, 1  # Number of states and actuators
upper_bounds = array([3.0, pi / 3, 2, 2])  # State constraints
lower_bounds = -upper_bounds  # State constraints


# Define nominal model and nominal controller:
A_nom = array([[0., 0., 1., 0.], [0., 0., 0., 1.], [0., -3.924, 0., 0.],
               [0., 34.335, 0., 0.]])  # Linearization of the true system around the origin
B_nom = array([[0.], [0.], [2.], [-5.]])  # Linearization of the true system around the origin
K_p = -array([[7.3394, 39.0028]])  # Proportional control gains
K_d = -array([[8.0734, 7.4294]])  # Derivative control gains
nominal_sys = LinearSystemDynamics(A=A_nom, B=B_nom)
A_cl = A_nom - dot(B_nom, concatenate((K_p, K_d), axis=1))
BK = dot(B_nom, concatenate((K_p, K_d), axis=1))

# Simulation parameters (data collection)
plot_traj_gen = False  # Plot trajectories generated for data collection
traj_origin = 'load_mat'  # gen_MPC - solve MPC to generate desired trajectories, load_mat - load saved trajectories
Ntraj = 50  # Number of trajectories to collect data from
dt = 1.0e-2  # Time step
N = int(2. / dt)  # Number of time steps
t_eval = dt * arange(N + 1)  # Simulation time points
noise_var = 0.1  # Exploration noise to perturb controller

# Koopman eigenfunction parameters
plot_eigen = False
eigenfunction_max_power = 3
Nlift = (eigenfunction_max_power+1)**n-1 + n
l2_diffeomorphism = 1e0  # Fix for current architecture
jacobian_penalty_diffeomorphism = 5e0  # Fix for current architecture
load_diffeomorphism_model = True
diffeomorphism_model_file = 'diff_model'
diff_n_epochs = 100
diff_train_frac = 0.9
diff_n_hidden_layers = 2
diff_layer_width = 100
diff_batch_size = 16
diff_learn_rate = 1e-3  # Fix for current architecture
diff_learn_rate_decay = 0.95  # Fix for current architecture
diff_dropout_prob = 0.5

# KEEDMD parameters
# Best: 0.024
l1_keedmd = 5e-2
l2_keedmd = 1e-2

# EDMD parameters
# Best 0.06
n_lift_edmd = (eigenfunction_max_power + 1) ** n - 1
l1_edmd = 1e-2
l2_edmd = 1e-2

# Learning loop parameters:
Nep = 5
w = linspace(0,1,Nep)

# Load trajectories
print("Collect data.")
print(" - Generate optimal desired path..", end=" ")
t0 = time.process_time()
R = sparse.eye(m)

t_d = t_eval
Q = sparse.diags([0, 0, 0, 0])
QN = sparse.diags([100000., 100000., 50000., 10000.])
umax = 5
MPC_horizon = 2  # [s]

mpc_controller = MPCController(linear_dynamics=nominal_sys,
                               N=int(MPC_horizon / dt),
                               dt=dt,
                               umin=array([-umax]),
                               umax=array([+umax]),
                               xmin=lower_bounds,
                               xmax=upper_bounds,
                               Q=Q,
                               R=R,
                               QN=QN,
                               x0=zeros(n),
                               xr=zeros(n))

x_0 = array([2., 0.1, 0.05, 0.05])
mpc_controller.eval(x_0, 0)
q_d = mpc_controller.parse_result().transpose()

savemat('./core/examples/cart_pole_d.mat', {'t_d': t_d, 'q_d': q_d})

if plot_traj_gen:
    figure()
    title('Input Trajectories')
    for j in range(n):
        subplot(n, 1, j + 1)
        [plot(t_eval, q_d[ii, :, j], linewidth=2) for ii in range(Ntraj)]
        grid()
    show()

# %% ===========================================    MAIN LEARNING LOOP     ===========================================

output = CartPoleTrajectory(system_true, q_d.transpose(), t_d)  # Trajectory tracking object
initial_controller = PDController(output, K_p, K_d, noise_var)  # Initial controller #TODO: Change to MPC when MPC is functional
eigenfunction_basis = KoopmanEigenfunctions(n=n, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK)
eigenfunction_basis.build_diffeomorphism_model(n_hidden_layers=diff_n_hidden_layers, layer_width=diff_layer_width,
                                               batch_size=diff_batch_size, dropout_prob=diff_dropout_prob)

handler = SimulationHandler(n,m,Nlift,Nep,w,initial_controller,noise_var,system_true)

for ep in range(Nep):
    X, Xd, U, Unom, t = handler.run()
    #Run handler.process() if data from handler.run() is not in desired format
    #TODO: Enable warm start of NN with weights from previous epsiode
    eigenfunction_basis.fit_diffeomorphism_model(X=X, t=t, X_d=Xd, l2=l2_diffeomorphism,
                                                 jacobian_penalty=jacobian_penalty_diffeomorphism,
                                                 learning_rate=diff_learn_rate, learning_decay=diff_learn_rate_decay,
                                                 n_epochs=diff_n_epochs, train_frac=diff_train_frac,
                                                 batch_size=diff_batch_size)
    eigenfunction_basis.construct_basis(ub=upper_bounds, lb=lower_bounds)
    keedmd_ep = Keedmd(eigenfunction_basis,n=n,l1=l1_keedmd,l2=l2_keedmd,episodic=True)
    handler.aggregate_data(X,Xd,U,Unom,t,keedmd_ep)
    keedmd_ep.fit(handler.X_agg, handler.Xd_agg, handler.Z_agg, handler.Zdot_agg, handler.U_agg, handler.Unom_agg)
    mpc_ep = None #TODO: design MPC controller based on keedmd_ep
    handler.aggregate_ctrl(mpc_ep)


# %% ========================================    PLOT AND ANALYZE RESULTS     ========================================