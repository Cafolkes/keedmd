#%%
"""Cart Pendulum Example"""
from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title, ylim, xlabel, ylabel, fill_between, close
from os import path
import os
import sys
from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title, scatter, savefig, hist
from numpy import arange, array, concatenate, cos, identity
from numpy import linspace, ones, sin, tanh, tile, zeros, pi, random, interp, dot, multiply, asarray
import numpy as np
#from numpy.random import uniform
from scipy.io import loadmat, savemat
from sys import argv
from core.systems import CartPole
import time
import dill
import control
from datetime import datetime
import random as veryrandom
import scipy.sparse as sparse
from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title, ylim, xlabel, ylabel, \
    fill_between
from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title, scatter, savefig, hist
from numpy import arange, array, concatenate, cos, identity, dstack
from numpy import linspace, ones, sin, tanh, tile, zeros, pi, random, interp, dot, multiply, asarray, zeros_like
import numpy as np
from core.dynamics import LinearSystemDynamics
from core.controllers import PDController, OpenLoopController, MPCController, MPCControllerDense
from core.learning import KoopmanEigenfunctions, RBF, Edmd, Keedmd, plot_trajectory, IdentityBF
import time
import dill
from pathlib import Path
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
        return [interp(t, self.t_d.flatten(),self.q_d[ii,:].flatten()) for ii in range(self.q_d.shape[0])]

    def drift(self, q, t):
        return self.robotic_dynamics.drift(q, t)

    def act(self, q, t):
        return self.robotic_dynamics.act(q, t)

# %%
# ! ===============================================   SET PARAMETERS    ===============================================

# Tuning parameters
folder = str(Path().absolute()) + '/experiments/episodic_KEEDMD/fast_drone_landing/'
datafile_lst = [folder + '09132019_222031/episodic_data.pickle', folder + '09132019_231840/episodic_data.pickle'] #Add multiple paths to list if multiple data files

# Diffeomorphism tuning parameters:
tune_diffeomorphism = True
n, m = 4, 1  # Number of states and actuators
n_search = 500
n_folds = 2
diffeomorphism_model_file = 'diff_model'
NN_parameter_file = 'scripts/NN_parameters.pickle'

l2_diffeomorphism = np.linspace(0.,5., 10)
jacobian_penalty_diffeomorphism = np.linspace(0.,5., 10)
diff_n_epochs = [50, 100, 200]
diff_n_hidden_layers = [2, 3, 4]
diff_layer_width = [10, 25, 50, 100, 200]
diff_batch_size = [16, 32, 64]
diff_learn_rate = np.linspace(1e-5, 1e-2, 10)  # Fix for current architecture
diff_learn_rate_decay = [0.8, 0.9, 0.95, 0.975, 0.99, 1.0]
diff_dropout_prob = [0., 0.05, 0.1, 0.25, 0.5]

# KEEDMD tuning parameters
tune_keedmd = True
eigenfunction_max_power = 3
l1_ratio = array([.1, .5, .7, .9, .95, .99, 1])  # Values to test

# EDMD tuning parameters
n_lift_edmd = (eigenfunction_max_power + 1) ** n - 1
l1_edmd = 1e-2
l1_ratio_edmd = 0.5  # 1e-2

# Define true system
system_true = CartPole(m_c=.5, m_p=.2, l=.4)
upper_bounds = array([3.0, pi / 3, 2, 2])  # Upper State constraints
lower_bounds = -upper_bounds  # Lower State constraints

# Define nominal model and nominal controller:
A_nom = array([[0., 0., 1., 0.], [0., 0., 0., 1.], [0., -3.924, 0., 0.],
               [0., 34.335, 0., 0.]])  # Linearization of the true system around the origin
B_nom = array([[0.], [0.], [2.], [-5.]])  # Linearization of the true system around the origin
K_p = -array([[7.3394, 39.0028]])  # Proportional control gains
K_d = -array([[8.0734, 7.4294]])  # Derivative control gains
K = concatenate((K_p, K_d),axis=1)
BK = dot(B_nom, K)
A_cl = A_nom + BK

nominal_sys = LinearSystemDynamics(A=A_nom, B=B_nom)

# Simulation parameters (data collection)
plot_traj_gen = False  # Plot trajectories generated for data collection
traj_origin = 'load_mat'  # gen_MPC - solve MPC to generate desired trajectories, load_mat - load saved trajectories
Ntraj = 20  # Number of trajectories to collect data from
dt = 1.0e-2  # Time step
N = int(2. / dt)  # Number of time steps
t_eval = dt * arange(N + 1)  # Simulation time points
noise_var = 0.1  # Exploration noise to perturb controller

# %%
# ! ===============================================    COLLECT DATA     ===============================================
# * Load trajectories
print("Collect data.")
print(" - Generate optimal desired path..", end=" ")
t0 = time.process_time()

R = sparse.eye(m)
t_d = t_eval
traj_bounds = [2, 0.25, 0.05, 0.05]  # x, theta, x_dot, theta_dot
q_d = zeros((Ntraj, N + 1, n))
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
                           xr=zeros(n))
for ii in range(Ntraj):
    x_0 = asarray([veryrandom.uniform(-i, i) for i in traj_bounds])
    mpc_controller.eval(x_0, 0)
    q_d[ii, :, :] = mpc_controller.parse_result().transpose()

outputs = [CartPoleTrajectory(system_true, q_d[i,:,:].transpose(), t_d) for i in range(Ntraj)]
pd_controllers = [PDController(outputs[i], K_p, K_d, noise_var) for i in range(Ntraj)]
pd_controllers_nom = [PDController(outputs[i], K_p, K_d, 0.) for i in range(Ntraj)]  # Duplicate of controllers with no noise perturbation
xs, us, us_nom, ts = [], [], [], []
for ii in range(Ntraj):
    x_0 = q_d[ii,0,:]
    xs_tmp, us_tmp = system_true.simulate(x_0, pd_controllers[ii], t_eval)
    us_nom_tmp = pd_controllers_nom[ii].eval(xs_tmp.transpose(), t_eval).transpose()
    xs.append(xs_tmp)
    us.append(us_tmp)
    us_nom.append(us_nom_tmp[:us_tmp.shape[0],:])
    ts.append(t_eval)

xs, us, us_nom, ts = array(xs), array(us), array(us_nom), array(ts)
#es = xs - q_d  # Tracking error

# %%
# !  ======================================     TUNE DIFFEOMORPHISM MODEL      ========================================
t0 = time.process_time()

cv_inds = np.arange(start=0, stop=xs.shape[0])
np.random.shuffle(cv_inds)
val_num = int(np.floor(xs.shape[0]/n_folds))

if tune_diffeomorphism:
    test_score = []
    best_score = np.inf
    for ii in range(n_search):

        # Sample parameters
        l2 = np.random.choice(l2_diffeomorphism)
        jac_pen = np.random.choice(jacobian_penalty_diffeomorphism)
        n_epochs = np.random.choice(diff_n_epochs)
        n_hidden = np.random.choice(diff_n_hidden_layers)
        layer_width = np.random.choice(diff_layer_width)
        batch_size = np.random.choice(diff_batch_size)
        learn_rate = np.random.choice(diff_learn_rate)
        rate_decay = np.random.choice(diff_learn_rate_decay)
        dropout = np.random.choice(diff_dropout_prob)

        fold_score = []
        for ff in range(n_folds):
            # Define data matrices:
            val_inds = cv_inds[ff*val_num:(ff+1)*val_num]
            train_inds = np.delete(cv_inds,np.linspace(ff*val_num,(ff+1)*val_num-1,val_num, dtype=int))
            t = ts[train_inds,:]
            X = xs[train_inds,:,:]
            Xd = q_d[train_inds,:,:]
            t_val = ts[val_inds, :]
            X_val = xs[val_inds, :, :]
            Xd_val = q_d[val_inds, :, :]

            # Fit model with current data set and hyperparameters
            eigenfunction_basis = KoopmanEigenfunctions(n=n, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK)
            eigenfunction_basis.build_diffeomorphism_model(jacobian_penalty=jac_pen,n_hidden_layers=n_hidden,
                                                            layer_width=layer_width,
                                                            batch_size=batch_size,
                                                            dropout_prob=dropout)

            score_tmp = eigenfunction_basis.fit_diffeomorphism_model(X=X, t=t, X_d=Xd, l2=l2,
                                                            learning_rate=learn_rate,
                                                            learning_decay=rate_decay, n_epochs=n_epochs,
                                                            train_frac=1.0, batch_size=batch_size, initialize=True,
                                                            verbose=False, X_val=X_val, t_val=t_val, Xd_val=Xd_val)
            fold_score.append(score_tmp)

        test_score.append(sum(fold_score)/len(fold_score))
        if test_score[-1] < best_score:
            best_score = test_score[-1]
            eigenfunction_basis.save_diffeomorphism_model(diffeomorphism_model_file) #Only save model if it is improving
            l2_b, jac_pen_b, n_epochs_b, n_hidden_b, layer_width_b, batch_size_b, learn_rate_b, rate_decay_b, dropout_b, test_score_b\
                = l2, jac_pen, n_epochs, n_hidden, layer_width, batch_size, learn_rate, rate_decay, dropout, test_score
            savemat('core/examples/cart_pole_best_params.mat',
                    {'l2': l2_b, 'jac_pen':jac_pen_b, 'n_epochs': n_epochs_b, 'n_hidden': n_hidden_b, 'layer_width': layer_width_b, 'batch_size': batch_size_b, 'learn_rate': learn_rate_b, 'rate_decay': rate_decay_b, 'dropout': dropout_b, 'test_score': test_score_b})

        print('Experiment ', ii, ' test loss with current configuration: ', format(test_score[-1], '08f'), 'best score: ', format(best_score, '08f'))
        print('Best parameters: ', l2_b, jac_pen_b, n_epochs_b, n_hidden_b, layer_width_b, batch_size_b, learn_rate_b, rate_decay_b, dropout_b)

# Load best/stored diffeomorphism model and construct basis:
#eigenfunction_basis.load_diffeomorphism_model(diffeomorphism_model_file)
eigenfunction_basis.construct_basis(ub=upper_bounds, lb=lower_bounds)

print('in {:.2f}s'.format(time.process_time() - t0))
# %%
# !  =========================================     TUNE KEEDMD MODEL      ===========================================