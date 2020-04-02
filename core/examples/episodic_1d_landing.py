#!/usr/bin/env python3

# Python
from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title, ylim, xlabel, ylabel, \
    fill_between, savefig, close, text, tight_layout
from matplotlib.ticker import MaxNLocator
from datetime import datetime
from numpy import arange, array, concatenate, cos, identity
from numpy import linspace, ones, sin, tanh, tile, zeros, pi, random, interp, dot, zeros_like
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import os
import dill

# KEEDMD
from core.learning import KoopmanEigenfunctions, Keedmd
from core.dynamics import LinearSystemDynamics
from core.handlers import Handler
from core.controllers import MPCControllerDense
from core.systems import OneDimDrone

# %% ===============================================   SET PARAMETERS    ===============================================
# Define system parameters of the drone:
mass = 1                                                    # Drone mass (kg)
rotor_rad = 0.08                                            # Rotor radius (m)
drag_coeff = 0.5                                            # Drag coefficient
air_dens = 1.25                                             # Air density (kg/m^3)
area = 0.04                                                 # Drone surface area in xy-plane (m^2)
gravity = 9.81                                              # Gravity (m/s^2)
T_hover = mass*gravity                                      # Hover thrust (N)
ground_altitude = 0.0                                       # Altitude corresponding to drone landed (m)
system = OneDimDrone(mass, rotor_rad, drag_coeff, air_dens, area, gravity, ground_altitude, T_hover)

# Define initial linearized model and ensemble of Bs (linearized around hover):
A_nom = np.array([[0., 1.], [0., 0.]])
B_nom = np.array([[0.],[1/mass]])
K_p, K_d = [[25.125]], [[10.6331]]
n = B_nom.shape[0]
m = B_nom.shape[1]

# Define simulation parameters:
q_0 = np.array([2., 0.])                                    # Initial position
dt = 1e-2                                                   # Time step length
t_max = 2.0                                                 # End time (sec)
t_eval = np.linspace(0, t_max, int(t_max/dt))               # Simulation time points

# Experiment parameters
n_waypoints = 2
pert_noise = 0.02
Nep = 6
w = linspace(0, 1, Nep)
upper_bounds = array([5., 4.])                              # State constraints
lower_bounds = array([0., -4.])                             # State constraints

# KEEDMD parameters:
# - Koopman eigenfunction parameters:
eigenfunction_max_power = 2                                 # Max power of variables in eigenfunction products
Nlift = (eigenfunction_max_power + 1) ** 2 + n              # Dimension of lifted state model
l2_diffeomorphism = 0.0                                     # l2 regularization strength
jacobian_penalty_diffeomorphism = 1e1                       # Estimator jacobian regularization strength
diff_n_epochs = 50                                          # Number of epochs
diff_train_frac = 0.9                                       # Fraction of data to be used for training
diff_n_hidden_layers = 2                                    # Number of hidden layers
diff_layer_width = 25                                       # Number of units in each layer
diff_batch_size = 8                                         # Batch size
diff_learn_rate = 0.06842                                   # Leaning rate
diff_learn_rate_decay = 0.95                                # Learning rate decay
diff_dropout_prob = 0.25                                    # Dropout rate

# - KEEDMD Parameters:
A_cl = A_nom - np.dot(B_nom, np.concatenate((K_p, K_d),axis=1))
BK = np.dot(B_nom, np.concatenate((K_p, K_d),axis=1))
tune_keedmd = False
l1_pos = 0.0010834166831560485                       # l1 regularization strength for position states
l1_ratio_pos = 1.0                                   # l1-l2 ratio for position states
l1_vel = 0.03518094179245991                         # l1 regularization strength for velocity states
l1_ratio_vel = 1.0                                   # l1-l2 ratio for velocity states
l1_eig = 0.22780288830462658                         # l1 regularization strength for eigenfunction states
l1_ratio_eig = 1.0                                   # l1-l2 ratio for eigenfunction states

# Model predictive controller parameters:
Q = np.array([[1e3, 0.], [0., 1e0]])
QN = Q
R = np.array([[1e0]])
Dsoft = sp.sparse.diags([5e2, 5e1])
MPC_horizon = 1.0  # [s]
N_steps = int(MPC_horizon / dt)
T_max = 25.
umin = np.array([-T_hover])
umax = np.array([T_max-T_hover])
xmin=np.array([ground_altitude, -5.])
xmax=np.array([10., 5.])
q_d = np.array([[ground_altitude+0.05 for _ in range(int(t_max/dt)+1)],
                [0. for _ in range(int(t_max/dt)+1)]])
fixed_point = True

# %% ========================================       SUPPORTING METHODS        ========================================
class DroneHandler(Handler):
    def __init__(self, n, m, Nlift, Nep, w, initial_controller, pert_noise, p_init, p_final, dt, hover_thrust):
        super(DroneHandler, self).__init__(n, m, Nlift, Nep, w, initial_controller, pert_noise)
        self.Tpert = 0.  # Brownian noise
        self.p_init = p_init
        self.p_final = p_final
        self.dt = dt
        self.hover_thrust = hover_thrust
        self.comp_time = []
        self.ctrl_history = []
        self.time_history = []
        self.nom_ctrl_history = []

    def clean_data(self, X, Xd, U, Upert, t, ctrl_hist=None):
        assert (X.shape[0] == self.X_agg.shape[0])
        assert (U.shape[0] == self.U_agg.shape[0])
        assert (Upert.shape[0] == self.Unom_agg.shape[0])

        Unom = U - Upert
        U -= self.hover_thrust  # TODO: Make sure this is consistent with data collected if changing initial controller
        Unom -= self.hover_thrust  # TODO: Make sure this is consistent with data collected if changing initial controller

        # Trim beginning and end of dataset until certain altitude is reached and duration has passed
        #start_altitude = self.p_init[2] - 0.3  # Start altitude in meters
        #max_dur = 2.0  # Max duration in seconds
        #first_ind = np.argwhere(X[0, 1:] < start_altitude)[0][0] + 1  # First data point is just initializing and excluded
        first_ind = 0
        t = t[first_ind:]
        t -= t[0]

        #end_ind = np.argwhere(t[0, :] > max_dur)[0][0]
        end_ind = X.shape[1]

        X = X[:, first_ind:first_ind + end_ind]
        Xd = Xd[:, first_ind:first_ind + end_ind]
        U = U[:, first_ind:first_ind + end_ind]
        Unom = Unom[:, first_ind:first_ind + end_ind]
        t = t[:end_ind]
        if ctrl_hist is not None:
            ctrl_hist = ctrl_hist[first_ind - 1:first_ind + end_ind - 1]

        return X, Xd, U, Unom, t, ctrl_hist

    def aggregate_landings_per_episode(self, X_w, Xd_w, U_w, Unom_w, t_w):
        nw = X_w.__len__()
        t_size_min = np.min([t_local.squeeze().shape[0] for t_local in t_w])
        X = np.zeros((X_w[0].shape[0], t_size_min, nw))
        Xd = np.zeros((Xd_w[0].shape[0], t_size_min, nw))
        U = np.zeros((U_w[0].shape[0], t_size_min-1, nw))
        Unom = np.zeros((Unom_w[0].shape[0], t_size_min-1, nw))
        t = np.zeros((t_size_min, nw))  # np.linspace(0,t_end,t_size_min)
        for i in range(nw):
            X[:, :, i] = X_w[i][:, :t_size_min]
            Xd[:, :, i] = Xd_w[i][:, :t_size_min]
            U[:, :, i] = U_w[i][:, :t_size_min-1]
            Unom[:, :, i] = Unom_w[i][:, :t_size_min-1]
            t[:, i] = t_w[i][:t_size_min]

        return X, Xd, U, Unom, t

    def eval(self, x, t):
        t0 = datetime.now().timestamp()
        x = x - self.p_final  # TODO: Check if this leads to correct behavior
        T_d = self.initial_controller.eval(x, t)
        ctrl_lst = np.zeros(len(self.controller_list))
        for ii in range(len(self.controller_list)):
            # Update control sequence
            ctrl_lst[ii] = self.weights[ii] * self.controller_list[ii].eval(x, 0.)[0]
        T_d += sum(ctrl_lst)
        self.nom_ctrl_history.append(T_d)
        self.Tpert += self.pert_noise * random.randn()
        T_d += self.Tpert
        self.time_history.append(t0)
        self.ctrl_history.append(ctrl_lst)
        self.comp_time.append((datetime.now().timestamp() - t0))
        return T_d

    def reset(self):
        pass

    def process(self, u):
        return u

# %% ===========================================    MAIN LEARNING LOOP     ===========================================

# Initialize robot

initialize_NN = True  # Initializes the weights of the NN when set to true
linearized_dyn = LinearSystemDynamics(A_nom, B_nom)
initial_controller = MPCControllerDense(linearized_dyn, N_steps, dt, umin, umax, xmin, xmax, Q, R, QN, q_d)
eigenfunction_basis = KoopmanEigenfunctions(n=n, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK)
eigenfunction_basis.build_diffeomorphism_model(jacobian_penalty=jacobian_penalty_diffeomorphism,
                                               n_hidden_layers=diff_n_hidden_layers, layer_width=diff_layer_width,
                                               batch_size=diff_batch_size, dropout_prob=diff_dropout_prob)
handler = DroneHandler(n, m, Nlift, Nep, w, initial_controller, pert_noise, q_0, q_d[:,-1], dt, T_hover)

X_ep = []
Xd_ep = []
U_ep = []
Unom_ep = []
t_ep = []
ctrl_history_ep = []

Xval_ep = []
Uval_ep = []
tval_ep = []
track_error = []
ctrl_effort = []

print('Starting episodic learning...')
for ep in range(Nep + 1):
    # Run single landing with no perturbation for validation plots:
    print("Executing trajectory with no perturbation noise...")
    handler.ctrl_history = []
    handler.time_history = []
    handler.nom_ctrl_history = []
    handler.pert_noise = 0.

    xs, us = system.simulate(q_0, handler, t_eval)
    us_nom = np.array(handler.nom_ctrl_history)
    X_val, Xd_val, U_val, _, t_val, ctrl_hist = handler.clean_data(xs.T, q_d, us.T, us_nom.T, t_eval, ctrl_hist=handler.ctrl_history)
    handler.pert_noise = pert_noise
    ctrl_history_ep.append(ctrl_hist)

    # Run training loop:
    X_w = []
    Xd_w = []
    U_w = []
    Unom_w = []
    t_w = []
    if ep == Nep:
        # Only run validation landing to evaluate performance after final episode
        Xval_ep.append(X_val)
        Uval_ep.append(U_val)
        tval_ep.append(t_val)
        track_error.append((t_val[-1] - t_val[0]) * np.sum(((X_val[0, :] - Xd_val[0, :]) ** 2) / X_val.shape[1]))
        ctrl_effort.append((t_val[-1] - t_val[0]) * np.sum(U_val ** 2, axis=1) / U_val.shape[1])
        continue

    for ww in range(n_waypoints):  # Execute multiple trajectories between training
        print("Executing trajectory ", ww + 1, " out of ", n_waypoints, "in episode ", ep)
        print("Executing fast landing with current controller...")
        handler.nom_ctrl_history = []
        xs, us = system.simulate(q_0, handler, t_eval)
        us_nom = np.array(handler.nom_ctrl_history)
        X, Xd, U, Unom, t, ctrl_hist = handler.clean_data(xs.T, q_d, us.T, us_nom.T, t_eval,
                                                                       ctrl_hist=handler.ctrl_history)

        # Must locally aggregate data from each waypoint and feed aggregated matrices to fit_diffeomorphism
        X_w.append(X)
        Xd_w.append(Xd)
        U_w.append(U)
        Unom_w.append(Unom)
        t_w.append(t)

    X, Xd, U, Unom, t = handler.aggregate_landings_per_episode(X_w, Xd_w, U_w, Unom_w, t_w)
    print("Fitting diffeomorphism...")
    eigenfunction_basis.fit_diffeomorphism_model(X=array(X.transpose()), t=t.transpose(), X_d=array(Xd.transpose()),
                                                 l2=l2_diffeomorphism,
                                                 learning_rate=diff_learn_rate, learning_decay=diff_learn_rate_decay,
                                                 n_epochs=diff_n_epochs, train_frac=diff_train_frac,
                                                 batch_size=diff_batch_size, initialize=initialize_NN, verbose=False)
    eigenfunction_basis.construct_basis(ub=upper_bounds, lb=lower_bounds)

    keedmd_ep = Keedmd(eigenfunction_basis, n, l1_pos=l1_pos, l1_ratio_pos=l1_ratio_pos, l1_vel=l1_vel,
                       l1_ratio_vel=l1_ratio_vel,
                       l1_eig=l1_eig, l1_ratio_eig=l1_ratio_eig, K_p=K_p, K_d=K_d, episodic=True)
    handler.aggregate_data(X, Xd, U, Unom, t, keedmd_ep)
    keedmd_ep.fit(handler.X_agg, handler.Xd_agg, handler.Z_agg, handler.Zdot_agg, handler.U_agg, handler.Unom_agg)
    keedmd_sys = LinearSystemDynamics(A=keedmd_ep.A, B=keedmd_ep.B)
    mpc_ep = MPCControllerDense(linear_dynamics=keedmd_sys,
                                N=N_steps,
                                dt=dt,
                                umin=umin,
                                umax=umax,
                                xmin=xmin,
                                xmax=xmax,
                                Q=Q,
                                R=R,
                                QN=QN,
                                xr=q_d,
                                lifting=True,
                                edmd_object=keedmd_ep,
                                #plotMPC=plotMPC,
                                name='KEEDMD',
                                soft=True,
                                D=Dsoft)
    handler.aggregate_ctrl(mpc_ep)
    handler.Tpert = 0.  # Reset Brownian noise
    initialize_NN = False  # Warm s tart NN after first episode

    # Store data for the episode:
    X_ep.append(X)
    Xd_ep.append(Xd)
    U_ep.append(U)
    Unom_ep.append(Unom)
    t_ep.append(t)

    # Plot episode results and calculate statistics
    Xval_ep.append(X_val)
    Uval_ep.append(U_val)
    tval_ep.append(t_val)
    track_error.append((t_val[-1] - t_val[0]) * np.sum(((X_val[0, :] - Xd_val[0, :]) ** 2) / X_val.shape[1]))
    ctrl_effort.append((t_val[-1] - t_val[0]) * np.sum(U_val ** 2, axis=1) / U_val.shape[1])

    handler.comp_time = []

print("Experiments finalized")

# %% ========================================    PLOT AND ANALYZE RESULTS     ========================================
folder = './core/examples/results/episodic_keedmd/'
data_list = [X_ep, Xd_ep, U_ep, Unom_ep, t_ep, Xval_ep, Uval_ep, tval_ep, track_error, ctrl_effort, ctrl_history_ep]
outfile = open('./core/examples/results/episodic_keedmd/episodic_data.pickle', 'wb')
dill.dump(data_list, outfile)
outfile.close()

X_ep_arr, Xd_ep_arr, U_ep_arr, Unom_ep_arr, t_ep_arr, Xval_ep_arr, Uval_ep_arr, tval_ep_arr, track_error_arr, ctrl_effort_arr, ctrl_history_ep_arr = np.array(X_ep), np.array(Xd_ep), np.array(U_ep), np.array(Unom_ep), np.array(t_ep), np.array(Xval_ep), np.array(Uval_ep), np.array(tval_ep), np.array(track_error), np.array(ctrl_effort), np.array(ctrl_history_ep)

# Plot summary of tracking error and control effort VS episode
track_error = array(track_error_arr)
ctrl_effort = array(ctrl_effort_arr)[:,0]

track_error_norm = track_error/ track_error[0]
ctrl_effort_norm = ctrl_effort/ctrl_effort[0]

figure(figsize=(12, 9)).gca()
ax = subplot(3, 1, 1)
title('Tracking error and control effort improvement')
ax.plot(range(len(track_error_norm)), track_error_norm, linewidth=2, label='$\int (z-z_d)^2 dt$ (normalized)')
ax.plot(range(len(ctrl_effort_norm)), ctrl_effort_norm, linewidth=2, label='$\int u^2 dt$ (normalized)')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ylabel('Normalized error/effort')
xlabel('Episode')
grid()
legend()

n_subplots = min(4, Xval_ep_arr.shape[0])
ep_lst = [(jj+1)*int(Nep/4) for jj in range(n_subplots)]
ep_lst[0], ep_lst[-1] = 0, Nep

for ii in range(n_subplots):

    ep_tmp = ep_lst[ii]

    X = Xval_ep_arr[ii,:,:]
    Xd = Xd_ep_arr[ii,0,:,:].T
    U = Uval_ep_arr[ii,:,:]
    t = tval_ep_arr[ii,:]

    bx_z = subplot(3, n_subplots, n_subplots + 1 + ii)
    title('Episode  ' + str(ep_lst[ii]))
    plot(t, X[0, :], linewidth=2, label='$z$')
    fill_between(t, Xd[0, :], X[0, :], alpha=0.2)
    plot(t, Xd[0, :], '--', linewidth=2, label='$z_d$')
    ylim((0., 3.))
    legend(fontsize=10, loc="upper right")
    ylabel('Altitude (m)')
    grid()
    err_norm = (t[-1] - t[0]) * sum((X[0, :] - Xd[0, :]) ** 2) / len(X[0, :])
    text(0.02, 0.3, "$\int (z-z_d)^2=${0:.2f}".format(err_norm))

    bx_u = subplot(3, n_subplots, 2*n_subplots + 1 + ii)
    plot(t[:-1], U[0, :]+T_hover, label='$T$')
    plot([t[0], t[-1]], [T_max, T_max], '--r', label='$T_{max}$')
    fill_between(t[:-1], zeros_like(U[0, :]), U[0, :]+T_hover, alpha=0.2)
    ylabel('Thrust (N)')
    xlabel('Time (sec)')
    legend(fontsize=10, loc="upper right")
    grid()
    ctrl_norm = (t[-1] - t[0]) * sum((U[0, :] + T_hover) ** 2) / len(U[0, :])
    text(0.02, 0.2, "$\int u_n^2=${0:.2f}".format(ctrl_norm))

tight_layout()
savefig(folder + 'episode_summary.pdf', format='pdf', dpi=2400)
