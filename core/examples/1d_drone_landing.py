from ..systems import OneDimDrone
from ..controllers import RobustMpcDense, MPCController, OpenLoopController
from ..dynamics import SystemDynamics, LinearSystemDynamics
from ..learning import InverseKalmanFilter

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec
import numpy as np

#%%
print("Starting 1D Drone Landing Simulation..")
#! ===============================================   SET PARAMETERS    ================================================
# Define system parameters of the drone:
mass = 1                                                    # Drone mass (kg)
rotor_rad = 0.08                                            # Rotor radius (m)
drag_coeff = 0.5                                            # Drag coefficient
air_dens = 1.25                                             # Air density (kg/m^3)
area = 0.04                                                 # Drone surface area in xy-plane (m^2)
gravity = 9.81                                              # Gravity (m/s^2)
T_hover = mass*gravity                                      # Hover thrust (N)
ground_altitude = 0.2                                       # Altitude corresponding to drone landed (m)
system = OneDimDrone(mass, rotor_rad, drag_coeff, air_dens, area, gravity, ground_altitude, T_hover)

# Define initial linearized model and ensemble of Bs (linearized around hover):
A = np.array([[0., 1.], [0., 0.]])
B_mean = np.array([[0.],[1/mass]])

# Define simulation parameters:
z_0 = np.array([4., 0.])                                    # Initial position
dt = 1e-2                                                   # Time step length
t_max = 2.                                                  # End time (sec)
t_eval = np.linspace(0, t_max, int(t_max/dt))               # Simulation time points
N_ep = 5                                                   # Number of episodes

# Model predictive controller parameters:
Q = np.array([[1e4, 0.], [0., 1.]])
QN = Q
R = np.array([[1.]])
N_steps = int(t_max/dt)-1
umin = np.array([-T_hover])
umax = np.array([30.-T_hover])
xmin=np.array([ground_altitude, -10.])
xmax=np.array([10., 10.])
ref = np.array([[ground_altitude+0.01 for _ in range(N_steps+1)],
                [0. for _ in range(N_steps+1)]])


#! Filter Parameters:
eta = 0.1**2 # measurement covariance
Nb = 3 # number of ensemble
nk = 5 # number of steps for multi-step prediction
B_ensemble = np.stack([B_mean-np.array([[0.],[0.5]]), B_mean, B_mean+np.array([[0.],[0.5]])],axis=2)

B_ensemble_list = [B_mean-np.array([[0.],[0.5]]), B_mean, B_mean+np.array([[0.],[0.5]])]
#%%
#! ===============================================   RUN EXPERIMENT    ================================================
true_sys = LinearSystemDynamics(A, B_mean)
inverse_kalman_filter = InverseKalmanFilter(A,B_mean, eta, B_ensemble, dt, nk )

x_ep, xd_ep, u_ep, traj_ep, B_ep, mpc_cost_ep, t_ep = [], [], [], [], [], [], []
# B_ensemble [Ns,Nu,Ne] numpy array
B_ep.append(B_ensemble) # B_ep[N_ep] of numpy array [Ns,Nu,Ne]


for ep in range(N_ep):
    print(f"Episode {ep}")
    # Calculate predicted trajectories for each B in the ensemble:
    traj_ep_tmp = []
    for i in range(Nb):
        lin_dyn = LinearSystemDynamics(A, B_ensemble[:,:,i])
        ctrl_tmp = MPCController(lin_dyn, N_steps, dt, umin, umax, xmin, xmax, Q, R, QN, ref)
        ctrl_tmp.eval(z_0, 0)
        traj_ep_tmp.append(ctrl_tmp.parse_result())
    traj_ep.append(traj_ep_tmp)

    # Design robust MPC with current ensemble of Bs and execute experiment:
    lin_dyn = LinearSystemDynamics(A, B_ep[-1][:,:,1])
    controller = MPCController(lin_dyn, N_steps, dt, umin, umax, xmin, xmax, Q, R, QN, ref)  # TODO: Implement Robust MPC
    x_tmp, u_tmp = system.simulate(z_0, controller, t_eval)
    x_ep.append(x_tmp)
    xd_ep.append(np.transpose(ref).tolist())
    u_ep.append(u_tmp)
    t_ep.append(t_eval.tolist())
    mpc_cost_ep.append(np.sum(np.diag((x_tmp[:-1,:].T-ref[:,:-1]).T@Q@(x_tmp[:-1,:].T-ref[:,:-1]) + u_tmp@R@u_tmp.T)))
    if ep == N_ep-1:
        break

    # Update the ensemble of Bs with inverse Kalman filter:
    x_flat, xd_flat, xdot_flat, u_flat, t_flat = inverse_kalman_filter.process(np.array(x_ep), np.array(xd_ep),
                                                                               np.array(u_ep), np.array(t_ep))
    inverse_kalman_filter.fit(x_flat, xdot_flat, u_flat) 
    B_ep.append(inverse_kalman_filter.B_ensemble)

x_ep, xd_ep, u_ep, traj_ep, B_ep, t_ep = np.array(x_ep), np.array(xd_ep), np.array(u_ep), np.array(traj_ep), \
                                         np.array(B_ep), np.array(t_ep)

#%%
#! ===============================================   PLOT RESULTS    =================================================

# Plot evolution of ensembles of B and predicted trajectories for each episode:
f1 = plt.figure(figsize=(12,6))
gs1 = gridspec.GridSpec(2,3, figure=f1)

# - Plot evolution of B ensemble:
n_B = B_ep[0].shape[2]
x_ensemble, y_ensemble = [], []
x_ep_plt, y_min, y_max = [], [], []
for ep in range(N_ep):
    x_ep_plt.append(ep)
    y_min.append(B_ep[ep][1,0,0])
    print(f"min {B_ep[ep][1,0,0]}, max {B_ep[ep][1,0,n_B-1]}")
    y_max.append(B_ep[ep][1,0,n_B-1]) # B_ep[N_ep] of numpy array [Ns,Nu,Ne]
    for ii in range(n_B):
        x_ensemble.append(ep)
        y_ensemble.append(B_ep[ep][1,0,ii])

a0 = f1.add_subplot(gs1[0,:])
a0.scatter(x_ensemble, y_ensemble)
a0.fill_between(x_ep_plt,y_min,y_max, color='b', alpha=0.1)
a0.set_title('Values of Bs in the Ensemble at Each Episode')
a0.set_xlabel('Episode')
a0.set_ylabel('B value')
a0.xaxis.set_major_locator(MaxNLocator(integer=True))
a0.grid()

# - Plot predicted trajectories for 3 selected episodes:
plot_ep = [0, int((N_ep-1)/2), N_ep-1]
a_lst = []
for ii in range(3):
    a_lst.append(f1.add_subplot(gs1[1, ii]))
    a_lst[ii].plot([t_eval[0], t_eval[-1]], [ground_altitude, ground_altitude], '--r', lw=2, label='Ground constraint')
    a_lst[ii].plot(t_eval, traj_ep[plot_ep[ii], 0, 0, :], label='Min B')
    a_lst[ii].plot(t_eval, traj_ep[plot_ep[ii], 1, 0, :], label='Mid B')
    a_lst[ii].plot(t_eval, traj_ep[plot_ep[ii], 2, 0, :], label='Max B')
    a_lst[ii].set_title('Predicted trajectories, ep ' + str(plot_ep[ii]))
    a_lst[ii].set_xlabel('Time (sec)')
    a_lst[ii].set_ylabel('z (m)')
    a_lst[ii].grid()
a_lst[-1].legend(loc='upper right')

gs1.tight_layout(f1)
f1.savefig('core/examples/results/b_ensemble.pdf', format='pdf', dpi=2400)


# Plot MPC cost and executed trajectory every episode:
f2 = plt.figure(figsize=(12,9))
gs2 = gridspec.GridSpec(3,3, figure=f2)

# - Plot evolution of MPC cost:
y_mpc = [c/mpc_cost_ep[0] for c in mpc_cost_ep]
b0 = f2.add_subplot(gs2[0,:])
b0.plot(x_ep_plt, y_mpc)
b0.set_title('MPC Cost Evolution')
b0.set_xlabel('Episode')
b0.set_ylabel('Normalized MPC Cost')
b0.xaxis.set_major_locator(MaxNLocator(integer=True))
b0.grid()

# - Plot executed trajectories and control effort for each episode:
b1_lst, b2_lst = [], []
for ii in range(3):
    b1_lst.append(f2.add_subplot(gs2[1, ii]))
    b1_lst[ii].plot([t_eval[0], t_eval[-1]], [ground_altitude, ground_altitude], '--r', lw=2, label='Ground constraint')
    b1_lst[ii].plot(t_eval, x_ep[plot_ep[ii], :, 0], label='z')
    b1_lst[ii].fill_between(t_eval, ref[0,:], x_ep[plot_ep[ii], :, 0], alpha=0.2)
    b1_lst[ii].plot(t_eval, x_ep[plot_ep[ii], :, 1], label='$\dot{z}$')
    err_norm = (t_eval[-1]-t_eval[0])*np.sum(np.square(x_ep[plot_ep[ii], :, 0].T - ref[0,:]))/x_ep[plot_ep[ii], :, 0].shape[0]
    b1_lst[ii].text(1.2, 0.5, "$\int (z-z_d)^2=${0:.2f}".format(err_norm))

    b1_lst[ii].set_title('Executed trajectory, ep ' + str(plot_ep[ii]))
    b1_lst[ii].set_xlabel('Time (sec)')
    b1_lst[ii].set_ylabel('z, $\dot{z}$ (m, m/s)')
    b1_lst[ii].grid()

    b2_lst.append(f2.add_subplot(gs2[2, ii]))
    b2_lst[ii].plot(t_eval[:-1], u_ep[plot_ep[ii], :, 0], label='T')
    b2_lst[ii].plot([t_eval[0], t_eval[-2]], [umax+T_hover, umax+T_hover], '--r', lw=2, label='Max thrust')
    b2_lst[ii].fill_between(t_eval[:-1], np.zeros_like(u_ep[plot_ep[ii], :, 0]), u_ep[plot_ep[ii], :, 0], alpha=0.2)
    ctrl_norm = (t_eval[-2] - t_eval[0]) * np.sum((np.square(u_ep[plot_ep[ii], :, 0]))/u_ep[plot_ep[ii], :, 0].shape[0])
    b2_lst[ii].text(1.2, 11, "$\int u_n^2=${0:.2f}".format(ctrl_norm))
    b2_lst[ii].set_title('Executed control effort, ep ' + str(plot_ep[ii]))
    b2_lst[ii].set_xlabel('Time (sec)')
    b2_lst[ii].set_ylabel('Thrust (N)')
    b2_lst[ii].grid()
b1_lst[-1].legend(loc='lower right')
b2_lst[-1].legend(loc='upper right')

gs2.tight_layout(f2)
f2.savefig('core/examples/results/executed_traj.pdf', format='pdf', dpi=2400)


