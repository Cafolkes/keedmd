#%%
"""Inverted Pendulum Example"""
from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title, ylim, xlabel, ylabel, fill_between, close
import os
from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title, scatter, savefig, hist
from numpy import arange, array, concatenate, cos, identity
from numpy import linspace, ones, sin, tanh, tile, zeros, pi, random, interp, dot, multiply, asarray, zeros_like
import numpy as np
from scipy.io import loadmat, savemat
from ..dynamics import RoboticDynamics, LinearSystemDynamics, AffineDynamics, SystemDynamics
from ..controllers import PDController, OpenLoopController, MPCController, MPCControllerDense
from ..learning import KoopmanEigenfunctions, RBF, Edmd, Keedmd, plot_trajectory, IdentityBF
import time
import dill
import control
import scipy.sparse as sparse

class InvertedPendulum(AffineDynamics, SystemDynamics):
    """Inverted pendulum model.
    States are x = (theta, theta_dot), where theta is the angle of the pendulum
    in rad clockwise from upright and theta_dot is the angular rate of the
    pendulum in rad/s clockwise. The input is u = (tau), where tau is torque in
    N * m, applied clockwise at the base of the pendulum.
    Attributes:
    Mass (kg), m: float
    Gravitational acceleration (m/s^2), g: float
    Length (m), l: float
    """

    def __init__(self, m, g, l):
        """Initialize an InvertedPendulum object.
        Inputs:
        Mass (kg), m: float
        Gravitational acceleration (m/s^2), g: float
        Length (m), l: float
        """

        AffineDynamics.__init__(self)
        SystemDynamics.__init__(self,n=2,m=1)
        self.m, self.g, self.l = m, g, l

    def drift(self, x, t):
        theta, theta_dot = x
        return array([theta_dot, self.g / self.l * sin(theta)])

    def act(self, x, t):
        return array([[0], [1 / (self.m * (self.l ** 2))]])


class InvertedPendulumFp(RoboticDynamics):
    def __init__(self, robotic_dynamics, xf):
        RoboticDynamics.__init__(self, 1, 1)
        self.robotic_dynamics = robotic_dynamics
        self.xf = xf

    def eval(self, q, t):
        if len(q.shape) == 1:
            return q - self.xf
        else:
            return q - self.xf.reshape((self.xf.size,1))

    def drift(self, q, t):
        return self.robotic_dynamics.drift(q, t)

    def act(self, q, t):
        return self.robotic_dynamics.act(q, t)

class InvertedPendulumTrajectory(InvertedPendulum):
    def __init__(self, robotic_dynamics, q_d, t_d):
        RoboticDynamics.__init__(self, 1., 1.)
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

#%%
#! ===============================================   SET PARAMETERS    ===============================================

# Define true system
system_true = InvertedPendulum(m=1., l=1., g=9.81)
n, m = 2, 1                                             # Number of states and actuators
upper_bounds = array([pi/3, 2.0])                       # Upper State constraints
lower_bounds = -upper_bounds                            # Lower State constraints

# Define nominal model and nominal controller:
A_nom = array([[0., 1.], [9.81, 0.]])                   # Linearization of the true system around the origin
B_nom = array([[0.],[1.]])                              # Linearization of the true system around the origin
K_p = -array([[-19.6708]])                              # Proportional control gains
K_d = -array([[-6.3515]])                               # Derivative control gains
nominal_sys = LinearSystemDynamics(A=A_nom, B=B_nom)

# Simulation parameters (data collection)
Ntraj = 20                                              # Number of trajectories to collect data from
dt = 1.0e-2                                             # Time step length
N = int(2./dt)                                          # Number of time steps
t_eval = dt * arange(N + 1)                             # Simulation time points
noise_var = 0.5                                         # Exploration noise to perturb controller

# Koopman eigenfunction parameters
eigenfunction_max_power = 5                             # Max power of variables in eigenfunction products
l2_diffeomorphism = 0.01                                # l2 regularization strength
jacobian_penalty_diffeomorphism = 0.01                  # Estimator jacobian regularization strength
diff_n_epochs = 200                                     # Number of epochs
diff_train_frac = 0.9                                   # Fraction of data to be used for training
diff_n_hidden_layers = 3                                # Number of hidden layers
diff_layer_width = 100                                  # Number of units in each layer
diff_batch_size = 16                                    # Batch size
diff_learn_rate = 0.06842                               # Leaning rate
diff_learn_rate_decay = 0.99                            # Learning rate decay
diff_dropout_prob = 0.25                                # Dropout rate

# KEEDMD parameters
l1_pos_keedmd = 8.195946542380519e-07                   # l1 regularization strength for position states
l1_pos_ratio_keedmd = 1.0                               # l1-l2 ratio for position states
l1_vel_keedmd = 0.008926319071231337                    # l1 regularization strength for velocity states
l1_vel_ratio_keedmd = 1.0                               # l1-l2 ratio for velocity states
l1_eig_keedmd = 0.0010856707261463175                   # l1 regularization strength for eigenfunction states
l1_eig_ratio_keedmd = 0.1                               # l1-l2 ratio for eigenfunction states

# EDMD parameters (benchmark to compare against)
n_lift_edmd = (eigenfunction_max_power+1)**n-1          # Lifting dimension EDMD (same number as for KEEDMD)
l1_edmd = 0.015656845050848606                          # l1 regularization strength
l1_ratio_edmd = 1.0                                     # l1-l2 ratio

# Open loop evaluation parameters
Ntraj_pred = 40                                         # Number of trajectories to use to evaluate open loop performance
noise_var_pred = 0.5                                    # Exploration noise to perturb controller
traj_bounds_pred = [2, 0.5, 0.1, 0.1]                   # State constraints, [x, theta, x_dot, theta_dot]

# Closed loop evaluation parameters
Q_mpc = sparse.diags([1000,1000])                       # MPC state penalty matrix
QN_mpc = Q_mpc                                          # MPC final state penalty matrix
R_mpc = sparse.eye(m)                                   # MPC control penalty matrix
D_mpc = sparse.diags([1,1])                             # MPC state constraint violation penalty matrix
upper_bounds_mpc = array([np.Inf, np.Inf])              # MPC state constraints
lower_bounds_mpc = -upper_bounds_mpc                    # MPC state constraints
umax_mpc = np.Inf                                       # MPC actuation constraint
horizon_mpc = 1.0                                       # MPC time horizon
test_traj = loadmat('core/examples/traj_pendulum.mat')  # Closed loop desired trajectory
t_pred = test_traj['t_plot']
qd_mpc = test_traj['traj_d']
x0_mpc = qd_mpc[:, 0]
t_pred = t_pred.squeeze()

#%%
#! ===============================================    COLLECT DATA     ===============================================
print("Collect data:")

# Simulate system from each initial condition
print(' - Simulate system with {} trajectories using PD controller'.format(Ntraj), end =" ")
t0 = time.process_time()

xf = zeros((2,))
outputs = InvertedPendulumFp(system_true, xf)
pd_controllers = PDController(outputs, K_p, K_d, noise_var)
pd_controllers_nom = PDController(outputs, K_p, K_d, 0.)  # Duplicate of controllers with no noise perturbation
xs, us, us_nom, ts = [], [], [], []
for ii in range(Ntraj):
    x_0 = np.random.rand(2)
    x_0 /= np.linalg.norm(x_0)
    xs_tmp, us_tmp = system_true.simulate(x_0, pd_controllers, t_eval)
    us_nom_tmp = pd_controllers_nom.eval(xs_tmp.transpose(), t_eval).transpose()
    xs.append(xs_tmp)
    us.append(us_tmp)
    us_nom.append(us_nom_tmp[:us_tmp.shape[0],:])
    ts.append(t_eval)

xs, us, us_nom, ts = array(xs), array(us), array(us_nom), array(ts)

print('in {:.2f}s'.format(time.process_time()-t0))
#%%
#!  ===============================================     FIT MODELS      ===============================================

# Construct basis of Koopman eigenfunctions for KEEDMD:
print('Construct Koopman eigenfunction basis:\n', end ="")
t0 = time.process_time()
A_cl = A_nom - dot(B_nom,concatenate((K_p, K_d),axis=1))
BK = dot(B_nom,concatenate((K_p, K_d),axis=1))
eigenfunction_basis = KoopmanEigenfunctions(n=n, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK)
eigenfunction_basis.build_diffeomorphism_model(jacobian_penalty=jacobian_penalty_diffeomorphism, n_hidden_layers = diff_n_hidden_layers, layer_width=diff_layer_width, batch_size= diff_batch_size, dropout_prob=diff_dropout_prob)
eigenfunction_basis.fit_diffeomorphism_model(X=xs, t=ts, X_d=zeros_like(xs), l2=l2_diffeomorphism,
    learning_rate=diff_learn_rate, learning_decay=diff_learn_rate_decay, n_epochs=diff_n_epochs, train_frac=diff_train_frac, batch_size=diff_batch_size)
eigenfunction_basis.construct_basis(ub=upper_bounds, lb=lower_bounds)

print('in {:.2f}s'.format(time.process_time()-t0))


# Fit KEEDMD model:
print(' - Fitting KEEDMD model...', end =" ")
t0 = time.process_time()
keedmd_model = Keedmd(eigenfunction_basis, n, l1_pos=l1_pos_keedmd, l1_ratio_pos=l1_pos_ratio_keedmd, l1_vel=l1_vel_keedmd, l1_ratio_vel=l1_vel_ratio_keedmd, l1_eig=l1_eig_keedmd, l1_ratio_eig=l1_eig_ratio_keedmd, K_p=K_p, K_d=K_d)
X, X_d, Z, Z_dot, U, U_nom, t = keedmd_model.process(xs, zeros_like(xs), us, us_nom, ts)
keedmd_model.fit(X, X_d, Z, Z_dot, U, U_nom)

print('in {:.2f}s'.format(time.process_time()-t0))

# Construct basis of RBFs for EDMD:
print(' - Constructing RBF basis...', end =" ")
t0 = time.process_time()

rbf_centers = multiply(random.rand(n,n_lift_edmd),(upper_bounds-lower_bounds).reshape((upper_bounds.shape[0],1)))+lower_bounds.reshape((upper_bounds.shape[0],1))
rbf_basis = RBF(rbf_centers, n)
rbf_basis.construct_basis()

print('in {:.2f}s'.format(time.process_time()-t0))

# Fit EDMD model
print(' - Fitting EDMD model...', end =" ")
t0 = time.process_time()
edmd_model = Edmd(rbf_basis, n, l1=l1_edmd, l1_ratio=l1_ratio_edmd)
X, X_d, Z, Z_dot, U, U_nom, t = edmd_model.process(xs, zeros_like(xs), us, us_nom, ts)
edmd_model.fit(X, X_d, Z, Z_dot, U, U_nom)

print('in {:.2f}s'.format(time.process_time()-t0))


#%%
#!  ==============================================  EVALUATE PERFORMANCE -- OPEN LOOP =========================================

# Set up trajectory and controller for prediction task:
print('Evaluate Performance with open loop prediction...', end =" ")
t0 = time.process_time()

# Define KEEDMD and EDMD systems:
keedmd_sys = LinearSystemDynamics(A=keedmd_model.A, B=keedmd_model.B)
edmd_sys = LinearSystemDynamics(A=edmd_model.A, B=edmd_model.B)

#Simulate all different systems
xs_pred = []
xs_keedmd = []
xs_edmd = []
xs_nom = []

for ii in range(Ntraj_pred):
    x_0 = multiply(random.rand(2, ), array([pi, 4])) - array([pi / 3, 2.])
    xs_pred_tmp, us_pred_tmp = system_true.simulate(x_0, pd_controllers, t_eval)
    xs_pred_tmp = xs_pred_tmp.transpose()

    # Create systems for each of the learned models and simulate with open loop control signal us_pred:
    keedmd_controller = OpenLoopController(keedmd_sys, us_pred_tmp, t_eval[:us_pred_tmp.shape[0]])
    z0_keedmd = keedmd_model.lift(array([x_0]).T, zeros((2,1))).squeeze()
    zs_keedmd,_ = keedmd_sys.simulate(z0_keedmd,keedmd_controller,t_eval)
    xs_keedmd_tmp = dot(keedmd_model.C,zs_keedmd.transpose())

    edmd_controller = OpenLoopController(edmd_sys, us_pred_tmp, t_eval[:us_pred_tmp.shape[0]])
    z0_edmd = edmd_model.lift(array([x_0]).T, zeros((2,1))).squeeze()
    zs_edmd,_ = edmd_sys.simulate(z0_edmd,edmd_controller,t_eval)
    xs_edmd_tmp = dot(edmd_model.C,zs_edmd.transpose())

    nom_controller = OpenLoopController(nominal_sys, us_pred_tmp, t_eval[:us_pred_tmp.shape[0]])
    xs_nom_tmp,_ = nominal_sys.simulate(x_0,nom_controller,t_eval)
    xs_nom_tmp = xs_nom_tmp.transpose()

    xs_pred.append(xs_pred_tmp)
    xs_keedmd.append(xs_keedmd_tmp)
    xs_edmd.append(xs_edmd_tmp)
    xs_nom.append(xs_nom_tmp)

# Calculate error statistics
mse_keedmd = array([(xs_keedmd[ii] - xs_pred[ii])**2 for ii in range(Ntraj_pred)])
mse_edmd = array([(xs_edmd[ii] - xs_pred[ii])**2 for ii in range(Ntraj_pred)])
mse_nom = array([(xs_nom[ii] - xs_pred[ii])**2 for ii in range(Ntraj_pred)])
e_keedmd = array(np.abs([xs_keedmd[ii] - xs_pred[ii] for ii in range(Ntraj_pred)]))
e_edmd = array(np.abs([xs_edmd[ii] - xs_pred[ii] for ii in range(Ntraj_pred)]))
e_nom = array(np.abs([xs_nom[ii] - xs_pred[ii] for ii in range(Ntraj_pred)]))
mse_keedmd = np.mean(np.mean(np.mean(mse_keedmd)))
mse_edmd = np.mean(np.mean(np.mean(mse_edmd)))
mse_nom = np.mean(np.mean(np.mean(mse_nom)))
e_mean_keedmd = np.mean(e_keedmd, axis=0)
e_mean_edmd = np.mean(e_edmd, axis=0)
e_mean_nom = np.mean(e_nom, axis=0)
e_std_keedmd = np.std(e_keedmd, axis=0)
e_std_edmd = np.std(e_edmd, axis=0)
e_std_nom = np.std(e_nom, axis=0)

Cmatrix = control.ctrb(A=edmd_model.A, B=edmd_model.B)
print('EDMD controllability matrix rank is {}, ns={}, nz={}'.format(np.linalg.matrix_rank(Cmatrix),n,edmd_model.A.shape[0]))
Cmatrix = control.ctrb(A=keedmd_model.A, B=keedmd_model.B)
print('KEEDMD controllability matrix rank is {}, ns={}, nz={}'.format(np.linalg.matrix_rank(Cmatrix),n,edmd_model.A.shape[0]))

#%%
#!==============================================  EVALUATE PERFORMANCE -- CLOSED LOOP =============================================
t0 = time.process_time()
print('Evaluate Performance with closed loop trajectory tracking...', end=" ")

# Define desired trajectory:
output_pred = InvertedPendulumTrajectory(system_true, qd_mpc, t_pred)

# Nominal model MPC:
nominal_controller = MPCControllerDense(linear_dynamics=nominal_sys,
                                               N=int(horizon_mpc / dt),
                                               dt=dt,
                                               umin=array([-umax_mpc]),
                                               umax=array([+umax_mpc]),
                                               xmin=lower_bounds_mpc,
                                               xmax=upper_bounds_mpc,
                                               Q=Q_mpc,
                                               R=R_mpc,
                                               QN=QN_mpc,
                                               xr=qd_mpc,
                                               plotMPC=False,
                                               name='Nominal')

xs_nom_mpc, us_nom_mpc = system_true.simulate(x0_mpc, nominal_controller, t_pred)
xs_nom_mpc = xs_nom_mpc.transpose()
us_nom_mpc = us_nom_mpc.transpose()

# EDMD MPC:
edmd_sys = LinearSystemDynamics(A=edmd_model.A, B=edmd_model.B)
edmd_controller = MPCControllerDense(linear_dynamics=edmd_sys,
                                     N=int(horizon_mpc / dt),
                                     dt=dt,
                                     umin=array([-umax_mpc]),
                                     umax=array([+umax_mpc]),
                                     xmin=lower_bounds_mpc,
                                     xmax=upper_bounds_mpc,
                                     Q=Q_mpc,
                                     R=R_mpc,
                                     QN=QN_mpc,
                                     xr=qd_mpc,
                                     lifting=True,
                                     edmd_object=edmd_model,
                                     plotMPC=False,
                                     soft=True,
                                     D=D_mpc,
                                     name='EDMD')

xs_edmd_mpc, us_emdm_MPC = system_true.simulate(x0_mpc, edmd_controller, t_pred)
xs_edmd_mpc = xs_edmd_mpc.transpose()
us_edmd_mpc = us_emdm_MPC.transpose()

# KEEDMD MPC:
keedmd_sys = LinearSystemDynamics(A=keedmd_model.A, B=keedmd_model.B)
keedmd_controller = MPCControllerDense(linear_dynamics=keedmd_sys,
                                       N=int(horizon_mpc / dt),
                                       dt=dt,
                                       umin=array([-umax_mpc]),
                                       umax=array([+umax_mpc]),
                                       xmin=lower_bounds_mpc,
                                       xmax=upper_bounds_mpc,
                                       Q=Q_mpc,
                                       R=R_mpc,
                                       QN=QN_mpc,
                                       xr=qd_mpc,
                                       lifting=True,
                                       edmd_object=keedmd_model,
                                       plotMPC=False,
                                       soft=True,
                                       D=D_mpc,
                                       name='KEEDMD')

xs_keedmd_mpc, us_keemdm_MPC = system_true.simulate(x0_mpc, keedmd_controller, t_pred)
xs_keedmd_mpc = xs_keedmd_mpc.transpose()
us_keedmd_mpc = us_keemdm_MPC.transpose()

print('in {:.2f}s'.format(time.process_time() - t0))

# Calculate statistics for the different models
mse_mpc_nom = sum(sum((xs_nom_mpc - qd_mpc) ** 2)) / xs_nom_mpc.size
mse_mpc_edmd = sum(sum((xs_edmd_mpc - qd_mpc) ** 2)) / xs_edmd_mpc.size
mse_mpc_keedmd = sum(sum((xs_keedmd_mpc - qd_mpc) ** 2)) / xs_keedmd_mpc.size
E_nom = np.linalg.norm(us_nom_mpc)
E_edmd = np.linalg.norm(us_edmd_mpc)
E_keedmd = np.linalg.norm(us_keedmd_mpc)

Q_d = Q_mpc.todense()
R_d = R_mpc.todense()
cost_nom = sum(np.diag(np.dot(np.dot((xs_nom_mpc - qd_mpc).T, Q_d), xs_nom_mpc - qd_mpc))) + sum(
    np.diag(np.dot(np.dot(us_nom_mpc.T, R_d), us_nom_mpc)))
cost_edmd = sum(np.diag(np.dot(np.dot((xs_edmd_mpc - qd_mpc).T, Q_d), xs_edmd_mpc - qd_mpc))) + sum(
    np.diag(np.dot(np.dot(us_edmd_mpc.T, R_d), us_edmd_mpc)))
cost_keedmd = sum(np.diag(np.dot(np.dot((xs_keedmd_mpc - qd_mpc).T, Q_d), xs_keedmd_mpc - qd_mpc))) + sum(
    np.diag(np.dot(np.dot(us_keedmd_mpc.T, R_d), us_keedmd_mpc)))
print('Tracking error (MSE), Nominal: ', mse_mpc_nom, ', EDMD: ', mse_mpc_edmd, 'KEEDMD: ', mse_mpc_keedmd)
print('Control effort (norm), Nominal:  ', E_nom, ', EDMD: ', E_edmd, ', KEEDMD: ', E_keedmd)
print('MPC cost, Nominal: ', cost_nom, ', EDMD: ', cost_edmd, ', KEEDMD: ', cost_keedmd)
print('MPC cost improvement, EDMD: ', (cost_edmd / cost_nom - 1) * 100, '%, KEEDMD: ',
      (cost_keedmd / cost_nom - 1) * 100, '%')

#%%
#!========================================  PLOT OPEN AND CLOSED LOOP RESULTS =========================================

# Plot errors of different models and statistics, open loop
ylabels = ['$\\theta$', '$\\dot{\\theta}$']
figure(figsize=(6,4))
for ii in range(n):
    subplot(n, 1, ii+1)
    plot(t_eval, np.abs(e_mean_nom[ii,:]), linewidth=2, label='Nominal', color='tab:gray')
    fill_between(t_eval, np.zeros_like(e_mean_nom[ii,:]), e_std_nom[ii,:], alpha=0.2, color='tab:gray')

    plot(t_eval, np.abs(e_mean_edmd[ii,:]), linewidth=2, label='$EDMD$', color='tab:green')
    fill_between(t_eval, np.zeros_like(e_mean_edmd[ii, :]), e_std_edmd[ii, :], alpha=0.2, color='tab:green')

    plot(t_eval, np.abs(e_mean_keedmd[ii,:]), linewidth=2, label='$KEEDMD$',color='tab:orange')
    fill_between(t_eval, np.zeros_like(e_mean_keedmd[ii,:]), e_std_keedmd[ii, :], alpha=0.2,color='tab:orange')

    ylabel(ylabels[ii])
    grid()
    if ii == 0:
        title('Mean absolute open loop prediction error (+ 1 std)')
    if ii == 1 or ii == 3:
        ylim(0., 2.)
    else:
        ylim(0.,.5)
xlabel('Time (s)')
legend(fontsize=10, loc='upper left')
show()

#! Plot the closed loop trajectory:
figure(figsize=(6,6))
for ii in range(n):
    subplot(n+1, 1, ii+1)
    plot(t_pred, qd_mpc[ii,:], linestyle="--",linewidth=2, label='Reference')
    plot(t_pred, xs_nom_mpc[ii, :], linewidth=2, label='Nominal', color='tab:gray')
    plot(t_pred, xs_edmd_mpc[ii,:], linewidth=2, label='EDMD', color='tab:green')
    plot(t_pred, xs_keedmd_mpc[ii, :], linewidth=2, label='KEEDMD',color='tab:orange')
    ylabel(ylabels[ii])
    grid()
    if ii == 0:
        title('Closed loop trajectory tracking performance')
xlabel('Time (s)')
legend(fontsize=10, loc='upper left')
subplot(n + 1, 1, n + 1)
plot(t_pred[:-1], us_nom_mpc[0, :], linewidth=2, label='Nominal', color='tab:gray')
plot(t_pred[:-1], us_edmd_mpc[0, :], linewidth=2, label='EDMD', color='tab:green')
plot(t_pred[:-1], us_keedmd_mpc[0, :], linewidth=2, label='KEEDMD', color='tab:orange')
xlabel('Time (s)')
ylabel('u')
grid()
show()