#%%
"""Cart Pendulum Example"""
from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title, ylim, xlabel, ylabel, fill_between, close
import os
import sys
from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title, scatter, savefig, hist
from numpy import arange, array, concatenate, cos, identity, zeros_like
from numpy import linspace, ones, sin, tanh, tile, zeros, pi, random, interp, dot, multiply, asarray
import numpy as np
#from numpy.random import uniform
from scipy.io import loadmat, savemat
from sys import argv
from ..systems import CartPole
from ..dynamics import LinearSystemDynamics
from ..controllers import Controller, PDController, OpenLoopController, MPCController, MPCControllerDense
from ..learning import KoopmanEigenfunctions, RBF, Edmd, Keedmd, plot_trajectory, IdentityBF
import time
import dill
import control
from datetime import datetime

import random as veryrandom
import scipy.sparse as sparse
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

class CompositeController(Controller):
    def __init__(self, controller_1, controller_2, C):
        self.controller_1 = controller_1
        self.controller_2 = controller_2
        self.C = C

    def eval(self, x, t):
        u_1 = self.controller_1.eval(x, t)
        u_2 = self.controller_2.eval(dot(self.C, zeros_like(x)), t)

        return array([u_1.item(), u_2.item()])




#%% 
#! ===============================================   SET PARAMETERS    ===============================================

# Define true system
system_true = CartPole(m_c=.5, m_p=.2, l=.4)
n, m = 4, 1  # Number of states and actuators
upper_bounds = array([3.0, pi/3, 2, 2])  # Upper State constraints
lower_bounds = -upper_bounds  # Lower State constraints

# Define nominal model and nominal controller:
A_nom = array([[0., 0., 1., 0.], [0., 0., 0., 1.], [0., -3.924, 0., 0.], [0., 34.335, 0., 0.]])  # Linearization of the true system around the origin
B_nom = array([[0.],[0.],[2.],[-5.]])  # Linearization of the true system around the origin
K_p = -array([[7.3394, 39.0028]])  # Proportional control gains
K_d = -array([[8.0734, 7.4294]])  # Derivative control gains
nominal_sys = LinearSystemDynamics(A=A_nom, B=B_nom)

# Simulation parameters (data collection)
plot_traj_gen = False                # Plot trajectories generated for data collection
traj_origin = 'gen_MPC'              # gen_MPC - solve MPC to generate desired trajectories, load_mat - load saved trajectories
Ntraj = 40                          # Number of trajectories to collect data from

dt = 1.0e-2                         # Time step
N = int(2./dt)                      # Number of time steps
t_eval = dt * arange(N + 1)         # Simulation time points
noise_var = 0.5                     # Exploration noise to perturb controller

# Koopman eigenfunction parameters
plot_eigen = True
eigenfunction_max_power = 2
l2_diffeomorphism = 0.0  #0.26316                 #Fix for current architecture
jacobian_penalty_diffeomorphism = 0.0 #0.55 #4.47368 #3.95   #Fix for current architecture
load_diffeomorphism_model = False
diffeomorphism_model_file = 'diff_model'
diff_n_epochs = 200  # TODO: set back to 500
diff_train_frac = 0.8
diff_n_hidden_layers = 2
diff_layer_width = 50
diff_batch_size = 16
diff_learn_rate = 0.00112#0.0737                  #Fix for current architecture
diff_learn_rate_decay = 0.975            #Fix for current architecture
diff_dropout_prob = 0.25

# KEEDMD parameters
l1_pos_keedmd = 0.0012773
l1_pos_ratio_keedmd = 0.1
l1_vel_keedmd = 0.0232935
l1_vel_ratio_keedmd = 1.0
l1_eig_keedmd = 0.0079657
l1_eig_ratio_keedmd = 0.1

# EDMD parameters
n_lift_edmd = (eigenfunction_max_power+1)**n-1
l1_edmd = 0.00687693796
l1_ratio_edmd = 1.00

# Simulation parameters (evaluate performance)
load_fit = False
test_open_loop = True
plot_open_loop = test_open_loop
save_traj = False
save_fit = not load_fit
Ntraj_pred = 30
experiment_filename = 'test_1/'
#datetime.now().strftime("%m%d%Y_%H%M%S/")
folder = 'core/examples_dev/results/'+experiment_filename
if not os.path.exists(folder):
    os.makedirs(folder)
dill_filename = folder+'models_traj.dat'
open_filename = folder+'open_loop.pdf'
closed_filename = folder+'closed_loop.pdf'
open_all_filename = folder+'open_all_loop.pdf'

#%% 
#! ===============================================    COLLECT DATA     ===============================================
#* Load trajectories
print("Collect data.")
print(" - Generate optimal desired path..",end=" ")
t0 = time.process_time()

R = sparse.eye(m)
if not load_fit:
    if (traj_origin == 'gen_MPC'):
        t_d = t_eval
        traj_bounds = [2.5,0.25,0.05,0.05] # x, theta, x_dot, theta_dot
        q_d = zeros((Ntraj,N+1,n))
        Q = sparse.diags([0,0,0,0])
        QN = sparse.diags([100000.,100000.,10000.,10000.])
        umax = 5
        MPC_horizon = 2 # [s]

        mpc_controller = MPCController(linear_dynamics=nominal_sys, 
                                    N=int(MPC_horizon/dt),
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
            x_0 = asarray([veryrandom.uniform(-i,i)  for i in traj_bounds])
            while abs(x_0[0]) < 1.25:
                x_0 = asarray([veryrandom.uniform(-i, i) for i in traj_bounds])
            mpc_controller.eval(x_0,0)
            q_d[ii,:,:] = mpc_controller.parse_result().transpose()

        savemat('./core/examples/cart_pole_d.mat', {'t_d': t_d, 'q_d': q_d})

        if plot_traj_gen:
            figure()
            title('Input Trajectories')
            for j in range(n):
                subplot(n,1,j+1)
                [plot(t_eval, q_d[ii,:,j] , linewidth=2) for ii in range(Ntraj)]
                grid()
            show()
                
            
    elif (traj_origin=='load_mat'):
        res = loadmat('./core/examples/cart_pole_d.mat') # Tensor (n, Ntraj, Ntime)
        q_d = res['q_d']  # Desired states
        t_d = res['t_d']  # Time points
        Ntraj = q_d.shape[0]  # Number of trajectories to execute

        if plot_traj_gen:
            figure()
            title('Input Trajectories')
            for j in range(n):
                subplot(n,1,j+1)
                [plot(t_eval, q_d[ii,:,j] , linewidth=2) for ii in range(Ntraj)]
                grid()
            show()

    print('in {:.2f}s'.format(time.process_time()-t0))
    t0 = time.process_time()
    # Simulate system from each initial condition
    print(' - Simulate system with {} trajectories using PD controller'.format(Ntraj), end =" ")
    save_traj = False
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


    if save_traj:
      savemat('./core/examples/results/cart_pendulum_pd_data.mat', {'xs': xs, 't_eval': t_eval, 'us': us, 'us_nom':us_nom})
    xs, us, us_nom, ts = array(xs), array(us), array(us_nom), array(ts)
    #es = xs - q_d  # Tracking error

    plot_traj = False
    if plot_traj:
        for ii in range(Ntraj):
            plot_trajectory(xs[ii], q_d[ii], us[ii], us_nom[ii], ts[ii])  # Plot simulated trajectory if desired

#%%
#!  ===============================================     FIT MODELS      ===============================================
print('in {:.2f}s'.format(time.process_time()-t0))
t0 = time.process_time()
if not load_fit:
    print("Fitting models:")
    # Construct basis of Koopman eigenfunctions for KEEDMD:
    print(' - Constructing Koopman eigenfunction basis....', end =" ")
    A_cl = A_nom - dot(B_nom,concatenate((K_p, K_d),axis=1))
    BK = dot(B_nom,concatenate((K_p, K_d),axis=1))
    eigenfunction_basis = KoopmanEigenfunctions(n=n, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK)
    eigenfunction_basis.build_diffeomorphism_model(jacobian_penalty=jacobian_penalty_diffeomorphism, n_hidden_layers = diff_n_hidden_layers, layer_width=diff_layer_width, batch_size= diff_batch_size, dropout_prob=diff_dropout_prob)
    if load_diffeomorphism_model:
        eigenfunction_basis.load_diffeomorphism_model(diffeomorphism_model_file)
    else:
        eigenfunction_basis.fit_diffeomorphism_model(X=xs, t=ts, X_d=q_d, l2=l2_diffeomorphism,
            learning_rate=diff_learn_rate, learning_decay=diff_learn_rate_decay, n_epochs=diff_n_epochs, train_frac=diff_train_frac, batch_size=diff_batch_size)
        eigenfunction_basis.save_diffeomorphism_model(diffeomorphism_model_file)
    eigenfunction_basis.construct_basis(ub=upper_bounds, lb=lower_bounds)

    if plot_eigen:
        eigenfunction_basis.plot_eigenfunction_evolution(xs, np.zeros_like(xs), t_eval)

    print('in {:.2f}s'.format(time.process_time()-t0))
    t0 = time.process_time()

    # Fit KEEDMD model:
    t0 = time.process_time()
    print(' - Fitting KEEDMD model...', end =" ")
    keedmd_model = Keedmd(eigenfunction_basis, n, l1_pos=l1_pos_keedmd, l1_ratio_pos=l1_pos_ratio_keedmd, l1_vel=l1_vel_keedmd, l1_ratio_vel=l1_vel_ratio_keedmd, l1_eig=l1_eig_keedmd, l1_ratio_eig=l1_eig_ratio_keedmd, K_p=K_p, K_d=K_d)
    X, X_d, Z, Z_dot, U, U_nom, t = keedmd_model.process(xs, q_d, us, us_nom, ts)
    keedmd_model.fit(X, X_d, Z, Z_dot, U, U_nom)
    print('in {:.2f}s'.format(time.process_time()-t0))
    
    # Construct basis of RBFs for EDMD:
    t0 = time.process_time()
    print(' - Constructing RBF basis...', end =" ")
    rbf_center_type = 'random_bounded'
    if rbf_center_type == 'random_subset':
        q_d_flat = np.reshape(q_d,(n,Ntraj*(N+1)))
        rbf_centers_vector = q_d_flat[:,np.random.choice(q_d_flat.shape[1], n_lift_edmd, replace=False)]
        rbf_centers = np.transpose(rbf_centers_vector)
        figure()
        scatter(q_d_flat[0,:],q_d_flat[2,:])
        scatter(rbf_centers_vector[0,:],rbf_centers_vector[2,:],color='red')
        grid()
        show()

    elif rbf_center_type == 'random_bounded':
        rbf_centers = multiply(random.rand(n,n_lift_edmd),(upper_bounds-lower_bounds).reshape((upper_bounds.shape[0],1)))+lower_bounds.reshape((upper_bounds.shape[0],1))
        #rbf_centers = multiply(random.rand(n, n_lift_edmd), (upper_bounds - lower_bounds).reshape(
         #   (lower_bounds.shape[0], 1))) + lower_bounds.reshape((lower_bounds.shape[0], 1))
    rbf_basis = RBF(rbf_centers, n)
    rbf_basis.construct_basis()
    print('in {:.2f}s'.format(time.process_time()-t0))
    
    # Fit EDMD model
    t0 = time.process_time()
    print(' - Fitting EDMD model...', end =" ")
    edmd_model = Edmd(rbf_basis, n, l1=l1_edmd, l1_ratio=l1_ratio_edmd)
    X, X_d, Z, Z_dot, U, U_nom, t = edmd_model.process(xs, q_d, us, us_nom, ts)
    edmd_model.fit(X, X_d, Z, Z_dot, U, U_nom)
    print('in {:.2f}s'.format(time.process_time()-t0))


#%%
#!  ==============================================  EVALUATE PERFORMANCE -- OPEN LOOP =========================================

if save_fit:
    data_list = [ q_d, t_d, edmd_model, keedmd_model, R, K_p, K_d]
    outfile = open(dill_filename,'wb')
    dill.dump(data_list,outfile)
    outfile.close()


if load_fit:
    infile = open(dill_filename,'rb')
    [ q_d, t_d, edmd_model, keedmd_model, R, K_p, K_d] = dill.load(infile)
    infile.close()


if test_open_loop:
    t0 = time.process_time()
    # Set up trajectory and controller for prediction task:
    print('Evaluate Performance with open loop prediction...', end =" ")
    t_pred = t_d.squeeze()
    noise_var_pred = 0.5

    if (traj_origin == 'gen_MPC'):
        Ntraj_pred = 40
        t_d = t_eval
        traj_bounds = [2, 0.5, 0.1, 0.1]  # x, theta, x_dot, theta_dot
        q_d_pred = zeros((Ntraj_pred, N + 1, n))

        for ii in range(Ntraj_pred):
            x_0 = asarray([random.uniform(-i, i) for i in traj_bounds])
            mpc_controller.eval(x_0, 0)
            q_d_pred[ii,:, :] = mpc_controller.parse_result().transpose()

        savemat('./core/examples/cart_pole_pred_d.mat', {'t_d': t_d, 'q_d_pred': q_d_pred})

    elif (traj_origin == 'load_mat'):
        res = loadmat('./core/examples/cart_pole_pred_d.mat')  # Tensor (n, Ntraj, Ntime)
        q_d_pred = res['q_d_pred']  # Desired states
        t_d = res['t_d']  # Time points
        Ntraj_pred = q_d.shape[0]  # Number of trajectories to execute

    # Define KEEDMD and EDMD systems:
    keedmd_sys = LinearSystemDynamics(A=keedmd_model.A, B=keedmd_model.B)
    edmd_sys = LinearSystemDynamics(A=edmd_model.A, B=edmd_model.B)

    #Simulate all different systems
    xs_pred = []
    xs_keedmd = []
    xs_edmd = []
    xs_nom = []

    # Modify KEEDMD system to account for nominal controller
    B_apnd = zeros_like(keedmd_model.B)
    B_apnd[n:, :] = -keedmd_model.B[n:, :]
    keedmd_model.B = concatenate((keedmd_model.B, B_apnd), axis=1)

    for ii in range(Ntraj_pred):
        output_pred = CartPoleTrajectory(system_true, q_d_pred[ii,:,:].T, t_pred)
        pd_controller_pred = PDController(output_pred, K_p, K_d, noise_var_pred)

        # Simulate true system (baseline):
        x0_pred = q_d_pred[ii,0,:]
        xs_pred_tmp, us_pred_tmp = system_true.simulate(x0_pred, pd_controller_pred, t_pred)
        xs_pred_tmp = xs_pred_tmp.transpose()

        # Create systems for each of the learned models and simulate with open loop control signal us_pred:
        keedmd_ol_ctrl = OpenLoopController(keedmd_sys, us_pred_tmp, t_pred[:us_pred_tmp.shape[0]])
        keedmd_controller = CompositeController(keedmd_ol_ctrl, pd_controller_pred, keedmd_model.C)
        z0_keedmd = keedmd_model.lift(x0_pred.reshape(x0_pred.shape[0],1), q_d_pred[ii,:1,:].transpose()).squeeze()
        zs_keedmd,_ = keedmd_sys.simulate(z0_keedmd,keedmd_controller,t_pred)
        xs_keedmd_tmp = dot(keedmd_model.C,zs_keedmd.transpose())

        edmd_controller = OpenLoopController(edmd_sys, us_pred_tmp, t_pred[:us_pred_tmp.shape[0]])
        z0_edmd = edmd_model.lift(x0_pred.reshape(x0_pred.shape[0],1), q_d_pred[ii,:1,:].transpose()).squeeze()
        zs_edmd,_ = edmd_sys.simulate(z0_edmd,edmd_controller,t_pred)
        xs_edmd_tmp = dot(edmd_model.C,zs_edmd.transpose())

        nom_controller = OpenLoopController(nominal_sys, us_pred_tmp, t_pred[:us_pred_tmp.shape[0]])
        xs_nom_tmp,_ = nominal_sys.simulate(x0_pred,nom_controller,t_pred)
        xs_nom_tmp = xs_nom_tmp.transpose()

        xs_pred.append(xs_pred_tmp)
        xs_keedmd.append(xs_keedmd_tmp)
        xs_edmd.append(xs_edmd_tmp)
        xs_nom.append(xs_nom_tmp)

    if save_traj:
        savemat('./core/examples/results/cart_pendulum_prediction.mat', {'t_pred':t_pred, 'xs_pred': xs_pred,
                                                                'xs_keedmd':xs_keedmd, 'xs_edmd':xs_edmd, 'xs_nom': xs_nom})

    # Calculate error statistics
    mse_keedmd  = array([(xs_keedmd[ii] - xs_pred[ii])**2 for ii in range(Ntraj_pred)])
    mse_edmd  = array([(xs_edmd[ii] - xs_pred[ii])**2 for ii in range(Ntraj_pred)])
    mse_nom   = array([(xs_nom[ii] - xs_pred[ii])**2 for ii in range(Ntraj_pred)])
    e_keedmd  = array(np.abs([xs_keedmd[ii] - xs_pred[ii] for ii in range(Ntraj_pred)]))
    e_edmd    = array(np.abs([xs_edmd[ii] - xs_pred[ii] for ii in range(Ntraj_pred)]))
    e_nom     = array(np.abs([xs_nom[ii] - xs_pred[ii] for ii in range(Ntraj_pred)]))
    mse_keedmd  = np.mean(np.mean(np.mean(mse_keedmd)))
    mse_edmd  = np.mean(np.mean(np.mean(mse_edmd)))
    mse_nom  = np.mean(np.mean(np.mean(mse_nom)))
    e_mean_keedmd  = np.mean(e_keedmd, axis=0)
    e_mean_edmd  = np.mean(e_edmd, axis=0)
    e_mean_nom = np.mean(e_nom, axis=0)
    e_std_keedmd = np.std(e_keedmd, axis=0)
    e_std_edmd = np.std(e_edmd, axis=0)
    e_std_nom = np.std(e_nom, axis=0)

    # Save open loop data for analysis and plotting:
    folder = "core/examples/results/" + datetime.now().strftime("%m%d%Y_%H%M%S")
    os.mkdir(folder)

    data_list = [t_pred, mse_keedmd, mse_edmd, mse_nom, e_keedmd, e_edmd, e_nom, e_mean_keedmd, e_mean_edmd, e_mean_nom, e_std_keedmd, e_std_edmd, e_std_nom, xs_keedmd, xs_edmd, xs_nom, xs_pred]
    outfile = open(folder + "/open_loop.pickle", 'wb')
    dill.dump(data_list, outfile)
    outfile.close()

    # Plot errors of different models and statistics
    plot_open_loop=True
    if plot_open_loop:
        ylabels = ['x', '$\\theta$', '$\\dot{x}$', '$\\dot{\\theta}$']
        figure(figsize=(5.8,10))
        for ii in range(n):
            subplot(n, 1, ii+1)
            plot(t_eval, np.abs(e_mean_nom[ii,:]), linewidth=2, label='$nom$')
            fill_between(t_eval, np.zeros_like(e_mean_nom[ii,:]), e_std_nom[ii,:], alpha=0.2)

            plot(t_eval, np.abs(e_mean_edmd[ii,:]), linewidth=2, label='$edmd$')
            fill_between(t_eval, np.zeros_like(e_mean_edmd[ii, :]), e_std_edmd[ii, :], alpha=0.2)

            plot(t_eval, np.abs(e_mean_keedmd[ii,:]), linewidth=2, label='$keedmd$')
            fill_between(t_eval, np.zeros_like(e_mean_keedmd[ii,:]), e_std_keedmd[ii, :], alpha=0.2)

            ylabel(str(ylabels[ii]))
            ylim(0., 5.)
            grid()
            if ii == 0:
                title('Predicted state evolution of different models with open loop control')
        xlabel('Time (sec)')
        legend(fontsize=10, loc='best')
        savefig(open_filename,format='pdf', dpi=2400)
        show()
        #close()
    print('in {:.2f}s'.format(time.process_time()-t0))

    print('in {:.2f}s'.format(time.process_time()-t0))

Cmatrix = control.ctrb(A=edmd_model.A, B=edmd_model.B)
print('edmd controllability matrix rank is {}, ns={}, nz={}'.format(np.linalg.matrix_rank(Cmatrix),n,edmd_model.A.shape[0]))
print(n_lift_edmd)



#%%  
#!==============================================  EVALUATE PERFORMANCE -- CLOSED LOOP =============================================
t0 = time.process_time()
print('Evaluate Performance with closed loop trajectory tracking...', end=" ")
# Set up trajectory and controller for prediction task:
x_0 = array([2., 0.25, 0., 0.])
mpc_controller.eval(x_0, 0)
q_d_pred = mpc_controller.parse_result()

x_0 = q_d_pred[:,0]
t_pred = t_d.squeeze()
t_pred = t_pred[:int(t_pred.shape[0]/2*2)]
noise_var_pred = 0.0
output_pred = CartPoleTrajectory(system_true, q_d_pred,t_pred)

# Set up MPC parameters
Q = sparse.diags([5000,5000,100,100])
QN = Q
D = sparse.diags([500,300,50,60])

upper_bounds_MPC_control = array([np.Inf, np.Inf, np.Inf, np.Inf])  # State constraints, check they are higher than upper_bounds
lower_bounds_MPC_control = -upper_bounds_MPC_control  # State constraints
umax_control = 5  # check it is higher than the control to generate the trajectories
MPC_horizon = 0.4 # [s]
plotMPC = False

# Linearized with PD
linearlize_PD_controller = PDController(output_pred, K_p, K_d, noise_var=0)
xs_lin_PD, us_lin_PD = system_true.simulate(x_0, linearlize_PD_controller, t_pred)
xs_lin_PD = xs_lin_PD.transpose()
us_lin_PD = us_lin_PD.transpose()


#* eDMD 
def check_controllability(A,B,n):
    Cmatrix = control.ctrb(A,B)
    rankCmatrix = np.linalg.matrix_rank(Cmatrix)
    print('controllability matrix rank is {}, ns={}, nz={}'.format(rankCmatrix,n,A.shape[0]))
    return rankCmatrix

# Check controllability of learned matrices:
Cmatrix = control.ctrb(A=keedmd_model.A, B=keedmd_model.B)
print('keedmd controllability matrix rank is {}, ns={}, nz={}'.format(np.linalg.matrix_rank(Cmatrix),n,keedmd_model.A.shape[0]))

Cmatrix = control.ctrb(A=edmd_model.A, B=edmd_model.B)
print('edmd controllability matrix rank is {}, ns={}, nz={}'.format(np.linalg.matrix_rank(Cmatrix),n,keedmd_model.A.shape[0]))

# Linearized with MPC
linearlize_mpc_controller = MPCControllerDense(linear_dynamics=nominal_sys,
                                                N=int(MPC_horizon/dt),
                                                dt=dt,
                                                umin=array([-umax_control]),
                                                umax=array([+umax_control]),
                                                xmin=lower_bounds_MPC_control,
                                                xmax=upper_bounds_MPC_control,
                                                Q=Q,
                                                R=R,
                                                QN=QN,
                                                xr=q_d_pred,
                                                plotMPC=plotMPC,
                                                name='Lin')

xs_lin_MPC, us_lin_MPC = system_true.simulate(x_0, linearlize_mpc_controller, t_pred)
xs_lin_MPC = xs_lin_MPC.transpose()
us_lin_MPC = us_lin_MPC.transpose()
if plotMPC:
    linearlize_mpc_controller.finish_plot(xs_lin_MPC,us_lin_MPC, us_lin_PD, t_pred,"LinMPC_thoughts.pdf")


figure()
hist(linearlize_mpc_controller.run_time*1000)
title('MPC Run Time Histogram sparse. Mean {:.2f}ms'.format(np.mean(linearlize_mpc_controller.run_time*1000)))
xlabel('Time(ms)')
savefig('MPC Run Time Histogram dense.png',format='pdf', dpi=1200)
#show()

# EDMD with MPC
edmd_sys = LinearSystemDynamics(A=edmd_model.A, B=edmd_model.B)
edmd_controller = MPCControllerDense(linear_dynamics=edmd_sys,
                                 N=int(MPC_horizon/dt),
                                 dt=dt,
                                 umin=array([-umax_control]),
                                 umax=array([+umax_control]),
                                 xmin=lower_bounds_MPC_control,
                                 xmax=upper_bounds_MPC_control,
                                 Q=Q,
                                 R=R,
                                 QN=QN,
                                 xr=q_d_pred,
                                 lifting=True,
                                 edmd_object=edmd_model,
                                 plotMPC=plotMPC,
                                 soft=True,
                                 D=D,
                                 name='EDMD')

xs_edmd_MPC, us_emdm_MPC = system_true.simulate(x_0, edmd_controller, t_pred)
xs_edmd_MPC = xs_edmd_MPC.transpose()
us_edmd_MPC = us_emdm_MPC.transpose()
    
if plotMPC:
     edmd_controller.finish_plot(xs_edmd_MPC, us_emdm_MPC, us_lin_PD, t_pred,"eDMD_thoughts.png")


#KEEDMD MPC:
keedmd_model.B = keedmd_model[:,:1] # Remove nominal controller modification
#TODO: Add matrix/lambda function for forcing term in MPC (B_apnd*K*q_d)
keedmd_sys = LinearSystemDynamics(A=keedmd_model.A, B=keedmd_model.B)
keedmd_controller = MPCControllerDense(linear_dynamics=keedmd_sys,
                                     N=int(MPC_horizon / dt),
                                     dt=dt,
                                     umin=array([-umax_control]),
                                     umax=array([+umax_control]),
                                     xmin=lower_bounds_MPC_control,
                                     xmax=upper_bounds_MPC_control,
                                     Q=Q,
                                     R=R,
                                     QN=QN,
                                     xr=q_d_pred,
                                     lifting=True,
                                     edmd_object=keedmd_model,
                                     plotMPC=plotMPC,
                                     soft=True,
                                     D=D,
                                     name='KEEDMD')

xs_keedmd_MPC, us_keemdm_MPC = system_true.simulate(x_0, keedmd_controller, t_pred)
xs_keedmd_MPC = xs_keedmd_MPC.transpose()
us_keedmd_MPC = us_keemdm_MPC.transpose()

if plotMPC:
    keedmd_controller.finish_plot(xs_keedmd_MPC, us_keemdm_MPC, us_lin_PD, t_pred, "eDMD_thoughts.png")


print('in {:.2f}s'.format(time.process_time()-t0))
t0 = time.process_time()

save_traj = False
if save_traj:
    savemat('./core/examples/results/cart_pendulum_prediction.mat', {'t_pred':t_pred, 'xs_pred': xs_pred, 'us_pred':us_pred,
                                                            'xs_keedmd':xs_keedmd, 'xs_edmd':xs_edmd, 'xs_nom': xs_nom})

#! Plot the closed loop trajectory
ylabels = ['$x$', '$\\theta$', '$\\dot{x}$', '$\\dot{\\theta}$']
figure(figsize=(5.5,10))
for ii in range(n):
    subplot(n+1, 1, ii+1)
    plot(t_pred, q_d_pred[ii,:], linestyle="--",linewidth=2, label='reference')
    plot(t_pred, xs_lin_MPC[ii, :], linewidth=2, label='Linearized dynamics with MPC', color='tab:green')
    plot(t_pred, xs_edmd_MPC[ii,:], linewidth=2, label='eDMD with MPC', color='tab:orange')
    plot(t_pred, xs_keedmd_MPC[ii,:], linewidth=2, label='KEEDMD with MPC',color='tab:gray')
    xlabel('Time (s)')
    ylabel(ylabels[ii])
    grid()
    if ii == 0:
        title('Closed loop performance of different models')
legend(fontsize=10, loc='best')
subplot(n + 1, 1, n + 1)
plot(t_pred[:-1], us_lin_MPC[0, :], linewidth=2, label='Linearized dynamics with MPC', color='tab:green')
plot(t_pred[:-1], us_edmd_MPC[0, :], linewidth=2, label='eDMD with MPC', color='tab:orange')
plot(t_pred[:-1], us_keedmd_MPC[0, :], linewidth=2, label='KEEDMD with MPC', color='tab:gray')
xlabel('Time (s)')
ylabel('u')
grid()
savefig(closed_filename,format='pdf', dpi=2400)
show()

# Calculate statistics for the different models
mse_mpc_nom = sum(sum((xs_lin_MPC-q_d_pred)**2))/xs_lin_MPC.size
mse_mpc_edmd = sum(sum((xs_edmd_MPC-q_d_pred)**2))/xs_edmd_MPC.size
mse_mpc_keedmd = sum(sum((xs_keedmd_MPC-q_d_pred)**2))/xs_keedmd_MPC.size
E_nom = np.linalg.norm(us_lin_MPC)
E_edmd = np.linalg.norm(us_edmd_MPC)
E_keedmd = np.linalg.norm(us_keedmd_MPC)

Q_d = Q.todense()
R_d = R.todense()
cost_nom = sum(np.diag(np.dot(np.dot((xs_lin_MPC-q_d_pred).T,Q_d), xs_lin_MPC-q_d_pred))) + sum(np.diag(np.dot(np.dot(us_lin_MPC.T,R_d),us_lin_MPC)))
cost_edmd = sum(np.diag(np.dot(np.dot((xs_edmd_MPC-q_d_pred).T,Q_d), xs_edmd_MPC-q_d_pred))) + sum(np.diag(np.dot(np.dot(us_edmd_MPC.T,R_d),us_edmd_MPC)))
cost_keedmd = sum(np.diag(np.dot(np.dot((xs_keedmd_MPC-q_d_pred).T,Q_d), xs_keedmd_MPC-q_d_pred))) + sum(np.diag(np.dot(np.dot(us_keedmd_MPC.T,R_d),us_keedmd_MPC)))
print('Tracking error (MSE), Nominal: ', mse_mpc_nom, ', EDMD: ', mse_mpc_edmd, 'KEEDMD: ', mse_mpc_keedmd)
print('Control effort (norm), Nominal:  ', E_nom, ', EDMD: ', E_edmd, ', KEEDMD: ', E_keedmd)
print('MPC cost, Nominal: ', cost_nom, ', EDMD: ', cost_edmd, ', KEEDMD: ', cost_keedmd)
print('MPC cost improvement, EDMD: ', (cost_edmd/cost_nom-1)*100, '%, KEEDMD: ', (cost_keedmd/cost_nom-1)*100, '%')

# Save closed loop data for analysis and plotting:
data_list = [t_pred, q_d_pred, xs_lin_MPC, xs_edmd_MPC, xs_keedmd_MPC, us_lin_MPC, us_edmd_MPC, us_keedmd_MPC, mse_mpc_nom, mse_mpc_edmd, mse_mpc_keedmd, E_nom, E_edmd, E_keedmd, cost_nom, cost_edmd, cost_keedmd]
outfile = open(folder + "/closed_loop.pickle", 'wb')
dill.dump(data_list, outfile)
outfile.close()