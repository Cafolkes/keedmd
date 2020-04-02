#%%
"""Inverted Pendulum Example"""
from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title, ylim, xlabel, ylabel, xticks, yticks, imshow, colorbar, tight_layout
from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title, scatter, savefig, hist, bar
from numpy import arange, array, concatenate, cos, identity
from numpy import linspace, ones, sin, tanh, tile, zeros, pi, random, interp, dot, multiply, asarray, zeros_like
import numpy as np
from ..dynamics import RoboticDynamics, LinearSystemDynamics, AffineDynamics, SystemDynamics
from ..controllers import PDController, OpenLoopController, MPCController, MPCControllerDense
from ..learning import KoopmanEigenfunctions, RBF, Edmd, Keedmd, Monomials
from ..systems import ClosedSubspaceSys
from ..learning.utils import calc_koopman_modes, calc_reduced_mdl, differentiate_vec
import time

#%%
#! ===============================================   SET PARAMETERS    ===============================================

# Define true system
mu, lambd = -0.2, -1.
system_true = ClosedSubspaceSys(mu, lambd)
n, m = 2, 0                                                 # Number of states and actuators
upper_bounds = array([1, 1])                                # Upper State constraints
lower_bounds = -upper_bounds                                # Lower State constraints

# Define nominal model and nominal controller:
A_nom = array([[mu, 0.], [0., lambd]])                      # Linearization of the true system around the origin
B_nom = array([[0.],[0.]])                                  # Linearization of the true system around the origin
K_p = -array([[0.]])                                        # Proportional control gains
K_d = -array([[0.]])                                        # Derivative control gains
nominal_sys = LinearSystemDynamics(A=A_nom, B=B_nom)

# Simulation parameters (data collection)
Ntraj = 20                                                  # Number of trajectories to collect data from
dt = 1.0e-1                                                 # Time step length
N = int(10/dt)                                              # Number of time steps
t_eval = dt * arange(N + 1)                                 # Simulation time points

# Koopman eigenfunction parameters
eigenfunction_max_power = 3                                 # Max power of variables in eigenfunction products
l2_diffeomorphism = 0.0                                     # l2 regularization strength
jacobian_penalty_diffeomorphism = 1e1                       # Estimator jacobian regularization strength
diff_n_epochs = 200                                         # Number of epochs
diff_train_frac = 0.9                                       # Fraction of data to be used for training
diff_n_hidden_layers = 2                                    # Number of hidden layers
diff_layer_width = 25                                       # Number of units in each layer
diff_batch_size = 8                                         # Batch size
diff_learn_rate = 1e-1                                      # Leaning rate
diff_learn_rate_decay = 0.975                               # Learning rate decay
diff_dropout_prob = 0.0                                     # Dropout rate

# KEEDMD parameters
l1_keedmd = 3.759897798258798e-05                           # l1 regularization strength for position states
l1_ratio_keedmd = 1.0                                       # l1-l2 ratio for position states
l1_ratio_edmd = [0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 1.0]

# EDMD Monomials parameters (benchmark to compare against)
n_lift_edmd_monomials = (eigenfunction_max_power+1)**n-1    # Lifting dimension EDMD (same number as for KEEDMD)
l1_edmd_monomials = 8.762018872019091e-05                   # l1 regularization strength
l1_ratio_edmd_monomials = 1.0                               # l1-l2 ratio

# EDMD RBFs parameters (benchmark to compare against)
n_lift_edmd_rbf = (eigenfunction_max_power+1)**n-1          # Lifting dimension EDMD (same number as for KEEDMD)
l1_edmd_rbf = 0.0011895308703369347                         # l1 regularization strength
l1_ratio_edmd_rbf = 0.5                                     # l1-l2 ratio

#%%
#! ===============================================    COLLECT DATA     ===============================================
print("Collect data:")

# Simulate system from each initial condition
print(' - Simulate system with {} trajectories using PD controller'.format(Ntraj), end =" ")
t0 = time.process_time()

xf = ones((2,))
empty_controller = OpenLoopController(system_true, np.atleast_2d(zeros_like(t_eval)).T, t_eval)
angle_0 = np.linspace(0, 2*pi, Ntraj)
xs, us, ts = [], [], []
for ii in range(Ntraj):
    x_0 = array([np.cos(angle_0[ii]), np.sin(angle_0[ii])])
    xs_tmp, us_tmp = system_true.simulate(x_0, empty_controller, t_eval)
    xs.append(xs_tmp)
    us.append(us_tmp)
    ts.append(t_eval)

xs, us, us_nom, ts = array(xs), array(us), array(us), array(ts) # us, us_nom all zero dummy variables

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
#keedmd_model = Keedmd(eigenfunction_basis, n, l1_pos=l1_pos_keedmd, l1_ratio_pos=l1_pos_ratio_keedmd, l1_vel=l1_vel_keedmd, l1_ratio_vel=l1_vel_ratio_keedmd, l1_eig=l1_eig_keedmd, l1_ratio_eig=l1_eig_ratio_keedmd, K_p=K_p, K_d=K_d)
#X, X_d, Z, Z_dot, U, U_nom, t = keedmd_model.process(xs, zeros_like(xs), np.atleast_3d(us), np.atleast_3d(us_nom), ts)
#keedmd_model.tune_fit(X, X_d, Z, Z_dot, U, U_nom)

#keedmd_model_full = Edmd(eigenfunction_basis, n, l1=1e-3, l1_ratio=l1_ratio_edmd_monomials, override_C=False, add_ones=False, add_state=False)
keedmd_model_full = Edmd(eigenfunction_basis, n, l1=l1_keedmd, l1_ratio=l1_ratio_keedmd)
X, X_d, Z, Z_dot, _, _, t = keedmd_model_full.process(xs, zeros_like(xs), np.atleast_3d(us), np.atleast_3d(us_nom), ts)
#keedmd_model_full.fit(X, X_d, Z, Z_dot)
keedmd_model_full.tune_fit(X, X_d, Z, Z_dot, l1_ratio=l1_ratio_edmd)

print('in {:.2f}s'.format(time.process_time()-t0))

# Construct basis of RBFs and fit EDMD:
print(' - Constructing RBF basis and Inferring EDMD model...', end =" ")
t0 = time.process_time()

rbf_centers = multiply(random.rand(n,n_lift_edmd_rbf),(upper_bounds-lower_bounds).reshape((upper_bounds.shape[0],1)))+lower_bounds.reshape((upper_bounds.shape[0],1))
rbf_basis = RBF(rbf_centers, n, gamma=2.)
rbf_basis.construct_basis()

edmd_model_rbf = Edmd(rbf_basis, n, l1=l1_edmd_rbf, l1_ratio=l1_ratio_edmd_rbf)
X, X_d, Z, Z_dot, _, _, t = edmd_model_rbf.process(xs, zeros_like(xs), np.atleast_3d(us), np.atleast_3d(us_nom), ts)
#edmd_model_rbf.fit(X, X_d, Z, Z_dot)
edmd_model_rbf.tune_fit(X, X_d, Z, Z_dot, l1_ratio=l1_ratio_edmd)

print('in {:.2f}s'.format(time.process_time()-t0))

# Construct basis of monomials and fit EDMD:
print(' - Constructing monomial basis and Inferring EDMD model...', end =" ")
t0 = time.process_time()

monomial_basis = Monomials(n, n_lift_edmd_monomials)
monomial_basis.construct_basis()

edmd_model_monomial = Edmd(monomial_basis, n, l1=l1_edmd_monomials, l1_ratio=l1_ratio_edmd_monomials)
X, X_d, Z, Z_dot, _, _, t = edmd_model_monomial.process(xs, zeros_like(xs), np.atleast_3d(us), np.atleast_3d(us_nom), ts)
edmd_model_monomial.fit(X, X_d, Z, Z_dot)
#edmd_model_monomial.tune_fit(X, X_d, Z, Z_dot, l1_ratio=l1_ratio_edmd)

print('in {:.2f}s'.format(time.process_time()-t0))

# Reduce model by removing columns not used for prediction:
A_red_keedmd, _, C_red_keedmd, useful_cols_keedmd = calc_reduced_mdl(keedmd_model_full)
A_red_edmd_rbf, _, C_red_edmd_rbf, useful_cols_edmd_rbf = calc_reduced_mdl(edmd_model_rbf)
A_red_edmd_monominal, _, C_red_edmd_monominal, useful_cols_edmd_monomial = calc_reduced_mdl(edmd_model_monomial)


#%%
#!  ==============================================  COMPARE KOOPMAN MODES =============================================
#x_0 = np.random.uniform(-1, 1, (2,))
#x_0 /= np.linalg.norm(x_0)
angle = 3*pi/4
x_0 = array([np.cos(angle), np.sin(angle)])
xs_eval, _ = system_true.simulate(x_0, empty_controller, t_eval)

koopman_outputs = lambda x, t: array([x[0], x[1], x[0]**2])
A_koop = array([[mu, 0., 0.],[0., lambd, -lambd], [0., 0., 2*mu]])
xs_modes, v, w, d = calc_koopman_modes(A_koop, koopman_outputs, x_0, t_eval)

output_edmd_monomial = lambda x, t: np.concatenate((np.atleast_2d(x), ones((1,1)), edmd_model_monomial.basis.lift(x.reshape(-1,1), zeros_like(x).reshape(-1,1))), axis=1).squeeze()[useful_cols_edmd_monomial]
#output_edmd_monomial = lambda x, t: edmd_model_monomial.basis.lift(x.reshape(-1,1), zeros_like(x).reshape(-1,1)).squeeze()[useful_cols_edmd_monomial]
xs_edmd_monomial_modes, v_edmd_monomial, w_edmd_monomial, d_edmd_monomial = calc_koopman_modes(A_red_edmd_monominal, output_edmd_monomial, x_0, t_eval)

output_edmd_rbf = lambda x, t: np.concatenate((np.atleast_2d(x), ones((1,1)), edmd_model_rbf.basis.lift(x, t)), axis=1).squeeze()[useful_cols_edmd_rbf]
#output_edmd_rbf = lambda x, t: edmd_model_rbf.basis.lift(x, t).squeeze()[useful_cols_edmd_rbf]
xs_edmd_rbf_modes, v_edmd_rbf, w_edmd_rbf, d_edmd_rbf = calc_koopman_modes(A_red_edmd_rbf, output_edmd_rbf, x_0, t_eval)

#output_keedmd = lambda x, t: np.concatenate((x, keedmd_model.basis.basis(x.reshape((-1,1)), zeros_like(x).reshape((-1,1)))))
#xs_keedmd_modes, v_keedmd, w_keedmd, d_keedmd = calc_koopman_modes(A_red_keedmd, output_keedmd, x_0, t_eval)

output_keedmd_full = lambda x, t: np.concatenate((x, ones((1,)), keedmd_model_full.basis.basis(x.reshape((-1,1)), zeros_like(x).reshape((-1,1)))))[useful_cols_keedmd]
#output_keedmd_full = lambda x, t: keedmd_model_full.basis.basis(x.reshape((-1,1)), zeros_like(x).reshape((-1,1)))[useful_cols_keedmd]
xs_keedmd_full_modes, v_keedmd_f, w_keedmd_f, d_keedmd_f = calc_koopman_modes(A_red_keedmd, output_keedmd_full, x_0, t_eval)

# Simulate learned systems to evaluate prediction performance:
edmd_rbf_sys = LinearSystemDynamics(A=A_red_edmd_rbf, B=zeros((A_red_edmd_rbf.shape[0],1)))
z0_edmd_rbf = output_edmd_rbf(x_0, 0.)
zs_edmd_rbf, _ = edmd_rbf_sys.simulate(z0_edmd_rbf,empty_controller, t_eval)
xs_edmd_rbf = np.dot(C_red_edmd_rbf, zs_edmd_rbf.T)

edmd_monomials_sys = LinearSystemDynamics(A=A_red_edmd_monominal, B=zeros((A_red_edmd_monominal.shape[0],1)))
z0_edmd_monomials = output_edmd_monomial(x_0, 0.)
zs_edmd_monomials, _ = edmd_monomials_sys.simulate(z0_edmd_monomials,empty_controller, t_eval)
xs_edmd_monomials = np.dot(C_red_edmd_monominal, zs_edmd_monomials.T)

keedmd_sys = LinearSystemDynamics(A=A_red_keedmd, B=zeros((A_red_keedmd.shape[0],1)))
z0_keedmd = output_keedmd_full(x_0, 0.)
zs_keedmd, _ = keedmd_sys.simulate(z0_keedmd,empty_controller, t_eval)
xs_keedmd = np.dot(C_red_keedmd, zs_keedmd.T)


width = 0.2
vmin = -1.5
vmax = 1.5

n_steps = 20
x_max = 0.75
x_pts = np.linspace(-x_max,x_max,n_steps)
x_heatmap = np.meshgrid(x_pts, x_pts)
diff_analytic = lambda x: [0, lambd/(2*mu-lambd)*x[0]**2]
diff_1_analytic_grid = np.zeros((n_steps,n_steps))
diff_2_analytic_grid = np.zeros((n_steps,n_steps))
diff_1_dnn_grid = np.zeros((n_steps,n_steps))
diff_2_dnn_grid = np.zeros((n_steps,n_steps))
for ii in range(n_steps):
    for jj in range(n_steps):
        diff_1_analytic_grid[ii, jj] = x_heatmap[0][ii, jj]
        diff_2_analytic_grid[ii,jj] = x_heatmap[1][ii, jj] + diff_analytic([x_heatmap[0][ii, jj], x_heatmap[1][ii, jj]])[1]
        diff_1_dnn_grid[ii, jj] = eigenfunction_basis.diffeomorphism(array([x_heatmap[0][ii, jj], x_heatmap[1][ii, jj]]).reshape(-1,1), array([0., 0.]).reshape(-1,1)).squeeze()[0]
        diff_2_dnn_grid[ii, jj] = eigenfunction_basis.diffeomorphism(array([x_heatmap[0][ii, jj], x_heatmap[1][ii, jj]]).reshape(-1,1), array([0., 0.]).reshape(-1,1)).squeeze()[1]

fig = figure(figsize=(10,4))
"""
subplot(4,2,1)
imshow(diff_1_analytic_grid, vmin=vmin, vmax=vmax, cmap='hot')
title('Analytic $c_1(x)$')
colorbar()
#xlabel('x_1')
#ylabel('x_2')

ax2 = subplot(4,2,2)
imshow(diff_1_dnn_grid, vmin=vmin, vmax=vmax, cmap='hot')
title('Learned (DNN) $c_1(x)$')
colorbar()
#xlabel('x_1')
#ylabel('x_2')
"""

ax3 = subplot(2,4,1)
imshow(diff_2_analytic_grid, vmin=vmin, vmax=vmax, cmap='hot')
title('Analytic $c_2(x)$')
colorbar()
xticks([0, n_steps], [-x_max, x_max])
yticks([0, n_steps], [x_max, -x_max])
xlabel('$x_1$')
ylabel('$x_2$')

ax4 = subplot(2,4,2)
imshow(diff_2_dnn_grid, vmin=vmin, vmax=vmax, cmap='hot')
title('Learned (NN) $c_2(x)$')
colorbar()
xticks([0, n_steps], [-x_max, x_max])
yticks([0, n_steps], [x_max, -x_max])
xlabel('$x_1$')
ylabel('$x_2$')

subplot(2,2,3)
bar(np.linspace(1,d.size, d.size, dtype=int)-1.5*width, np.flip(np.abs(d)), color='r', width=width, label='Analytic')
bar(np.linspace(1,d_keedmd_f.size, d_keedmd_f.size, dtype=int)-0.5*width, np.flip(np.abs(d_keedmd_f)), width=width, label='KEEDMD')
bar(np.linspace(1,d_edmd_monomial.size, d_edmd_monomial.size, dtype=int)+0.5*width, np.flip(np.abs(d_edmd_monomial)), width=width, label='EDMD, monomial')
bar(np.linspace(1,d_edmd_rbf.size, d_edmd_rbf.size, dtype=int)+1.5*width, np.flip(np.abs(d_edmd_rbf)), width=width, label='EDMD, rbf')
xticks(np.linspace(1, max(d.size, d_edmd_monomial.size, d_keedmd_f.size), max(d.size, d_edmd_monomial.size, d_keedmd_f.size), dtype=int))

title('Active Eigenvalue Magnitudes')
xlabel('Eigenvalue #')
ylabel('Mgnitude, $| \lambda |$')
legend(loc='best')

subplot(1,2,2)
plot(xs_eval[:,0], xs_eval[:,1], ':r', linewidth=3, label='True')
#plot(zs_edmd_monomials[:,0], zs_edmd_monomials[:, 1], label='EDMD, monomials')
#plot(zs_edmd_rbf[:,0], zs_edmd_rbf[:, 1], label='EDMD, RBF')
#plot(zs_keedmd[:,0], zs_keedmd[:,1], label='KEEDMD')
plot(xs_keedmd[0,:], xs_keedmd[1,:], label='KEEDMD')
plot(xs_edmd_monomials[0,:], xs_edmd_monomials[1,:], label='EDMD, monomial')
plot(xs_edmd_rbf[0,:], xs_edmd_rbf[1,:], label='EDMD, RBF')
title('Phase Plot of True and Predicted Evolution')
xlabel('$x_1$')
ylabel('$x_2$')
legend(loc='lower left')

tight_layout()
savefig('koopman_mode_comp_8.pdf', format='pdf', dpi=2400)
show()
"""
subplot(4,2,8)
plot(t_eval, xs_eval[:,1], ':r', linewidth=3, label='True')
plot(t_eval, zs_edmd_monomials[:, 1], label='EDMD, monomials')
plot(t_eval, zs_edmd_rbf[:, 1], label='EDMD, RBF')
plot(t_eval, zs_keedmd[:, 1], label='KEEDMD')
title('Predicted evolution, $x_2$')
xlabel('Time (sec)')
legend(loc='best')
show()

xs_data = xs.reshape((xs.shape[0]*xs.shape[1], xs.shape[2]))
figure()
scatter(xs_data[:,0], xs_data[:,1])
xlabel('$x_1$')
ylabel('$x_2$')
show()
"""