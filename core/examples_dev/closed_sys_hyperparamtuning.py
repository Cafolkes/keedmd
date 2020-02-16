#%%
"""Closed Koopman system example"""
from ..dynamics import RoboticDynamics, LinearSystemDynamics, AffineDynamics, SystemDynamics
from scipy.io import loadmat, savemat
from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title, scatter, savefig, hist
from numpy import arange, array, concatenate, cos, identity, dstack
from numpy import linspace, ones, sin, tanh, tile, zeros, pi, random, interp, dot, multiply, asarray, zeros_like
import numpy as np
from core.dynamics import LinearSystemDynamics
from core.controllers import PDController, OpenLoopController, MPCController, MPCControllerDense
from core.learning import KoopmanEigenfunctions, RBF, Edmd, Keedmd, plot_trajectory, IdentityBF, Monomials
from core.systems import ClosedSubspaceSys
import time
import dill
from pathlib import Path

# %%
# ! ===============================================   SET PARAMETERS    ===============================================

# Tuning parameters
#folder = str(Path().absolute()) + '/experiments/episodic_KEEDMD/fast_drone_landing/'
#datafile_lst = [folder + '09132019_222031/episodic_data.pickle', folder + '09132019_231840/episodic_data.pickle'] #Add multiple paths to list if multiple data files

# Define true system
mu, lambd = -0.2, -1.
system_true = ClosedSubspaceSys(mu, lambd)
n, m = 2, 0                                             # Number of states and actuators
upper_bounds = array([1, 1])                            # Upper State constraints
lower_bounds = -upper_bounds                            # Lower State constraints

# Define nominal model and nominal controller:
A_nom = array([[mu, 0.], [0., lambd]])                  # Linearization of the true system around the origin
B_nom = array([[0.],[0.]])                              # Linearization of the true system around the origin
K_p = -array([[0.]])                                    # Proportional control gains
K_d = -array([[0.]])                                    # Derivative control gains
A_cl = A_nom - dot(B_nom,concatenate((K_p, K_d),axis=1))
BK = dot(B_nom,concatenate((K_p, K_d),axis=1))
nominal_sys = LinearSystemDynamics(A=A_nom, B=B_nom)

# Diffeomorphism tuning parameters:
l2_diffeomorphism = np.linspace(0.,5e-1, 20)
jacobian_penalty_diffeomorphism = [1e1]
diff_n_epochs = [50, 100, 200, 500]
diff_n_hidden_layers = [1, 2, 3, 4]
diff_layer_width = [10, 25, 50]
diff_batch_size = [8, 16, 32]
diff_learn_rate = np.linspace(1e-5, 1e-1, 20)  # Fix for current architecture
diff_learn_rate_decay = [0.8, 0.9, 0.95, 0.975, 0.99, 1.0]
diff_dropout_prob = [0., 0.05, 0.1, 0.25, 0.5]

n_folds = 2
n_search = 1000
tune_diffeomorphism = True
diffeomorphism_model_file = 'diff_model'
NN_parameter_file = 'scripts/NN_parameters.pickle'

# EDMD models tuning parameters
eigenfunction_max_power = 3
n_lift_edmd_monomials = (eigenfunction_max_power + 1) ** n - 1
n_lift_edmd_rbf = (eigenfunction_max_power + 1) ** n - 1
l1_ratio = array([.1, .5, .7, .9, .95, .99, 1])  # Values to test
tune_edmd_models = True

# Simulation parameters (data collection)
Ntraj = 50                                              # Number of trajectories to collect data from
dt = 1.0e-1                                             # Time step length
N = int(7.5/dt)                                         # Number of time steps
t_eval = dt * arange(N + 1)                             # Simulation time points

#%%
#! ===============================================    COLLECT DATA     ===============================================
print("Collect data:")

# Simulate system from each initial condition
print(' - Simulate system with {} trajectories using PD controller'.format(Ntraj), end =" ")
t0 = time.process_time()

xf = ones((2,))
empty_controller = OpenLoopController(system_true, np.atleast_2d(zeros_like(t_eval)).T, t_eval)
xs, us, ts = [], [], []
for ii in range(Ntraj):
    x_0 = np.random.uniform(-1,1,(2,))
    x_0 /= np.linalg.norm(x_0)
    xs_tmp, us_tmp = system_true.simulate(x_0, empty_controller, t_eval)
    xs.append(xs_tmp)
    us.append(us_tmp)
    ts.append(t_eval)

xs, us, us_nom, ts = array(xs), array(us), array(us), array(ts) # us, us_nom all zero dummy variables

print('in {:.2f}s'.format(time.process_time()-t0))

# %%
# !  ======================================     TUNE DIFFEOMORPHISM MODEL      ========================================
print('Start random parameter search:')
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
            Xd = np.zeros_like(xs[train_inds,:,:])
            t_val = ts[val_inds, :]
            X_val = xs[val_inds, :, :]
            Xd_val = np.zeros_like(xs[val_inds, :, :])

            # Fit model with current data set and hyperparameters
            eigenfunction_basis = KoopmanEigenfunctions(n=n, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK)
            eigenfunction_basis.build_diffeomorphism_model(jacobian_penalty=jac_pen,n_hidden_layers=n_hidden,
                                                            layer_width=layer_width,
                                                            batch_size=batch_size,
                                                            dropout_prob=dropout)

            score_tmp = eigenfunction_basis.fit_diffeomorphism_model(X=X, t=t, X_d=Xd, l2=l2,
                                                            learning_rate=learn_rate,
                                                            learning_decay=rate_decay, n_epochs=n_epochs,
                                                            train_frac=None, batch_size=batch_size, initialize=True,
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
if tune_edmd_models:
    # Fit KEEDMD model:
    print('Tuning KEEDMD...')
    keedmd_model = Keedmd(eigenfunction_basis, n, K_p=K_p, K_d=K_d)
    X, X_d, Z, Z_dot, _, _, t = keedmd_model.process(xs, zeros_like(xs), np.atleast_3d(us), np.atleast_3d(us_nom), ts)
    keedmd_model.tune_fit(X, X_d, Z, Z_dot)

    print('Tuning KEEDMD full A-matrix learning...')
    keedmd_model_full = Edmd(eigenfunction_basis, n)
    X, X_d, Z, Z_dot, _, _, t = keedmd_model_full.process(xs, zeros_like(xs), np.atleast_3d(us), np.atleast_3d(us_nom),ts)
    keedmd_model_full.tune_fit(X, X_d, Z, Z_dot)

    # Construct basis of RBFs and fit EDMD:
    print('Tuning EDMD RBFs...')
    rbf_centers = multiply(random.rand(n, n_lift_edmd_rbf),
                           (upper_bounds - lower_bounds).reshape((upper_bounds.shape[0], 1))) + lower_bounds.reshape(
        (upper_bounds.shape[0], 1))
    rbf_basis = RBF(rbf_centers, n, gamma=2.)
    rbf_basis.construct_basis()

    edmd_model_rbf = Edmd(rbf_basis, n)
    X, X_d, Z, Z_dot, _, _, t = edmd_model_rbf.process(xs, zeros_like(xs), np.atleast_3d(us), np.atleast_3d(us_nom), ts)
    edmd_model_rbf.tune_fit(X, X_d, Z, Z_dot)

    # Construct basis of monomials and fit EDMD:
    print('Tuning EDMD Monomials...')
    monomial_basis = Monomials(n, n_lift_edmd_monomials)
    monomial_basis.construct_basis()

    edmd_model_monomial = Edmd(monomial_basis, n)
    X, X_d, Z, Z_dot, _, _, t = edmd_model_monomial.process(xs, zeros_like(xs), np.atleast_3d(us),
                                                            np.atleast_3d(us_nom), ts)
    edmd_model_monomial.tune_fit(X, X_d, Z, Z_dot)
