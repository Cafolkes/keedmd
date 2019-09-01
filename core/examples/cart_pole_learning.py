"""Cart Pendulum Example"""

from os import path
import sys
from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title
from numpy import arange, array, concatenate, cos, identity, linspace, ones, sin, tanh, tile, zeros, pi, random, interp, dot
#from numpy.random import uniform
from scipy.io import loadmat, savemat
from sys import argv
from core.systems import CartPole
from core.dynamics import LinearSystemDynamics
from core.controllers import PDController
from core.learning_keedmd import KoopmanEigenfunctions


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


# Define true system
system_true = CartPole(m_c=.5, m_p=.2, l=.4)
n, m = 4, 1  # Number of states and actuators
upper_bounds = array([[2.5, pi/3, 2, 2]])  # State constraints
lower_bounds = -upper_bounds  # State constraints

# Define nominal model and nominal controller:
A_nom = array([[0., 0., 1., 0.], [0., 0., 0., 1.], [0., -3.924, 0., 0.], [0., 34.335, 0., 0.]])  # Linearization of the true system around the origin
B_nom = array([[0.],[0.],[2.],[-5.]])  # Linearization of the true system around the origin
K_p = -array([[7.3394, 39.0028]])  # Proportional control gains
K_d = -array([[8.0734, 7.4294]])  # Derivative control gains
nominal_model = LinearSystemDynamics(A=A_nom, B=B_nom)

# Load trajectories
res = loadmat('./core/examples/cart_pole_d.mat') # Tensor (n, Ntraj, Ntime)
q_d = res['Q_d']  # Desired states
t_d = res['t_d']  # Time points
Ntraj = q_d.shape[1]  # Number of trajectories to execute #TODO: Reset to sim all trajectories

# Simulation parameters
dt = 1.0e-2  # Time step
N = t_d[0,-1]/dt  # Number of time steps
t_eval = dt * arange(N + 1) # Simulation time points

# Simulate system from each initial condition
outputs = [CartPoleTrajectory(system_true, q_d[:,i,:],t_d) for i in range(Ntraj)]
pd_controllers = [PDController(outputs[i], K_p, K_d) for i in range(Ntraj)]
xs, us = [], []
for ii in range(Ntraj):
    x_0 = q_d[:,ii,0]
    xs_tmp, us_tmp = system_true.simulate(x_0, pd_controllers[ii], t_eval)
    xs.append(xs_tmp)
    us.append(us_tmp)
savemat('./core/examples/results/cart_pendulum_pd_data.mat', {'xs': xs, 't_eval': t_eval, 'us': us})

# Plot the first simulated trajectory
figure()
subplot(2, 1, 1)
plot(t_eval, xs[2][:,0], linewidth=2, label='$x$')
plot(t_eval, xs[2][:,2], linewidth=2, label='$\\dot{x}$')
plot(t_eval, q_d[0,2,:], '--', linewidth=2, label='$x_d$')
plot(t_eval, q_d[2,2,:], '--', linewidth=2, label='$\\dot{x}_d$')
title('Trajectory Tracking with PD controller (2nd trajectory plotted)')
legend(fontsize=12)
grid()
subplot(2, 1, 2)
plot(t_eval[:-1], us[2][:,0], label='$u$')
legend(fontsize=12)
grid()
#show()  # TODO: Create plot of all collected trajectories (subplot with one plot for each state), not mission critical

# Construct basis of Koopman eigenfunctions
A_cl = A_nom - dot(B_nom,concatenate((K_p, K_d),axis=1))
eigenfunction_basis = KoopmanEigenfunctions(n=n, max_power=3, A_cl=A_cl)
eigenfunction_basis.build_diffeomorphism_model()
eigenfunction_basis.fit_diffeomorphism_model(X=xs, t=t_eval, X_d=q_d)
eigenfunction_basis.construct_basis(ub=upper_bounds, lb=lower_bounds)



