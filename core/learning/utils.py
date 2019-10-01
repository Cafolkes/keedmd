from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title, savefig, ylim, ylabel, xlabel
from numpy import array, gradient, zeros, tile
import numpy as np

def plot_trajectory(X, X_d, U, U_nom, t, display=True, save=False, filename=''):
    """ Plots the position, velocity and control input

    # Inputs:
    - state X, numpy 2d array [number of time steps 'N', number of states 'n'] 
    - desired state X_d, numpy 2d array [number of time steps 'N', number of states 'n']
    - control input U, numpy 2d array [number of time steps, number of inputs]
    - nominal control input U_nom, numpy 2d array [number of time steps, number of inputs]
    - time t, numpy 1d array [number of time steps 'N']
    """
    figure()
    subplot(2, 1, 1)
    plot(t, X[:,0], linewidth=2, label='$x$')
    plot(t, X[:,2], linewidth=2, label='$\\dot{x}$')
    plot(t, X_d[:,0], '--', linewidth=2, label='$x_d$')
    plot(t, X_d[:,2], '--', linewidth=2, label='$\\dot{x}_d$')
    title('Trajectory Tracking with PD controller')
    legend(fontsize=12)
    grid()
    subplot(2, 1, 2)
    plot(t[:-1], U[:,0], label='$u$')
    plot(t[:-1], U_nom[:,0], label='$u_{nom}$')
    legend(fontsize=12)
    grid()
    if display:
        show()
    if save:
        savefig(filename)

def plot_trajectory_ep(X, X_d, U, U_nom, t, display=True, save=False, filename='', episode=0):
    # Plot the first simulated trajectory
    figure(figsize=(4.7,5.5))
    subplot(3, 1, 1)
    title('Trajectory tracking with MPC, episode ' + str(episode))
    plot(t, X[0,:], linewidth=2, label='$x$')
    plot(t, X[2,:], linewidth=2, label='$\\dot{x}$')
    plot(t, X_d[0,:], '--', linewidth=2, label='$x_d$')
    plot(t, X_d[2,:], '--', linewidth=2, label='$\\dot{x}_d$')
    legend(fontsize=10, loc='lower right', ncol=4)
    ylim((-4.5, 2.5))
    ylabel('$x$, $\\dot{x}$')
    grid()
    subplot(3, 1, 2)
    plot(t, X[1,:], linewidth=2, label='$\\theta$')
    plot(t, X[3,:], linewidth=2, label='$\\dot{\\theta}$')
    plot(t, X_d[1,:], '--', linewidth=2, label='$\\theta_d$')
    plot(t, X_d[3,:], '--', linewidth=2, label='$\\dot{\\theta}_d$')
    legend(fontsize=10, loc='lower right', ncol=4)
    ylim((-2.25,1.25))
    ylabel('$\\theta$, $\\dot{\\theta}$')
    grid()

    subplot(3, 1, 3)
    plot(t[:-1], U[0,:], label='$u$')
    plot(t[:-1], U_nom[0,:], label='$u_{nom}$')
    legend(fontsize=10, loc='upper right', ncol=2)
    ylabel('u')
    xlabel('Time (sec)')
    grid()
    if save:
        savefig(filename)
    if display:
        show()

def differentiate_vec(xs, ts, L=3):
    assert(xs.shape[0] == ts.shape[0])
    return array([differentiate(xs[:,ii], ts) for ii in range(xs.shape[1])]).transpose()

def differentiate(xs, ts):
    """
    Compute the discrete derivative of a Python function
    f on [a,b] using n intervals. Internal points apply
    a centered difference, while end points apply a one-sided
    difference. Vectorized version.
    """
    #dx = zeros_like(xs)                     # dx/dt
    dt = ts[1] - ts[0]
    #dx[1:-1] = (xs[2:] - xs[:-2])/(2*dt)    # Internal mesh points
    #dx[0]  = (xs[1]  - xs[0])/dt           # End point
    #dx[-1] = (xs[-1] - xs[-2])/dt           # End point
    dx = gradient(xs, dt, edge_order=2)
    return dx

def rbf(X, C, type='gauss', eps=1.):
    N = X.shape[1]
    n = X.shape[0]
    Cbig = C
    Y = zeros((C.shape[1],N))
    for ii in range(C.shape[1]):
        C = Cbig[:,ii]
        C = tile(C.reshape((C.size,1)), (1, N))
        r_sq = np.sum((X-C)**2,axis=0)
        if type == 'gauss':
            y = np.exp(-eps**2*r_sq)

        Y[ii,:] = y

    return Y