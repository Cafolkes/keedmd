from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title, savefig, ylim, ylabel, xlabel

def plot_trajectory(X, X_d, U, U_nom, t, display=True, save=False, filename=''):
    # Plot the first simulated trajectory
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
    plot(t, X[:,0], linewidth=2, label='$x$')
    plot(t, X[:,2], linewidth=2, label='$\\dot{x}$')
    plot(t, X_d[:,0], '--', linewidth=2, label='$x_d$')
    plot(t, X_d[:,2], '--', linewidth=2, label='$\\dot{x}_d$')
    legend(fontsize=10, loc='lower right', ncol=4)
    ylim((-4.5, 2.5))
    ylabel('$x$, $\\dot{x}$')
    grid()
    subplot(3, 1, 2)
    plot(t, X[:, 1], linewidth=2, label='$\\theta$')
    plot(t, X[:, 3], linewidth=2, label='$\\dot{\\theta}$')
    plot(t, X_d[:, 1], '--', linewidth=2, label='$\\theta_d$')
    plot(t, X_d[:, 3], '--', linewidth=2, label='$\\dot{\\theta}_d$')
    legend(fontsize=10, loc='lower right', ncol=4)
    ylim((-2.25,1.25))
    ylabel('$\\theta$, $\\dot{\\theta}$')
    grid()

    subplot(3, 1, 3)
    plot(t[:-1], U[:,0], label='$u$')
    plot(t[:-1], U_nom[:,0], label='$u_{nom}$')
    legend(fontsize=10, loc='upper right', ncol=2)
    ylabel('u')
    xlabel('Time (sec)')
    grid()
    if save:
        savefig(filename)
    if display:
        show()