from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title

def plot_trajectory(X, X_d, U, U_nom, t):
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
    show()