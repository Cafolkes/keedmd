
import matplotlib.pyplot as plt

def plot_state(X, t):
    """ Plots the states for cart pendulum

    # Inputs:
    - state X, numpy 2d array [n,N+1] 
    - desired state X_d, numpy 2d array [n,N+1]
    - control input U, numpy 2d array [number of time steps, number of inputs]
    - nominal control input U_nom, numpy 2d array [number of time steps, number of inputs]
    - time t, numpy 1d array [number of time steps 'N']
    """

    ylabels = ['$x$', '$\\theta$', '$\\dot{x}$', '$\\dot{\\theta}$']
    n = X.shape[0]
    plt.figure()
    for ii in range(n):
        plt.subplot(n, 1, ii+1)
        plt.plot(t, X[ii,:], linestyle="--",linewidth=2, label='reference')
        plt.xlabel('Time (s)')
        plt.ylabel(ylabels[ii])
        plt.grid()
        if ii == 0:
            plt.title('Closed loop performance of different models with open loop control')
    plt.legend(fontsize=10, loc='best')
    plt.show()