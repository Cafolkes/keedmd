import os
import dill
from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title, ylim, xlabel, ylabel, \
    fill_between, savefig, tight_layout
import numpy as np


# Plotting parameters
folder = 'core/examples/results/09192019_122146/'
filename = 'error_data.pickle'
figname = 'openloop_error.pdf'
display_plots = True
plot_open_loop = True
plot_closed_loop = False
n = 4

#Import data and aggregate
infile = open(folder + filename, 'rb')
[t_pred, mse_keedmd, mse_edmd, mse_nom, e_keedmd, e_edmd, e_nom, e_mean_keedmd, e_mean_edmd, e_mean_nom, e_std_keedmd, e_std_edmd, e_std_nom] = dill.load(infile)
infile.close()

# Plot tracking error in both states and control effort for each episode of all experiments
if plot_open_loop:
    ylabels = ['x', '$\\theta$', '$\\dot{x}$', '$\\dot{\\theta}$']
    figure(figsize=(5.8, 5.5))
    for ii in range(int(n/2)):
        subplot(2, 1, ii + 1)
        plot(t_pred, np.abs(e_mean_nom[ii, :]), linewidth=2, label='Nominal$')
        fill_between(t_pred, np.zeros_like(e_mean_nom[ii, :]), e_std_nom[ii, :], alpha=0.2)

        plot(t_pred, np.abs(e_mean_edmd[ii, :]), linewidth=2, label='$EDMD$')
        fill_between(t_pred, np.zeros_like(e_mean_edmd[ii, :]), e_std_edmd[ii, :], alpha=0.2)

        plot(t_pred, np.abs(e_mean_keedmd[ii, :]), linewidth=2, label='$KEEDMD$')
        fill_between(t_pred, np.zeros_like(e_mean_keedmd[ii, :]), e_std_keedmd[ii, :], alpha=0.2)

        ylabel(str(ylabels[ii]))

        if ii == 1 or ii == 3:
            ylim(0., .5)
        else:
            ylim(0., .5)

        grid()
        if ii == 0:
            title('Mean Absolute Open Loop State Prediction Error')
            legend(fontsize=10, loc='upper left')
        if ii == 1:
            xlabel('Time (sec)')
    tight_layout()
    savefig(folder + figname, format='pdf', dpi=2400)
    show()
    # close()