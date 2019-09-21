import os
import dill
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title, ylim, xlabel, ylabel, \
    fill_between, savefig, tight_layout
import numpy as np

# Plotting parameters
folder = 'core/examples/results/09212019_005154/'
open_loop = 'open_loop.pickle'
closed_loop = 'closed_loop.pickle'
figname_ol = 'openloop_error.pdf'
figname_cl = 'closedloop.pdf'
display_plots = True
plot_open_loop = False
plot_closed_loop = True
n = 4

#Import data and aggregate
infile = open(folder + open_loop, 'rb')
[t_pred, mse_keedmd, mse_edmd, mse_nom, e_keedmd, e_edmd, e_nom, e_mean_keedmd, e_mean_edmd, e_mean_nom, e_std_keedmd, e_std_edmd, e_std_nom] = dill.load(infile)
infile.close()
infile = open(folder + closed_loop, 'rb')
[t_pred, q_d_pred, xs_lin_MPC, xs_edmd_MPC, xs_keedmd_MPC, us_lin_MPC, us_edmd_MPC, us_keedmd_MPC, mse_mpc_nom, mse_mpc_edmd, mse_mpc_keedmd, E_nom, E_edmd, E_keedmd, cost_nom, cost_edmd, cost_keedmd] = dill.load(infile)
infile.close()

# Plot tracking error in both states and control effort for each episode of all experiments
if plot_open_loop:
    ylabels = ['x', '$\\theta$', '$\\dot{x}$', '$\\dot{\\theta}$']
    figure(figsize=(5.8, 5.5))
    for ii in range(int(n/2)):
        subplot(2, 1, ii + 1)
        plot(t_pred, np.abs(e_mean_nom[ii, :]), linewidth=2, label='Mean, Nominal', color='tab:gray')
        #fill_between(t_pred, np.zeros_like(e_mean_nom[ii, :]), e_std_nom[ii, :], alpha=0.2, color='tab:gray')
        plot(t_pred, e_std_nom[ii, :], linewidth=1, linestyle='--', label='Std, Nominal', color='tab:gray')

        plot(t_pred, np.abs(e_mean_edmd[ii, :]), linewidth=2, label='Mean, EDMD', color='tab:green')
        #fill_between(t_pred, np.zeros_like(e_mean_edmd[ii, :]), e_std_edmd[ii, :], alpha=0.2, color='tab:green')
        plot(t_pred, e_std_edmd[ii, :], linewidth=1, linestyle='--', label='Std, EDMD', color='tab:green')

        plot(t_pred, np.abs(e_mean_keedmd[ii, :]), linewidth=2, label='Mean, KEEDMD', color='tab:orange')
        #fill_between(t_pred, np.zeros_like(e_mean_keedmd[ii, :]), e_std_keedmd[ii, :], alpha=0.2, color='tab:orange')
        plot(t_pred, e_std_keedmd[ii, :], linewidth=1, linestyle='--', label='Std, KEEDMD', color='tab:orange')

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
    savefig(folder + figname_ol, format='pdf', dpi=2400)
    show()
    # close()

if plot_closed_loop:
    ylabels = ['$x$', '$\\theta$', '$\\dot{x}$', '$\\dot{\\theta}$']
    figure(figsize=(5.5, 10))
    for ii in range(n):
        subplot(n + 1, 1, ii + 1)
        plot(t_pred, q_d_pred[ii, :], linestyle="--", linewidth=2, label='Reference')
        plot(t_pred, xs_lin_MPC[ii, :], linewidth=2, label='Nominal', color='tab:gray')
        plot(t_pred, xs_edmd_MPC[ii, :], linewidth=2, label='EDMD', color='tab:green')
        plot(t_pred, xs_keedmd_MPC[ii, :], linewidth=2, label='KEEDMD', color='tab:orange')
        #xlabel('Time (s)')
        ylabel(ylabels[ii])
        grid()
        if ii == 0:
            title('Trajectory Tracking with MPC')
    legend(fontsize=10, loc='best')
    subplot(n + 1, 1, n + 1)
    plot(t_pred[:-1], us_lin_MPC[0, :], linewidth=2, label='Nominal', color='tab:gray')
    plot(t_pred[:-1], us_edmd_MPC[0, :], linewidth=2, label='EDMD', color='tab:green')
    plot(t_pred[:-1], us_keedmd_MPC[0, :], linewidth=2, label='KEEDMD', color='tab:orange')
    xlabel('Time (s)')
    ylabel('u')
    grid()
    tight_layout()
    savefig(folder + figname_cl, format='pdf', dpi=2400)
    show()

    print('Tracking error (MSE), Nominal: ', mse_mpc_nom, ', EDMD: ', mse_mpc_edmd, 'KEEDMD: ', mse_mpc_keedmd)
    print('Control effort (norm), Nominal:  ', E_nom, ', EDMD: ', E_edmd, ', KEEDMD: ', E_keedmd)
    print('MPC cost, Nominal: ', cost_nom, ', EDMD: ', cost_edmd, ', KEEDMD: ', cost_keedmd)
    print('MPC cost improvement, EDMD: ', (cost_edmd / cost_nom - 1) * 100, '%, KEEDMD: ',
          (cost_keedmd / cost_nom - 1) * 100, '%')