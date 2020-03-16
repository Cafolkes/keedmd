import matplotlib
from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title, ylim, xlabel, ylabel, fill_between, close
from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, title, savefig, tight_layout
from matplotlib.ticker import MaxNLocator
import dill

# Save closed loop data for analysis and plotting:
folder = "core/examples/results/" + '03122020_165527/'
dill_filename = folder + 'open_loop.pickle'
infile = open(dill_filename,'rb')
[t_pred, mse_keedmd, mse_edmd, mse_nom, e_keedmd, e_edmd, e_nom, e_mean_keedmd, e_mean_edmd, e_mean_nom, e_std_keedmd, e_std_edmd, e_std_nom, xs_keedmd, xs_edmd, xs_nom, xs_pred]  = dill.load(infile)
infile.close()

dill_filename = folder + 'closed_loop.pickle'
infile = open(dill_filename,'rb')
[t_pred_mpc, qd_mpc, xs_nom_mpc, xs_edmd_mpc, xs_keedmd_mpc, us_nom_mpc, us_edmd_mpc, us_keedmd_mpc, mse_mpc_nom, mse_mpc_edmd, mse_mpc_keedmd, E_nom, E_edmd, E_keedmd, cost_nom, cost_edmd, cost_keedmd] = dill.load(infile)
infile.close()

# Plot errors of different models and statistics, open loop
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

ylabels = ['$e_x$', '$e_{\\theta}$']
ax = figure(figsize=(6,5)).gca()
for ii in range(2):
    subplot(2, 1, ii+1)
    plot(t_pred, e_mean_nom[ii,:], linewidth=2, label='Nominal', color='tab:gray')
    fill_between(t_pred, e_mean_nom[ii,:]-e_std_nom[ii,:], e_mean_nom[ii,:]+e_std_nom[ii,:], alpha=0.2, color='tab:gray')

    plot(t_pred, e_mean_edmd[ii,:], linewidth=2, label='EDMD', color='tab:green')
    fill_between(t_pred, e_mean_edmd[ii,:] - e_std_edmd[ii, :], e_mean_edmd[ii,:] + e_std_edmd[ii, :], alpha=0.2, color='tab:green')

    plot(t_pred, e_mean_keedmd[ii,:], linewidth=2, label='KEEDMD',color='tab:orange')
    fill_between(t_pred, e_mean_keedmd[ii,:]- e_std_keedmd[ii, :], e_mean_keedmd[ii,:] + e_std_keedmd[ii, :], alpha=0.2,color='tab:orange')

    ylabel(ylabels[ii])
    grid()
    if ii == 0:
        title('Mean open loop prediction error (+/- 1 std)')
        legend(fontsize=10, loc='upper left')
        ylim(-2., 2.)
    else:
        ylim(-4., 4.)

xlabel('Time (sec)')
#ax.yaxis.set_major_locator(MaxNLocator(integer=True))
#matplotlib.rcParams['pdf.fonttype'] = 42
#matplotlib.rcParams['ps.fonttype'] = 42
tight_layout()
savefig('core/examples/results/openloop_error.pdf', format='pdf', dpi=2400)

# Plot the closed loop trajectory:
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

ylabels = ['$x$', '$\\theta$']
bx = figure(figsize=(6,5)).gca()
for ii in range(2):
    subplot(2, 1, ii+1)
    plot(t_pred, qd_mpc[ii,:], linestyle="--",linewidth=2, label='Reference')
    plot(t_pred, xs_nom_mpc[ii, :], linewidth=2, label='Nominal', color='tab:gray')
    plot(t_pred, xs_edmd_mpc[ii,:], linewidth=2, label='EDMD', color='tab:green')
    plot(t_pred, xs_keedmd_mpc[ii,:], linewidth=2, label='KEEDMD',color='tab:orange')
    ylabel(ylabels[ii])
    grid()
    if ii == 0:
        title('Closed loop trajectory tracking with MPC')
        legend(fontsize=10, loc='lower left')
xlabel('Time (sec)')
#bx.yaxis.set_major_locator(MaxNLocator(integer=True))
#matplotlib.rcParams['pdf.fonttype'] = 42
#matplotlib.rcParams['ps.fonttype'] = 42
tight_layout()
savefig('core/examples/results/closedloop.pdf', format='pdf', dpi=2400)
show()
