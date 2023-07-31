# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from deap.tools._hypervolume import hv
from matplotlib.pyplot import cm
from matplotlib.lines import Line2D
from statistics import mean
import math

from ._target_space import normalize


# % Clip x between xmin and xmax
def clip(x, xmin, xmax):
    for i, xx in enumerate(x):
        if xmin[i] is not None:
            if xx < xmin[i]:
                x[i] = xmin[i]
        if xmax[i] is not None:
            if xx > xmax[i]:
                x[i] = xmax[i]
    return

def set_idx_val(x, idx, val):
    x_ = x.copy()
    x_[idx] = val
    return x_

def equal_except_on_idxs(x1, x2, lidx):
    if len(x1) != len(x2):
        return False
    for i in range(len(x1)):
        if i not in lidx:
            if x1[i] != x2[i]:
                return False
    return True


# % Visualization


def plot_PF_2obj(res, title='', f1=r'$f_1$', f2=r'$f_2$', estim=True, err=False, outdir=None):
    """Plot estimated PF, observed PF and eventually selected point for evaluation for
    a problem with 2 objective functions.

    Keyword Arguments:
    res -- Checkpoint results file
    title -- Title to give to the figure
    f1 -- Name of first objective
    f2 -- Name of second objective
    estim -- Bool specifying if one want to plat the estimated Pareto front or not
    err -- Bool specifying if confidence intervals should be plotted for each point
           of the estimated PF
    outdir -- Name of the dir where to save figure. If None, figure is not saved
    """
    fig, ax = plt.subplots(figsize=(14,10))
    plt.title(title)
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    if estim and err:
        # Estimated PF with confidence bounds
        f1_model = res.models[0]
        f1err = 1.96 * f1_model.predict(normalize(res.estim_PS, res.pbounds), return_std=True)[1] #95% confidence interval
        f2_model = res.models[1]
        f2err = 1.96 * f2_model.predict(normalize(res.estim_PS, res.pbounds), return_std=True)[1]
        ax.errorbar(res.estim_PF[:, 0], res.estim_PF[:, 1], xerr=f1err, yerr=f2err, fmt="o", color='blue', label='Estimated PF', zorder=0)
    elif estim:
        ax.scatter(res.estim_PF[:, 0], res.estim_PF[:, 1], label='Estimated PF', zorder=0, color='blue')

    # Observed PF
    ax.scatter(-res.current_PF[:, 0], -res.current_PF[:, 1], label="Observed PF", color='red', zorder=1)

    # Selected point if relevant
    if estim and res.rd_next_point == False:
        ax.scatter([res.next_point_f[0]], [res.next_point_f[1]], label="Selected point", marker='*', s=60, color='lightgreen', zorder=2)
    ax.grid(color = 'lightgray', linestyle = '--', linewidth = 0.5)
    plt.legend()

    if outdir is None:
        plt.show()
    else:
        plt.savefig(outdir+f'PF_{f1}_{f2}_{title}.png', bbox_inches='tight')

def plot_PF_3obj(res, title='', f1=r'$f_1$', f2=r'$f_2$', f3=f'$f_3$', err=False, outdir=None):
    """Plot estimated PF, observed PF and eventually selected point for evaluation for
    a problem with 3 objective functions.

    Keyword Arguments:
    res -- Checkpoint results file
    title -- Title to give to the figure
    f1 -- Name of first objective
    f2 -- Name of second objective
    f3 -- Name of the third objective
    err -- Bool specifying if confidence intervals should be plotted for each point
           of the estimated PF
    outdir -- Name of the dir where to save figure. If None, figure is not saved
    """
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(projection='3d')
    plt.title(title)
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_zlabel(f3)
    if err:
        # Estimated PF with confidence bounds
        f1_model = res.models[0]
        f1err = 1.96 * f1_model.predict(normalize(res.estim_PS, res.pbounds), return_std=True)[1] #95% confidence interval
        f2_model = res.models[1]
        f2err = 1.96 * f2_model.predict(normalize(res.estim_PS, res.pbounds), return_std=True)[1]
        f3_model = res.models[2]
        f3err = 1.96 * f3_model.predict(normalize(res.estim_PS, res.pbounds), return_std=True)[1]
        ax.errorbar(res.estim_PF[:, 0], res.estim_PF[:, 1], res.estim_PF[:, 2], xerr=f1err, yerr=f2err, zerr=f3err, fmt="o", color='blue', label='Estimated PF')
    else:
        ax.scatter(res.estim_PF[:, 0], res.estim_PF[:, 1], res.estim_PF[:, 2], label='Estimated PF', zorder=0, color='blue')

    # Observed PF
    ax.scatter(-res.current_PF[:, 0], -res.current_PF[:, 1], -res.current_PF[:, 2], label="Observed PF", color='red')

    # Selected point if relevant
    if res.rd_next_point == False:
        ax.scatter([res.next_point_f[0]], [res.next_point_f[1]], [res.next_point_f[2]], label="Selected point", marker='*', s=60, color='lightgreen')

    ax.set_xlim([0,15])
    ax.set_ylim([70,0])
    ax.yaxis.labelpad=16
    ax.xaxis.labelpad=16
    ax.zaxis.labelpad=16
    plt.legend()

    if outdir is None:
        plt.show()
    else:
        plt.savefig(outdir+f'PF_{f1}_{f2}_{f3}_{title}.png', bbox_inches='tight')

def plot_observations_2obj(res, title='', f1=r'$f_1$', f2=r'$f_2$', outdir=None):
    """Plot the observations in a f2(f1) graph. Works for problems with 2 objectives.

    Keyword Arguments:
    res -- Checkpoint results file
    title -- Title to give to the figure
    f1 -- Name of first objective
    f2 -- Name of second objective
    outdir -- Name of the dir where to save figure. If None, figure is not saved
    """
    fig, ax = plt.subplots(figsize=(14,10))
    plt.title(title)
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.scatter(-res.observed_f[:-1, 0], -res.observed_f[:-1, 1], label="Observations", color='red', marker='.')
    ax.scatter(-res.observed_f[-1:, 0], -res.observed_f[-1:, 1], label="New point observed", marker='*', color='lightgreen', s=60)
    ax.grid(color = 'lightgray', linestyle = '--', linewidth = 0.5)
    plt.legend()
    if outdir is None:
        plt.show()
    else:
        plt.savefig(outdir+f'observations_{f1}_{f2}_{title}.png', bbox_inches='tight')

def plot_GP_1D(res, true_func=None, n_points=100, objectives_names= None, dim_name=None, title='', outdir=None):
    """Plot the Gaussian Processes for all objectives of the problem. Works for
    problems with a 1 dimension input space and any nb of objectives.

    Keyword Arguments:
    res -- Checkpoint results file
    true_func -- Callable corresponding to the true function. If None, true function
                 is not represented on the plot
    n_points -- Number of points used to create the plot
    objectives_names -- List of strings corresponding to the names of the objectives
    dim_name -- Name of the dimension of the problem
    outdir -- Name of the dir where to save figure. If None, figure is not saved
    """
    bounds = res.pbounds[0]
    X = np.linspace(bounds[0], bounds[1], n_points)
    X = X.reshape(-1, 1)
    Xt = normalize(X, [bounds])
    Xt = Xt.reshape(-1, 1)
    dim_name = 'x' if dim_name is None else dim_name

    if true_func is not None:
        f = [true_func(x) for x in X]
        f = np.asarray(f)
    for iobj, model in enumerate(res.models):
        title_ = fr'$f_{iobj+1}$' if objectives_names is None else objectives_names[iobj]
        fig, ax = plt.subplots(figsize=(14,10))
        plt.title(title)
        ax.set_xlabel(dim_name)
        ax.set_ylabel(title_)
        # Model
        mu, sigma = model.predict(Xt, return_std=True)
        ax.plot(X, -mu, "g--", label=fr"$\mu(x)$", zorder=2)
        ax.fill(np.concatenate([X, X[::-1]]),
                np.concatenate([-mu - 1.9600 * sigma,
                                (-mu + 1.9600 * sigma)[::-1]]),
                alpha=.2, fc="g", ec="None", zorder=0)

        # Observations
        ax.scatter(res.observed_x[:-1], -res.observed_f[:-1,iobj], color='red', label='Observations', zorder=3)

        # Next point
        ax.scatter(res.observed_x[-1:], -res.observed_f[-1:,iobj], color='red', marker='*', label='New observed point', zorder=4, s=60)

        # True func
        if true_func is not None:
            fobj = f[:,iobj]
            ax.plot(X, fobj, "k-", label=fr"$f_{iobj+1}$", zorder=1)
        ax.grid(color = 'lightgray', linestyle = '--', linewidth = 0.5)
        plt.legend()

        if outdir is None:
            plt.show()
        else:
            plt.savefig(outdir+f'GP_{title_}_{title}.png', bbox_inches='tight')

def plot_GPmu_2D(res, iobj, objective_name, dims_names, estimated_pareto_set, n_points=100, outdir=None):
    bounds1 = res.pbounds[0]
    bounds2 = res.pbounds[1]
    X1 = np.linspace(bounds1[0], bounds1[1], n_points)
    X1 = X1.reshape(-1, 1)
    X2 = np.linspace(bounds2[0], bounds2[1], n_points)
    X2 = X2.reshape(-1, 1)
    X = np.array([np.array([xe,ye]).reshape(1,-1) for ye in X2 for xe in X1]).reshape(-1, 2)
    X1_plot, X2_plot = np.meshgrid(X1, X2)
    Xt = normalize(X, res.pbounds)

    fig, ax = plt.subplots(figsize=(14,10))
    plt.xlabel(dims_names[0])
    plt.ylabel(dims_names[1])
    plt.title(objective_name)

    mu = res.models[iobj].predict(Xt)
    mu_plot = mu.reshape(X1_plot.shape)
    mu_plot = -mu_plot
    levels = np.linspace(mu_plot.min(), mu_plot.max(), 26)
    c = ax.contourf(X1_plot, X2_plot, mu_plot, levels=levels, cmap=cm.viridis)
    cbar = plt.colorbar(c, ax=ax)
    ax.scatter(res.current_PS[:,0], res.current_PS[:,1], color='red', marker='+', s=220, label="Observed PS", linewidths=3, zorder=10)
    ax.scatter(res.estim_PS[:,0], res.estim_PS[:,1], color='k', marker='+', s=60, label="Estimated PS (NSGA-II)", zorder=9)
    ax.scatter(estimated_pareto_set[:,0], estimated_pareto_set[:,1], color='silver', marker='.', s=450, label="Estimated PS (evenly sampled points)")


    lines = [Line2D([0], [0], marker='+', markersize=15, markeredgewidth=3, label='Observed PS', markeredgecolor='red',
                          markerfacecolor='red', linestyle='None'),
            Line2D([0], [0], marker='+', markersize=8, label='Estimated PS (NSGA-II)', markeredgecolor='k',
                                  markerfacecolor='k', linestyle='None'),
            Line2D([0], [0], marker='o', label='Estimated PS', markeredgecolor='silver', markersize=17,
                                  markerfacecolor='silver', linestyle='None')]
    fig.legend(lines, ['Observed PS', 'Estimated PS (NSGA-II)', 'Estimated PS (evenly sampled)'],
        fontsize=22, loc='upper right', bbox_to_anchor=(1.01, 1.04))

    if outdir is None:
        plt.show()
    else:
        plt.savefig(outdir+f'GPmu_{objective_name}.pdf', bbox_inches='tight')

def plot_GP_2D(res, true_func=None, n_points=100, objectives_names=None, objectives_names_all=None, dims_names=None, estimated_pareto_set=None, title='', outdir=None, retrn=False):
    bounds1 = res.pbounds[0]
    bounds2 = res.pbounds[1]
    X1 = np.linspace(bounds1[0], bounds1[1], n_points)
    X1 = X1.reshape(-1, 1)
    X2 = np.linspace(bounds2[0], bounds2[1], n_points)
    X2 = X2.reshape(-1, 1)
    X = np.array([np.array([xe,ye]).reshape(1,-1) for ye in X2 for xe in X1]).reshape(-1, 2)
    X1_plot, X2_plot = np.meshgrid(X1, X2)
    Xt = normalize(X, res.pbounds)

    nb_subplots_y = len(res.models)
    if true_func is not None:
        nb_subplots_x = 3
        F = [true_func(x) for x in X]
        F = np.asarray(F)
    else:
        nb_subplots_x = 2

    fig, axs = plt.subplots(nb_subplots_y,nb_subplots_x, sharex=True, sharey=True, figsize=(16,10))
    fig.suptitle(title)
    dims_names = [r'$x_1$', r'$x_2$'] if dims_names is None else dims_names
    fig.text(0.5, 0.05, dims_names[0], ha='center')
    fig.text(0.07, 0.5, dims_names[1], va='center', rotation='vertical')
    fig.subplots_adjust(top=0.88)

    for iobj, model in enumerate(res.models):
        # Mu
        mu, sigma = model.predict(Xt, return_std=True)
        mu_plot = mu.reshape(X1_plot.shape)
        mu_plot = -mu_plot
        #c = axs[iobj][0].pcolormesh(X1_plot, X2_plot, -mu_plot, cmap=cm.viridis)
        levels = np.linspace(mu_plot.min(), mu_plot.max(), 26)
        c = axs[iobj][0].contourf(X1_plot, X2_plot, mu_plot, levels=levels, cmap=cm.viridis)
        cbar = plt.colorbar(c, ax=axs[iobj][0])
        title_ = fr"$\mu_{iobj+1}$(X)" if objectives_names is None and objectives_names_all is None else fr"{objectives_names[iobj]} - $\mu$(X)" if objectives_names_all is None else rf"{objectives_names_all[0][iobj]}"
        axs[iobj][0].set_title(title_)
        axs[iobj][0].scatter(res.current_PS[:,0], res.current_PS[:,1], color='red', marker='+', s=80, label="Observed PS", zorder=10)

        # Sigma
        sigma_plot = sigma.reshape(X1_plot.shape)
        #c = axs[iobj][1].pcolormesh(X1_plot, X2_plot, sigma_plot, cmap=cm.viridis)
        levels = np.linspace(sigma_plot.min(), sigma_plot.max(), 26)
        c = axs[iobj][1].contourf(X1_plot, X2_plot, sigma_plot, levels=levels, cmap=cm.viridis)
        cbar = plt.colorbar(c, ax=axs[iobj][1])
        title_ = fr"$\sigma_{iobj+1}$(X)" if objectives_names is None and objectives_names_all is None else fr"{objectives_names[iobj]} - $\sigma$(X)" if objectives_names_all is None else rf"{objectives_names_all[1][iobj]}"
        axs[iobj][1].set_title(title_)
        axs[iobj][1].scatter(res.observed_x[:,0], res.observed_x[:,1], marker='.', color='red', label="Observed points")

        # True func
        if true_func is not None:
            f_plot = F[:,iobj]
            f_plot = f_plot.reshape(X1_plot.shape)
            c = axs[iobj][2].pcolormesh(X1_plot, X2_plot, f_plot, cmap=cm.viridis)
            cbar = plt.colorbar(c, ax=axs[iobj][2])
            title_ = fr"$f_{iobj+1}$(X)" if objectives_names is None else f"{objectives_names[iobj]}(X)"
            axs[iobj][2].set_title(title_)

        # Estimated Pareto set
        if estimated_pareto_set is not None:
            axs[iobj][0].scatter(estimated_pareto_set[:,0], estimated_pareto_set[:,1], color='silver', marker='.', s=36, label="Estimated PS")


    lines = [Line2D([0], [0], marker='+', markersize=10, label='Observed PS', markeredgecolor='red',
                          markerfacecolor='red', linestyle='None'),
            Line2D([0], [0], marker='o', label='Estimated PS', markeredgecolor='silver', markersize=8,
                                  markerfacecolor='silver', linestyle='None'),
            Line2D([0], [0], marker='o', label='Observations', markeredgecolor='red',
                          markerfacecolor='red', linestyle='None')]
    fig.legend(lines, ['Observed PS', 'Estimated PS', 'Observations'], fontsize=22, loc='lower right', bbox_to_anchor=(0.95, 0.05))

    if outdir is None and retrn == False:
        plt.show()
    elif outdir is not None:
        plt.savefig(outdir+f'GP_{title}.png', bbox_inches='tight')
    else:
        return axs

def plot_GP_nD(res, dim1, dim2, default_x, true_func=None, n_points=100, objectives_names=None, dims_names=None, title='', outdir=None):
    bounds1 = res.pbounds[dim1]
    bounds2 = res.pbounds[dim2]
    X1 = np.linspace(bounds1[0], bounds1[1], n_points)
    X2 = np.linspace(bounds2[0], bounds2[1], n_points)
    X = np.array([np.array(set_idx_val(set_idx_val(default_x, dim1, x1), dim2, x2)).reshape(1,-1) for x2 in X2 for x1 in X1]).reshape(-1, len(default_x))
    Xt = normalize(X, res.pbounds)
    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(-1, 1)
    X1_plot, X2_plot = np.meshgrid(X1, X2)

    nb_subplots_y = len(res.models)
    if true_func is not None:
        nb_subplots_x = 3
        F = [true_func(x) for x in X]
        F = np.asarray(F)
    else:
        nb_subplots_x = 2

    fig, axs = plt.subplots(nb_subplots_y,nb_subplots_x, sharex=True, sharey=True, figsize=(16,10))
    fig.suptitle(title)
    dims_names = [fr'$x_{dim1}$', fr'$x_{dim2}$'] if dims_names is None else [dims_names[dim1], dims_names[dim2]]
    fig.text(0.5, 0.05, dims_names[0], ha='center')
    fig.text(0.07, 0.5, dims_names[1], va='center', rotation='vertical')
    fig.subplots_adjust(top=0.88)

    for iobj, model in enumerate(res.models):
        # Mu
        mu, sigma = model.predict(Xt, return_std=True)
        mu_plot = mu.reshape(X1_plot.shape)
        #c = axs[iobj][0].pcolormesh(X1_plot, X2_plot, -mu_plot, cmap=cm.viridis)
        minus_mu_plot = -mu_plot
        levels = np.linspace(minus_mu_plot.min(), minus_mu_plot.max(), 30)
        c = axs[iobj][0].contourf(X1_plot, X2_plot, minus_mu_plot, levels=levels, cmap=cm.viridis)
        cbar = plt.colorbar(c, ax=axs[iobj][0])
        title_ = fr"$\mu_{iobj+1}$(X)" if objectives_names is None else fr"{objectives_names[iobj]} - $\mu$(X)"
        axs[iobj][0].set_title(title_)
        curr_ps_at_default_x_1 = np.array([e[dim1] for e in res.current_PS if equal_except_on_idxs(e, default_x, [dim1, dim2])])
        curr_ps_at_default_x_2 = np.array([e[dim2] for e in res.current_PS if equal_except_on_idxs(e, default_x, [dim1, dim2])])
        axs[iobj][0].plot(curr_ps_at_default_x_1, curr_ps_at_default_x_2, "r.", label="Observed PF")

        # Sigma
        sigma_plot = sigma.reshape(X1_plot.shape)
        #c = axs[iobj][1].pcolormesh(X1_plot, X2_plot, sigma_plot, cmap=cm.viridis)
        levels = np.linspace(sigma_plot.min(), sigma_plot.max(), 30)
        c = axs[iobj][1].contourf(X1_plot, X2_plot, sigma_plot, levels=levels, cmap=cm.viridis)
        cbar = plt.colorbar(c, ax=axs[iobj][1])
        title_ = fr"$\sigma_{iobj+1}$(X)" if objectives_names is None else fr"{objectives_names[iobj]} - $\sigma$(X)"
        axs[iobj][1].set_title(title_)
        obs_at_default_x_1 = np.array([e[dim1] for e in res.observed_x if equal_except_on_idxs(e, default_x, [dim1, dim2])])
        obs_at_default_x_2 = np.array([e[dim2] for e in res.observed_x if equal_except_on_idxs(e, default_x, [dim1, dim2])])
        axs[iobj][1].scatter(obs_at_default_x_1, obs_at_default_x_2, marker='.', color='cyan', label="Observed points")

        # True func
        if true_func is not None:
            f_plot = F[:,iobj]
            f_plot = f_plot.reshape(X1_plot.shape)
            c = axs[iobj][2].pcolormesh(X1_plot, X2_plot, f_plot, cmap=cm.viridis)
            cbar = plt.colorbar(c, ax=axs[iobj][2])
            title_ = fr"$f_{iobj+1}$(X)" if objectives_names is None else f"{objectives_names[iobj]}(X)"
            axs[iobj][2].set_title(title_)

    lines = [Line2D([0], [0], marker='o', label='Observed PF', markeredgecolor='red',
                          markerfacecolor='red', linestyle='None'),
            Line2D([0], [0], marker='o', label='Observations', markeredgecolor='cyan',
                          markerfacecolor='cyan', linestyle='None')]
    fig.legend(lines, ['Observed PF', 'Observations'], fontsize=22, loc='lower right', bbox_to_anchor=(0.9, 0.05))

    if outdir is None:
        plt.show()
    else:
        plt.savefig(outdir+f'GP_{dim1}_{dim2}_{default_x}_{title}.png', bbox_inches='tight')

def plot_nds_hv_convergence(res, n_init_points=0, outdir=None):
    final_PF = -res[-1].current_PF
    all_obs_f = -res[-1].observed_f
    n_obj = len(res[0].current_PF[0])

    ## NDS
    C = [np.nan] * n_init_points
    for r in res:
        c = 0
        for obs in r.current_PF:
            if -obs in final_PF:
                c += 1
        C.append(c)

    ## HV
    ref = [max(all_obs_f[:,obji]) for obji in range(n_obj)]
    ref_min = [min(all_obs_f[:,obji]) for obji in range(n_obj)]
    HVs = [np.nan] * n_init_points
    for r in res:
        # Normalize based on ref point
        normalized_current_PF = normalize(-r.current_PF, np.array([[ref_min[iobj],ref[iobj]] for iobj in range(n_obj)]))
        hvol = hv.hypervolume(normalized_current_PF, [1. for iobj in range(n_obj)])
        HVs.append(hvol)
    print(HVs)
    fig, ax1 = plt.subplots(figsize=(14,10))
    ax1.set_xlabel(r'$q$ (Number of simulations)')
    ax1.set_ylabel('Hypervolume (HV)')
    lns1 = ax1.plot(range(1,len(HVs)+1), HVs, marker='.', color='black', label='HV')
    ax1.grid(color = 'lightgray', linestyle = '--', linewidth = 0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Number of non-dominated solutions (NDS)')
    lns2 = ax2.plot(range(1,len(C)+1), C, marker='.', color='silver', label='NDS')

    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    if outdir is None:
        plt.show()
    else:
        plt.savefig(outdir+f'convergence_NDS_HV.png', bbox_inches='tight')

def plot_convergence(res, outdir=None):
    final_PF = -res[-1].current_PF
    all_obs_f = -res[-1].observed_f
    n_obj = len(res[0].current_PF[0])

    ## Nb of points of final PF found per iter
    C = []
    for r in res:
        c = 0
        for obs in r.current_PF:
            if -obs in final_PF:
                c += 1
        C.append(c)
    fig, ax = plt.subplots(figsize=(14,10))
    ax.set_xlabel(r'$q$ (Number of simulations)')
    #ax.set_ylabel('Part of final Pareto set found (%)')
    ax.set_ylabel('Number of non-dominated solutions (NDS)')
    #ax.plot(range(1,len(C)+1), [100*c/C[-1] for c in C], marker='.', color='black')
    ax.plot(range(1,len(C)+1), C, marker='.', color='black')
    ax.grid(color = 'lightgray', linestyle = '--', linewidth = 0.5)

    if outdir is None:
        plt.show()
    else:
        #plt.savefig(outdir+f'convergence_percentage_points_found.png', bbox_inches='tight')
        plt.savefig(outdir+f'convergence_nds.png', bbox_inches='tight')

    ## Nb of points NDS points found per iter
    C = []
    for r in res:
        C.append(len(r.current_PF))
    fig, ax = plt.subplots(figsize=(14,10))
    ax.set_xlabel(r'$q$ (Number of simulations)')
    #ax.set_ylabel('Part of final Pareto set found (%)')
    ax.set_ylabel('Number of non-dominated solutions (NDS)')
    #ax.plot(range(1,len(C)+1), [100*c/C[-1] for c in C], marker='.', color='black')
    ax.plot(range(1,len(C)+1), C, marker='.', color='black')
    ax.grid(color = 'lightgray', linestyle = '--', linewidth = 0.5)

    if outdir is None:
        plt.show()
    else:
        #plt.savefig(outdir+f'convergence_percentage_points_found.png', bbox_inches='tight')
        plt.savefig(outdir+f'convergence_nds_bis.png', bbox_inches='tight')

    ## Hypervolume
    ref = [max(max(all_obs_f[:,obji]),max([max(r.estim_PF[:,obji]) for r in res])) for obji in range(n_obj)]
    ref_min = [min(min(all_obs_f[:,obji]),min([min(r.estim_PF[:,obji]) for r in res])) for obji in range(n_obj)]
    HVs = []
    eHVs = []
    for r in res:
        # Normalize based on ref point
        normalized_current_PF = normalize(-r.current_PF, np.array([[ref_min[iobj],ref[iobj]] for iobj in range(n_obj)]))
        normalized_estim_PF = normalize(r.estim_PF, np.array([[ref_min[iobj],ref[iobj]] for iobj in range(n_obj)]))
        hvol = hv.hypervolume(normalized_current_PF, [1. for iobj in range(n_obj)])
        ehvol = hv.hypervolume(normalized_estim_PF, [1. for iobj in range(n_obj)])
        HVs.append(hvol)
        eHVs.append(ehvol)
    fig, ax = plt.subplots(figsize=(14,10))
    ax.set_xlabel(r'$q$ (Number of simulations)')
    ax.set_ylabel('Hypervolume (HV)')
    ax.plot(range(1,len(HVs)+1), HVs, marker='.', color='black', label=r'HV($\Phi^{(q)}$)')
    ax.plot(range(1,len(eHVs)+1), eHVs, marker='.', color='grey', label=r'HV($\hat{\Phi}^{(q)}$)')
    ax.grid(color = 'lightgray', linestyle = '--', linewidth = 0.5)
    plt.legend()

    if outdir is None:
        plt.show()
    else:
        plt.savefig(outdir+f'convergence_HV.png', bbox_inches='tight')

    ## Diff between hypervolume of estim_PF and current_PF
    dHVs = []
    dHVs_ = []
    for r in res:
        ref = [max(max(-r.current_PF[:,obji]),max(r.estim_PF[:,obji])) for obji in range(n_obj)]
        ref_min = [min(min(-r.current_PF[:,obji]),min(r.estim_PF[:,obji])) for obji in range(n_obj)]
        normalized_current_PF = normalize(-r.current_PF, np.array([[ref_min[iobj],ref[iobj]] for iobj in range(n_obj)]))
        normalized_estim_PF = normalize(r.estim_PF, np.array([[ref_min[iobj],ref[iobj]] for iobj in range(n_obj)]))
        hvol = hv.hypervolume(normalized_current_PF, [1. for iobj in range(n_obj)])
        ehvol = hv.hypervolume(normalized_estim_PF, [1. for iobj in range(n_obj)])
        dHVs.append(ehvol - hvol)
        dHVs_.append(math.exp(-2.5*abs(ehvol - hvol)))
    fig, ax = plt.subplots(figsize=(14,10))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Relative difference between observed and estimated Hypervolumes (rdHV)')
    ax.plot(range(1,len(dHVs)+1), dHVs, marker='.', color='black')
    ax.grid(color = 'lightgray', linestyle = '--', linewidth = 0.5)

    fig, ax = plt.subplots(figsize=(14,10))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('W3')
    ax.plot(range(1,len(dHVs)+1), dHVs_, marker='.', color='black')
    ax.grid(color = 'lightgray', linestyle = '--', linewidth = 0.5)

    if outdir is None:
        plt.show()
    else:
        plt.savefig(outdir+f'convergence_rdHV.png', bbox_inches='tight')

    ## Diversity
    # Find the extreme solutions of the last iteration
    lexsort_keys = []
    for iobj in range(n_obj):
        lexsort_keys.append(final_PF[:,iobj])
    lexsort_keys = tuple(lexsort_keys)
    ind = np.lexsort(lexsort_keys)
    sorted_final_PF = final_PF[ind]
    final_PF_up = sorted_final_PF[0]
    final_PF_low = sorted_final_PF[-1]
    # Find the extreme solutions from all sampled points
    lexsort_keys = []
    for iobj in range(n_obj):
        lexsort_keys.append(all_obs_f[:,iobj])
    lexsort_keys = tuple(lexsort_keys)
    ind = np.lexsort(lexsort_keys)
    sorted_all_obs_f = all_obs_f[ind]
    all_obs_f_up = sorted_all_obs_f[0]
    all_obs_f_low = sorted_all_obs_f[-1]

    diversities = []
    for r in res:
        # Sort Pareto set
        lexsort_keys = []
        for iobj in range(n_obj):
            lexsort_keys.append(r.current_PF[:,iobj])
        lexsort_keys = tuple(lexsort_keys)
        ind = np.lexsort(lexsort_keys)
        sorted_pf = r.current_PF[ind]
        # Deduce the distances between consecutive points
        consec_dists = []
        for i in range(len(sorted_pf)-1):
            if i == 0:
                df = np.linalg.norm(sorted_pf[i] - all_obs_f_up)
            if i == len(sorted_pf) - 2:
                dl = np.linalg.norm(sorted_pf[i+1] - all_obs_f_low)
            consec_dists.append(np.linalg.norm(sorted_pf[i+1] - sorted_pf[i]))
        mean_d = mean(consec_dists)
        diversity = (df + dl + sum([abs(d - mean_d) for d in consec_dists])) / (df + dl + len(consec_dists)*mean_d)
        diversities.append(diversity)
    fig, ax = plt.subplots(figsize=(14,10))
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'Diversity ($\Delta$)')
    ax.plot(range(1,len(diversities)+1), diversities, marker='.', color='black')
    ax.grid(color = 'lightgray', linestyle = '--', linewidth = 0.5)

    if outdir is None:
        plt.show()
    else:
        plt.savefig(outdir+f'convergence_diversity.png', bbox_inches='tight')
