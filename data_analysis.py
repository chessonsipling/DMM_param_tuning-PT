import os
import json
import pickle
import numpy as np
import torch
import matplotlib
import plt_config
import matplotlib.pyplot as plt
color = plt.rcParams['axes.prop_cycle'].by_key()['color']
matplotlib.use('Agg')  # much faster but no interactive sessions

import sys

eqn_choice = 'diventra_choice' #sys.argv[1] #eqn_choice can ONLY take on the values 'sean_choice', 'diventra_choice', 'yuanhang_choice', and 'zeta_zero' (and 'R_zero', 'rudy_choice', or 'rudy_simple' for XORSAT)
prob_type = '3R3X' #sys.argv[1] #prob_type can ONLY take on the values '3SAT', '3R3X', OR '5R5X'

ns = np.array([40, 50, 60])

folder = f'results/{prob_type}'
graph_dir = f'graphs_colossus/{prob_type}'
os.makedirs(graph_dir, exist_ok=True)
data = []
f = open(f'{folder}/stats_{ns}.p', 'rb')
while True:
    try:
        data_i = pickle.load(f)
        data.append(data_i)
    except EOFError:
        break
    except Exception as e:
        print(e)
        break
f.close()

data_size = len(data)
n, dmm_params, unsat_moments, avalanche_stats, metrics = \
    zip(*[[d['n'], d['param'], d['unsat_moments'], d['avalanche_stats'],
           d['metric']] for d in data])
ns = n[0]
dmm_params = np.array(dmm_params, dtype=object)

metrics = np.array(metrics)
metrics[np.isnan(metrics)] = 0
best_metric = np.min(metrics)
best_metric_idx = np.argmin(metrics)
best_mask = metrics <= np.percentile(metrics, 10)
optimal_param = dmm_params[best_metric_idx]
#eqn_choice = optimal_param['eqn_choice']

if eqn_choice == 'rudy_simple':
    params = ['alpha', 'delta', 'chi', 'zeta']
    param_is_log = [True, False, True, True]
elif eqn_choice == 'rudy_choice':
    params = ['alpha_by_beta', 'beta', 'delta', 'chi', 'zeta']
    param_is_log = [True, True, False, True, True]
elif eqn_choice == 'zeta_zero' or eqn_choice == 'R_zero':
    params = ['alpha_by_beta', 'beta',  'gamma', 'delta_by_gamma']
    param_is_log = [False, True, False, False, False]
else: #sean_choice, diventra_choice, and yuanhang_choice
    params = ['alpha_by_beta', 'beta', 'gamma', 'delta_by_gamma', 'zeta']
    param_is_log = [False, True, False, False, True]
relevant_stats = ['unsat_mean', 'unsat_std', 'unsat_skewness', 'unsat_kurtosis', 'slope', 'intercept', 'r', 'avl_max']
n_param = len(params)
optim_param = np.zeros(n_param)

with open(f'{graph_dir}/optimal_param_{ns}.json', 'w') as f:
    json.dump(optimal_param, f)

'''unsat_moments = np.stack(unsat_moments, axis=0)
unsat_moments[np.isnan(unsat_moments)] = 0
unsat_moments[np.isinf(unsat_moments)] = 0
unsat_moments_mean = np.mean(unsat_moments, axis=2)
unsat_moments_std = np.std(unsat_moments, axis=2)

avalanche_stats = np.stack(avalanche_stats, axis=0)
avalanche_stats[np.isnan(avalanche_stats)] = 0

relevant_stats_mean = np.zeros((len(ns), len(relevant_stats)))
relevant_stats_std = np.zeros((len(ns), len(relevant_stats)))

for n_idx, ni in enumerate(ns):
    # unsolved_stats = unsolved_stats.reshape(-1, n_epochs, 5) [:, 0, :]
    # avalanche_stats = avalanche_stats.reshape(-1, 7)
    unsat_mean, unsat_std, unsat_skewness, unsat_kurtosis = unsat_moments_mean[:, n_idx].transpose()
    unsat_mean_std, unsat_std_std, unsat_skewness_std, unsat_kurtosis_std = unsat_moments_std[:, n_idx].transpose()
    slope, intercept, r, avl_max = avalanche_stats[:, n_idx].transpose()

    for i, stat_i in enumerate(relevant_stats):
        stat = eval(stat_i)
        try:
            stat_std = eval(stat_i + '_std')
        except NameError:
            stat_std = None
        fig, ax = plt.subplots()
        if stat_std is not None:
            sc = ax.errorbar(stat, metrics, xerr=stat_std, fmt='o', c=color[0], ms=5)
        else:
            sc = ax.scatter(stat, metrics, c=color[0], s=10)
        ax.set_xlabel(stat_i)
        ax.set_ylabel('Metric')
        # ax.set_yscale('log')
        ax.set_title(f'{stat_i}')
        plt.savefig(f'{graph_dir}/Stat_{stat_i}_{ni}.png', dpi=300, bbox_inches='tight')
        plt.close()

        best_stat = stat[best_mask]
        # IQR = np.percentile(best_stat, 75) - np.percentile(best_stat, 25)
        # bin_size = 2 * IQR / (len(best_stat) ** (1 / 3))
        std = np.std(best_stat)
        bin_size = 3.5 * std / (len(best_stat) ** (1 / 3))
        if bin_size == 0:
            bin_size = 1
        n_bins = int((best_stat.max() - best_stat.min()) / bin_size)
        if n_bins == 0:
            n_bins = 1
        count, bin_edges = np.histogram(best_stat, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        fig, ax = plt.subplots()
        ax.hist(best_stat, bins=n_bins, color=color[0])
        ax.set_xlabel(stat_i)
        ax.set_ylabel('count')
        ax.set_title(f'{stat_i} distribution for top 10% metric')
        plt.savefig(f'{graph_dir}/Stat_{stat_i}_distribution_{ni}.png', dpi=300, bbox_inches='tight')
        plt.close()

        relevant_stats_mean[n_idx, i] = np.mean(best_stat)
        relevant_stats_std[n_idx, i] = np.std(best_stat)

for i, param in enumerate(params):
    param_i = np.array([p[param] for p in dmm_params])
    fig, ax = plt.subplots()
    ax.scatter(param_i, metrics, c=color[0], s=5)
    ax.set_xlabel(param)
    ax.set_ylabel('Metric')
    if param_is_log[i]:
        ax.set_xscale('log')
    ax.set_title(f'{param}')
    plt.savefig(f'{graph_dir}/Param_{param}_vs_metric.png', dpi=300, bbox_inches='tight')
    plt.close()

    best_param = param_i[best_mask]
    if param_is_log[i]:
        best_param = np.log10(best_param)
    IQR = np.percentile(best_param, 75) - np.percentile(best_param, 25)
    bin_size = 2 * IQR / (len(best_param) ** (1 / 3))
    n_bins = int((best_param.max() - best_param.min()) / bin_size)
    n_bins = max(n_bins, 1)
    count, bin_edges = np.histogram(best_param, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, ax = plt.subplots()
    ax.hist(best_param, bins=n_bins, color=color[0])
    ax.set_xlabel(f'log10{param}' if param_is_log[i] else param)
    ax.set_ylabel('count')
    ax.set_title(f'{param} distribution for top 10% metric')
    plt.savefig(f'{graph_dir}/Param_{param}_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


for i, relevant_stat in enumerate(relevant_stats):
    fit = np.polyfit(np.log(ns), np.log(relevant_stats_mean[:, i]), 1)
    fig, ax = plt.subplots()
    ax.plot(ns, relevant_stats_mean[:, i], label=relevant_stat, marker='o')
    ax.fill_between(ns, relevant_stats_mean[:, i] - relevant_stats_std[:, i],
                    relevant_stats_mean[:, i] + relevant_stats_std[:, i], alpha=0.5)
    ax.plot(ns, np.exp(fit[1]) * ns ** fit[0], 'r--', label=f'$\sim${np.exp(fit[1]):.2f}x^{fit[0]:.2f}')
    ax.set_xlabel('n')
    ax.set_ylabel(f'{relevant_stat}')
    ax.set_title(f'{relevant_stat} vs n')
    ax.legend()
    plt.savefig(f'{graph_dir}/Stat_{relevant_stat}_vs_n.png', dpi=300, bbox_inches='tight')
    plt.close()'''
