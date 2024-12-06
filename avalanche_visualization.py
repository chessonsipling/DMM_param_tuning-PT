import os
import json
import pickle
import numpy as np
from scipy.stats import linregress
import torch
import matplotlib
import plt_config
import matplotlib.pyplot as plt
color = plt.rcParams['axes.prop_cycle'].by_key()['color']
matplotlib.use('Agg')  # much faster but no interactive sessions

folder = 'results'
graph_dir = 'graphs'
os.makedirs(graph_dir, exist_ok=True)
ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
time_windows = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

for time_window in time_windows:
    xs = []
    ys = []
    fig, ax = plt.subplots()
    for i, n in enumerate(ns):
        cluster_sizes = pickle.load(open(f'{folder}/cluster_size_{n}_{time_window}.pkl', 'rb'))
        cluster_sizes = cluster_sizes[cluster_sizes > 0]

        log_cluster_size = np.log10(cluster_sizes)
        log_cluster_size = log_cluster_size[log_cluster_size >= 0]
        # quartiles = np.percentile(log_cluster_size, [25, 50, 75, 100])
        # IQR = quartiles[2] - quartiles[0]
        # bin_width = 2 * IQR * (len(log_cluster_size) ** (-1 / 3))
        mean = np.mean(log_cluster_size)
        std = np.std(log_cluster_size)
        max_size = np.max(log_cluster_size)
        bin_width = 3 * std / (len(log_cluster_size) ** (1 / 3))
        bin_width = max(bin_width, 0.02)
        n_bins = int(6 / bin_width)  # assuming all avalanches smaller than 10^6
        n_bins = max(n_bins, 1)
        bins = bin_width * np.arange(n_bins + 1)
        hist, bin_edges = np.histogram(log_cluster_size, bins=bins)
        bin_edges_linear = 10 ** bin_edges
        bin_sizes_linear = np.diff(bin_edges_linear)
        hist = hist / bin_sizes_linear
        hist = hist / hist.sum()
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_centers = bin_centers[hist > 0]
        hist = hist[hist > 0]
        slope, intercept, r, p, se = linregress(bin_centers, np.log10(hist))
        # avalanche_data.append([slope, intercept, r] + quartiles.tolist())
        # avalanche_data = np.array([slope, intercept, r, mean, std, max_size, len(cluster_sizes)])

        ax.scatter(10 ** bin_centers, hist, s=10, alpha=0.5, color=color[i],
                label=f'N = {n}')
        if n == 640:
            ax.plot(10 ** bin_centers, 10 ** (slope * bin_centers + intercept), '--', color='k')
        xs.append(10 ** bin_centers)
        ys.append(hist)
    ax.text(1, 10**-3, f'$\sim s^{{{slope:.2f}}}$', fontsize=18)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Avalanche size s')
    ax.set_ylabel('Probability P(s)')
    ax.legend(fontsize=14)
    plt.title(f'Time window = {time_window}')
    plt.savefig(f'graphs/time_window_{time_window:.2f}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    for i, x, y, n in zip(range(len(xs)), xs, ys, ns):
        ax.scatter(x / n, y * x ** -slope, s=10, alpha=0.5, color=color[i],
                label=f'N = {n}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$s/N$')
    ax.set_ylabel(f'$s^{{{-slope:.2f}}} P(s)$')
    ax.legend(fontsize=14)
    plt.title(f'Time window = {time_window}')
    plt.savefig(f'graphs/time_window_{time_window:.2f}_scaled.png',
                dpi=300, bbox_inches='tight')
