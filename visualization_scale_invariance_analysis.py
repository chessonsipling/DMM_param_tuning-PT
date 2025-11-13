import os
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import plt_config
import torch.multiprocessing as mp
from model import DMM
from dmm_utils import run_dmm, avalanche_analysis_mp


#Plots avalanche size distributions for a particular collection of instances
def avalanche_size_distribution(n_list, list_of_cluster_sizes, name):
    list_of_hist = []
    list_of_bin_centers = []

    fig, ax = plt.subplots(figsize=(3.0, 1.75))

    for i, cluster_sizes in enumerate(list_of_cluster_sizes):
        cluster_sizes = cluster_sizes[cluster_sizes > 0]
        log_cluster_size = np.log10(cluster_sizes)
        log_cluster_size = log_cluster_size[log_cluster_size >= 0]

        mean = np.mean(log_cluster_size)
        std = np.std(log_cluster_size)
        max_size = np.max(log_cluster_size)

        bin_width = 3.5 * std / (len(log_cluster_size) ** (1 / 3))
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

        list_of_hist.append(hist)
        list_of_bin_centers.append(bin_centers)

        if i == 0:
            color = 'orange'
        elif i == 1:
            color = 'green'
        elif i == 2:
            color = 'blue'
            try:
                slope, intercept, r, p, se = linregress(bin_centers[1:-8], np.log10(hist)[1:-8])
            except:
                slope, intercept, r, p, se = 0.0, 0.0, 0.0, None, None
        ax.scatter(10**(bin_centers), hist, s=20, color=color)

        #try:
        #    ax.plot(10**(bin_centers), 10**(slope * bin_centers + intercept), 'r--', label=f'{slope:.2f}x+{intercept:.2f} r={r:.2f}', color='red', linestyle='--')
        #except:
        #    pass

    ax.set_xscale('log')
    ax.set_xlim(0.9, 110)
    ax.set_yscale('log')
    #ax.set_ylim(bottom=5e-6)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.set_xlabel(r'Cluster Size $s$')
    ax.set_ylabel(r'$P(s)$')
    plt.savefig(f'{name}_{slope:.2f}_{intercept:.2f}_{r:.2f}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    
    fig, ax = plt.subplots(figsize=(3.0, 1.75))
    
    for j in range(len(list_of_bin_centers)):
        hist = list_of_hist[j]
        bin_centers = list_of_bin_centers[j]
        n = n_list[j]
        if j == 0:
            color = 'orange'
        elif j == 1:
            color = 'green'
        elif j == 2:
            color = 'blue'
        ax.scatter(10**(bin_centers)/n, ((10**(bin_centers))**(-1*slope))*hist, s=20, color=color)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.set_xlabel(r'$s/N$')
    ax.set_ylabel(rf'$s^{{{-1*slope:.2f}}}P(s)$')
    plt.savefig(f'{name}_finite_size_{slope:.2f}_{intercept:.2f}_{r:.2f}.png',
                dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    __spec__ = None
    mp.set_start_method('spawn', force=True)
    os.makedirs('results/3SAT/Benchmark/varied_all_avalanche_redone', exist_ok=True)

    batch = 100
    num_iterations = 100
    n_list = [60, 80, 100]
    list_of_cluster_sizes = []

    param = {"alpha_by_beta": 0.45313481433413916,
            "beta": 236.4915240060791,
            "gamma": 0.3635604327568345,
            "delta_by_gamma": 0.21883211263830715,
            "zeta": 0.06294441488786634,
            "dt_0": 0.0898215588038146,
            "time_window": 0.006,
            "lr": 1.0,
            "alpha_inc": 0}

    for n in n_list:
        '''files = []
        for instance_num in range(batch):
            file = f'data/p0_080/ratio_4_30/var_{n}/instances/transformed_barthel_n_{n}_r_4.300_p0_0.080_instance_{instance_num+1:03d}.cnf'
            files.append(file)

        for iter in range(num_iterations):
            dmm = DMM(files, True, batch=batch, param=param, eqn_choice='sean_choice')
            _, _, _, spin_traj_n, time_traj_n, _ = run_dmm(dmm, int(1e6), True, 6000, 0, break_threshold=0.5)

            #For standard avalanche extraction
            cluster_size, _, out_of_memory_flag = avalanche_analysis_mp(spin_traj_n, time_traj_n, dmm.edges_var, mp.Pool(5),
                                                                    int(np.ceil(batch / 5)), 5, 0.006)
            with open(f'results/3SAT/Benchmark/varied_all_avalanche_redone/cluster_sizes_{n}_17_3.0_1.0_0.006.txt', 'a') as f:
                for cluster in cluster_size:
                    f.write(f'{cluster}\n')'''

        with open(f'results/3SAT/Benchmark/varied_all_avalanche_redone/cluster_sizes_{n}_17_3.0_1.0_0.006.txt', 'r') as f:
            cluster_sizes = np.array([float(element.strip()) for element in f.readlines()])
        list_of_cluster_sizes.append(cluster_sizes)

    #Plots avalanche size distribution for a list of n
    avalanche_stats = avalanche_size_distribution(n_list, list_of_cluster_sizes, f'results/3SAT/Benchmark/varied_all_avalanche_redone/17_3.0_1.0_{n_list}_0.006_redone')