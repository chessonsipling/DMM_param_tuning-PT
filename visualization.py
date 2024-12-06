import os
import numpy as np
import plt_config
import matplotlib.pyplot as plt
color = plt.rcParams['axes.prop_cycle'].by_key()['color']

import sys

prob_type = '3R3X' #sys.argv[1] #prob_type can ONLY take on the values '3SAT', '3R3X', OR '5R5X'

ns = [10, 20, 30]

os.makedirs(f'results/{prob_type}/Benchmark/{ns}', exist_ok=True)

folder = f'results/{prob_type}/Benchmark/{ns}'

files = os.listdir(folder)
for file in files:
    if 'step' not in file or not file.endswith('.txt'):
        continue
    data = np.loadtxt(f'{folder}/{file}')
    ns, step = data[:, 0], data[:, 1]
    fit = np.polyfit(np.log(ns), np.log(step), 1)
    fig, ax = plt.subplots(1, 1)
    x_fit = np.linspace(np.min(ns), np.max(ns), 1000)
    y_fit = np.exp(fit[1]) * x_fit ** fit[0]
    ax.scatter(ns, step, label=f'~ $N^{{{fit[0]:.2f}}}$', color=color[0])
    ax.plot(x_fit, y_fit, color='r', linestyle='--')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of variables')
    plt.ylabel('Median steps')
    plt.savefig(f'{folder}/{file.split(".")[0]}.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
