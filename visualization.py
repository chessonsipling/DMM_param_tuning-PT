import os
import numpy as np
import plt_config
import matplotlib.pyplot as plt
color = plt.rcParams['axes.prop_cycle'].by_key()['color']

import sys

prob_type = '3SAT' #sys.argv[1] #prob_type can ONLY take on the values '3SAT', '3R3X', OR '5R5X'

ns = [10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 210, 250, 300, 350, 400, 450, 500, 600, 700, 900, 1100, 1300, 1500, 1700, 2000]

os.makedirs(f'results/{prob_type}/Benchmark/{ns}', exist_ok=True)

#Plots median TTS (in number of steps)
folder = f'results/{prob_type}/Benchmark/{ns}'

files = os.listdir(folder)
for file in files:
    if 'step' not in file or not file.endswith('.txt'):
        continue
    data = np.loadtxt(f'{folder}/{file}')
    fig, ax = plt.subplots(1, 1)
    ns, step = data[:, 0], data[:, 1]
    ns1 = ns #ns[1:43]
    step1 = step #step[1:43]
    fit1 = np.polyfit(np.log(ns1), np.log(step1), 1)
    x_fit1 = np.linspace(np.min(ns1), np.max(ns1), 1000)
    y_fit1 = np.exp(fit1[1]) * x_fit1 ** fit1[0]
    #ns2 = ns[43:]
    #step2 = step[43:]
    #fit2 = np.polyfit(np.log(ns2), np.log(step2), 1)
    #x_fit2 = np.linspace(np.min(ns2), np.max(ns2), 1000)
    #y_fit2 = np.exp(fit2[1]) * x_fit2 ** fit2[0]
    ax.scatter(ns, step, label=f'~ $N^{{{fit1[0]:.2f}}}$', color=color[0])
    ax.plot(x_fit1, y_fit1, color='r', linestyle='--')
    #ax.plot(x_fit2, y_fit2, color='g', linestyle='--')
    plt.legend(fontsize=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of variables')
    plt.ylabel('Median steps')
    plt.savefig(f'{folder}/{file.split(".")[0]}.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

#Plots active memories
folder = f'results/{prob_type}/Benchmark/{ns}'
batch = 100
        
files = os.listdir(folder)
for file in files:
    if 'total_active_memories' not in file or not file.endswith('.txt'):
        continue
    data = np.loadtxt(f'{folder}/{file}')
    fig, ax = plt.subplots(1, 1)
    ns, active_mems = data[:, 0], data[:, 1]
    active_percent = active_mems / (batch * ns)
    ax.scatter(ns, active_percent)
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel('Number of variables')
    plt.ylabel('Active memories (%)')
    plt.savefig(f'{folder}/{file.split(".")[0]}.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()