import os
import numpy as np
import plt_config
import matplotlib.pyplot as plt
color = plt.rcParams['axes.prop_cycle'].by_key()['color']

import sys

prob_type = '3SAT' #sys.argv[1] #prob_type can ONLY take on the values '3SAT', '3R3X', OR '5R5X'

ns = np.array([10, 20, 30])
#ns = '[10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 210, 250, 300, 350, 400, 450, 500, 600, 700, 900, 1100, 1300, 1500, 1700, 2000] (param comparison)'

os.makedirs(f'results/{prob_type}/Benchmark/{ns}', exist_ok=True)

#Plots median TTS (in number of steps)
folder = f'results/{prob_type}/Benchmark/{ns}'

files = sorted(os.listdir(folder))
ns = []
step = []
x_fit = []
y_fit = []
for file in files:
    if 'step' not in file or not file.endswith('.txt'):
        continue
    data = np.loadtxt(f'{folder}/{file}')
    fig, ax = plt.subplots(1, 1)
    ns1, step1 = data[:, 0], data[:, 1]
    #ns1 = ns[1:43]
    #step1 = step[1:43]
    fit1 = np.polyfit(np.log(ns1), np.log(step1), 1)
    x_fit1 = np.linspace(np.min(ns1), np.max(ns1), 1000)
    y_fit1 = np.exp(fit1[1]) * x_fit1 ** fit1[0]
    #ns2 = ns[43:]
    #step2 = step[43:]
    #fit2 = np.polyfit(np.log(ns2), np.log(step2), 1)
    #x_fit2 = np.linspace(np.min(ns2), np.max(ns2), 1000)
    #y_fit2 = np.exp(fit2[1]) * x_fit2 ** fit2[0]
    ns.append(ns1)
    step.append(step1)
    x_fit.append(x_fit1)
    y_fit.append(y_fit1)

ax.scatter(ns1, step1, label=f'~ $N^{{{fit1[0]:.2f}}}$', color=color[0])
ax.plot(x_fit1, y_fit1, color='r', linestyle='--')

plt.legend(fontsize=18)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of variables')
plt.ylabel('Median steps')
plt.savefig(f'{folder}/{file.split(".")[0]}.png', dpi=300, bbox_inches='tight')
# plt.show()
plt.close()


#Plots different param choices alongside one another
'''params = ['Optimal', r'$\beta = 10*\beta_{opt}$', r'$\beta = 100*\beta_{opt}$', r'$\zeta = 0.1*\zeta_{opt}$', r'$\zeta = 0.01*\zeta_{opt}$', r'$\beta = 10*\beta_{opt}, dt_0=0.25$', r'$\zeta = 0.1*\zeta_{opt}, dt_0=0.25$']

ax.scatter(ns[0], step[0], label=f'{params[0]}', color='g') #original
#ax.scatter(ns[1], step[1], label=f'{params[1]}', color='y') #10*\beta_{opt}
#ax.scatter(ns[2], step[2], label=f'{params[2]}', color='r') #100*\beta_{opt}
ax.scatter(ns[3], step[3], label=f'{params[3]}', color='y', marker='x') #0.1*\zeta_{opt}
#ax.scatter(ns[4], step[4], label=f'{params[4]}', color='r', marker='x') #0.01*\zeta_{opt}
#ax.scatter(ns[5], step[5], label=f'{params[5]}', color='y', marker='*') #10*\beta_{opt}, dt_0=0.25
ax.scatter(ns[6], step[6], label=f'{params[6]}', color='y', marker='*') #0.1*\zeta_{opt}, dt_0=0.25

#ax.plot(x_fit1, y_fit1, color='r', linestyle='--')
#ax.plot(x_fit2, y_fit2, color='g', linestyle='--')

plt.legend(fontsize=12, loc= 'lower right')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of variables')
plt.ylabel('Median steps')
plt.savefig(f'{folder}/zeta (dt comparison).png', dpi=300, bbox_inches='tight')
# plt.show()
plt.close()'''


#Plots active memories
'''folder = f'results/{prob_type}/Benchmark/{ns}'
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
    plt.close()'''