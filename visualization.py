import os
import numpy as np
import matplotlib.pyplot as plt
color = plt.rcParams['axes.prop_cycle'].by_key()['color']

import sys

prob_type = '3SAT' #sys.argv[1] #prob_type can ONLY take on the values '3SAT', '3R3X', OR '5R5X'

ns = np.array([10, 20, 30])

os.makedirs(f'results/{prob_type}/Benchmark/{ns}', exist_ok=True)

#Plots median TTS (in number of steps)
folder = f'results/{prob_type}/Benchmark/{subfolder}'

files = sorted(os.listdir(folder))
ns = []
step = []
x_fit = []
y_fit = []
a_opt = []
b_opt = []
c_opt = []
for i, file in enumerate(files):
    if 'step' not in file or not file.endswith('.txt'):
        continue
    data = np.loadtxt(f'{folder}/{file}')
    ns1, step1 = data[:, 0], data[:, 1]
    if i == 1:
        fit1 = np.polyfit(np.log(ns1[1:43]), np.log(step1[1:43]), 1)
        x_fit1 = np.linspace(np.min(ns1[1:43]), np.max(ns1[1:43]), 1000)
        y_fit1 = np.exp(fit1[1]) * x_fit1 ** fit1[0]
        print(f'alpha = {fit1[0]}')
    elif i == 9: #(bigger \beta)
        #opt_params, cov = curve_fit(exponential_func, ns1[1:], step1[1:])
        #a_opt1, b_opt1, c_opt1 = opt_params
        a_opt1, b_opt1, c_opt1 = 62, 0.041, 50
        x_fit1 = np.linspace(np.min(ns1[1:]), np.max(ns1[1:]), 1000)
        y_fit1 = exponential_func(x_fit1, a_opt1, b_opt1, c_opt1)
        a_opt.append(a_opt1)
        b_opt.append(b_opt1)
        c_opt.append(c_opt1)
    elif i == 10: #(smaller \beta)
        #opt_params, cov = curve_fit(exponential_func, ns1[1:], step1[1:])
        #a_opt1, b_opt1, c_opt1 = opt_params
        a_opt1, b_opt1, c_opt1 = 10, 0.068, 350
        x_fit1 = np.linspace(np.min(ns1[1:]), np.max(ns1[1:]), 1000)
        y_fit1 = exponential_func(x_fit1, a_opt1, b_opt1, c_opt1)
        a_opt.append(a_opt1)
        b_opt.append(b_opt1)
        c_opt.append(c_opt1)
    #Not able to fit to larger \zeta
    elif i == 12: #(smaller \zeta)
        #opt_params, cov = curve_fit(exponential_func, ns1[1:], step1[1:])
        #a_opt1, b_opt1, c_opt1 = opt_params
        a_opt1, b_opt1, c_opt1 = 2, 0.120, 190
        x_fit1 = np.linspace(np.min(ns1[1:]), np.max(ns1[1:]), 1000)
        y_fit1 = exponential_func(x_fit1, a_opt1, b_opt1, c_opt1)
        a_opt.append(a_opt1)
        b_opt.append(b_opt1)
        c_opt.append(c_opt1)
    elif i == 22: #({\beta, \zeta} = {0.8\beta_{opt}, 0.8\zeta_{opt}})
        fit1 = np.polyfit(np.log(ns1[3:]), np.log(step1[3:]), 1)
        x_fit1 = np.linspace(np.min(ns1[3:]), np.max(ns1[3:]), 1000)
        y_fit1 = np.exp(fit1[1]) * x_fit1 ** fit1[0]
        print(f'alpha = {fit1[0]}')
    if i > 0:
        ns.append(ns1)
        step.append(step1)
    if i in [1, 9, 10, 12, 22]: #not able to fit for larger \zeta
        x_fit.append(x_fit1)
        y_fit.append(y_fit1)

fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.5))


#Plots different param choices alongside one another
'''params = [r'$\beta = 10*\beta_{opt}$', 'Optimal', r'$\beta = 100*\beta_{opt}$']
colors = ['y', 'g', 'r']

ax.scatter(ns[1], step[1], label=f'{params[1]}', color=colors[1])
ax.scatter(ns[0], step[0], label=f'{params[0]}', color=colors[0])
ax.scatter(ns[2], step[2], label=f'{params[2]}', color=colors[2])

#ax.plot(x_fit1, y_fit1, color='r', linestyle='--')
#ax.plot(x_fit2, y_fit2, color='g', linestyle='--')

plt.legend(fontsize=18)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of variables')
plt.ylabel('Median steps')
plt.savefig(f'{folder}/combined.png', dpi=300, bbox_inches='tight')
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