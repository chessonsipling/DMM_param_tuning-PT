import os
import numpy as np
import plt_config
import matplotlib.pyplot as plt
color = plt.rcParams['axes.prop_cycle'].by_key()['color']

from scipy.optimize import curve_fit

import sys

def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

prob_type = '3SAT' #sys.argv[1] #prob_type can ONLY take on the values '3SAT', '3R3X', OR '5R5X'

#ns = np.array([10, 20, 30])
ns = 'scalability_comparison' #'[10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 210, 250, 300, 350, 400, 450, 500, 600, 700, 900, 1100, 1300, 1500, 1700, 2000] (param comparison)'

os.makedirs(f'results/{prob_type}/Benchmark/{ns}', exist_ok=True)

plt.ioff()

#Plots median TTS (in number of steps)
folder = f'results/{prob_type}/Benchmark/{ns}'

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
    #ns1 = ns[1:43]
    #step1 = step[1:43]
    if i == 1:
        fit1 = np.polyfit(np.log(ns1[1:43]), np.log(step1[1:43]), 1)
        x_fit1 = np.linspace(np.min(ns1[1:43]), np.max(ns1[1:43]), 1000)
        y_fit1 = np.exp(fit1[1]) * x_fit1 ** fit1[0]
        print(f'alpha = {fit1[0]}')
    elif i == 9:
        #opt_params, cov = curve_fit(exponential_func, ns1[1:], step1[1:], p0=[62, 0.041, 50])
        a_opt1, b_opt1, c_opt1 = 62, 0.041, 50
        #a_opt1, b_opt1, c_opt1 = opt_params
        x_fit1 = np.linspace(np.min(ns1[1:]), np.max(ns1[1:]), 1000)
        y_fit1 = exponential_func(x_fit1, a_opt1, b_opt1, c_opt1)
        a_opt.append(a_opt1)
        b_opt.append(b_opt1)
        c_opt.append(c_opt1)
    elif i == 10:
        #opt_params, cov = curve_fit(exponential_func, ns1[1:], step1[1:], p0=[2, 0.120, 190])
        a_opt1, b_opt1, c_opt1 = 10, 0.068, 350
        #a_opt1, b_opt1, c_opt1 = opt_params
        x_fit1 = np.linspace(np.min(ns1[1:]), np.max(ns1[1:]), 1000)
        y_fit1 = exponential_func(x_fit1, a_opt1, b_opt1, c_opt1)
        a_opt.append(a_opt1)
        b_opt.append(b_opt1)
        c_opt.append(c_opt1)
    elif i == 12:
        #opt_params, cov = curve_fit(exponential_func, ns1[1:], step1[1:], p0=[2, 0.120, 190])
        a_opt1, b_opt1, c_opt1 = 2, 0.120, 190
        #a_opt1, b_opt1, c_opt1 = opt_params
        x_fit1 = np.linspace(np.min(ns1[1:]), np.max(ns1[1:]), 1000)
        y_fit1 = exponential_func(x_fit1, a_opt1, b_opt1, c_opt1)
        a_opt.append(a_opt1)
        b_opt.append(b_opt1)
        c_opt.append(c_opt1)
    elif i == 22:
        fit1 = np.polyfit(np.log(ns1[3:]), np.log(step1[3:]), 1)
        x_fit1 = np.linspace(np.min(ns1[3:]), np.max(ns1[3:]), 1000)
        y_fit1 = np.exp(fit1[1]) * x_fit1 ** fit1[0]
        print(f'alpha = {fit1[0]}')
    '''elif i == 10:
        #opt_params, cov = curve_fit(exponential_func, ns1[1:], step1[1:], p0=[2, 0.120, 190])
        a_opt1, b_opt1, c_opt1 = 250, 0.018, 0 #???
        #a_opt1, b_opt1, c_opt1 = opt_params
        x_fit1 = np.linspace(np.min(ns1[1:]), np.max(ns1[1:]), 1000)
        y_fit1 = exponential_func(x_fit1, a_opt1, b_opt1, c_opt1)
        a_opt.append(a_opt1)
        b_opt.append(b_opt1)
        c_opt.append(c_opt1)'''
    if i > 0:
        ns.append(ns1)
        step.append(step1)
    if i in [1, 9, 10, 12, 22]: #not able to fit for i==10 (larger \zeta)
        x_fit.append(x_fit1)
        y_fit.append(y_fit1)

fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.5))

'''ax.scatter(ns1, step1, label=f'~ $N^{{{fit1[0]:.2f}}}$', color=color[0])
ax.plot(x_fit1, y_fit1, color='r', linestyle='--')

plt.legend(fontsize=18)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of variables')
plt.ylabel('Median steps')
plt.savefig(f'{folder}/{file.split(".")[0]}.png', dpi=300, bbox_inches='tight')
# plt.show()
plt.close()'''


#Plots different param choices alongside one another
params = [f'Optimal', r'$\beta = 10\beta_{opt}$',
          r'$\beta = 100\beta_{opt}$', r'$\zeta = \zeta_{opt}/10$',
          r'$\zeta = \zeta_{opt}/100$', r'$\beta = 10\beta_{opt}$, dt$_0=0.25$',
          r'$\zeta = \zeta_{opt}/10$, dt$_0=0.25$', r'Optimal, Tuned dt$_0$',
          r'$\beta = 10\beta_{opt}$', r'$\beta = \beta_{opt}/10$',
          r'$\zeta = 10\zeta_{opt}$', r'$\zeta = \zeta_{opt}/10$',
          r'$(1.5, 1.5)$, old tuned $dt_0$', r'$(3.0, 3.0)$, old tuned $dt_0$',
          r'$(1.5, 1.5)$, new tuned $dt_0$', r'$(1.5, 1.5)$, $dt_0=1.0$',
          r'$(3.0, 3.0)$, $dt_0=1.0$', r'$(1.0, 1.0)$, $dt_0=1.0$, Exp $x_m^l$',
          r'$(1.0, 1.0)$, $dt_0=1.0$, Lin $x_m^l$', r'$(1.5, 1.5)$, $dt_0=1.0$, Lin $x_m^l$',
          r'$(3.0, 3.0)$, $dt_0=1.0$, Lin $x_m^l$', r'$\beta=0.8\beta_{opt}$'',\n'r'$\zeta=0.8\zeta_{opt}$']

ax.scatter(ns[0][1:43], step[0][1:43], label=f'{params[0]}', color='limegreen', marker='.') #original
#ax.scatter(ns[1], step[1], label=f'{params[1]}', color='y') #10*\beta_{opt}
#ax.scatter(ns[2], step[2], label=f'{params[2]}', color='r') #100*\beta_{opt}
#ax.scatter(ns[3], step[3], label=f'{params[3]}', color='r') #0.1*\zeta_{opt}
#ax.scatter(ns[4], step[4], label=f'{params[4]}', color='r', marker='x') #0.01*\zeta_{opt}
#ax.scatter(ns[5], step[5], label=f'{params[5]}', color='y', marker='*') #10*\beta_{opt}, dt_0=0.25
#ax.scatter(ns[6], step[6], label=f'{params[6]}', color='y', marker='*') #0.1*\zeta_{opt}, dt_0=0.25
#ax.scatter(ns[7], step[7], label=f'{params[7]}', color='g', marker='o') #original, tuned dt_0
ax.scatter(ns[8][1:], step[8][1:], label=f'{params[8]}', color='darkorange', marker='.') #10*\beta_{opt}, tuned dt_0
ax.scatter(ns[9][1:], step[9][1:], label=f'{params[9]}', color='darkorange', marker='*') #0.1*\beta_{opt}, tuned dt_0
ax.scatter(ns[10][1:], step[10][1:], label=f'{params[10]}', color='blue', marker='.') #10*\zeta_{opt}, tuned dt_0
ax.scatter(ns[11][1:], step[11][1:], label=f'{params[11]}', color='blue', marker='*') #0.1*\zeta_{opt}, tuned dt_0
#ax.scatter(ns[12][1:], step[12][1:], label=f'{params[12]}', color='red', marker='*') #(1.5, 1.5), previously tuned dt_0
#ax.scatter(ns[13][1:], step[13][1:], label=f'{params[13]}', color='orange', marker='*') #(3.0, 3.0), previously tuned dt_0
#ax.scatter(ns[14][1:], step[14][1:], label=f'{params[14]}', color='yellow', marker='*') #(1.5, 1.5), newly tuned dt_0
#ax.scatter(ns[15][1:], step[15][1:], label=f'{params[15]}', color='blue', marker='*') #(1.5, 1.5), dt_0=1.0
#ax.scatter(ns[16][1:], step[16][1:], label=f'{params[16]}', color='purple', marker='*') #(3.0, 3.0), dt_0=1.0
#ax.scatter(ns[17][1:], step[17][1:], label=f'{params[17]}', color='red', marker='*') #(1.0, 1.0), dt_0=1.0, cluster, exp xlm
#ax.scatter(ns[18][1:], step[18][1:], label=f'{params[18]}', color='orange', marker='*') #(1.0, 1.0), dt_0=1.0, cluster, lin xlm
#ax.scatter(ns[19][1:], step[19][1:], label=f'{params[19]}', color='yellow', marker='*') #(1.5, 1.5), dt_0=1.0, cluster, lin xlm
#ax.scatter(ns[20][1:], step[20][1:], label=f'{params[20]}', color='blue', marker='*') #(3.0, 3.0), dt_0=1.0, cluster, lin xlm
ax.scatter(ns[21][1:], step[21][1:], label=f'{params[21]}', color='green', marker='*') #(0.8, 0.8), dt_0=1.0, cluster, lin xlm

ax.plot(x_fit[0], y_fit[0], color='lime', linestyle='--')
ax.plot(x_fit[1], y_fit[1], color='orange', linestyle='--')
ax.plot(x_fit[2], y_fit[2], color='orange', linestyle='--')
#ax.plot(x_fit[3], y_fit[3], color='red', linestyle='--') #not able to fit for i==10 (larger \zeta)
ax.plot(x_fit[3], y_fit[3], color='blue', linestyle='--')
ax.plot(x_fit[4], y_fit[4], color='green', linestyle='--')

plt.legend(fontsize=14, loc='upper left')
plt.xscale('log')
plt.yscale('log')
#plt.ylim(bottom=2e1, top=3e7)
plt.xlabel(r'Number of variables $N$', fontsize=24)
plt.ylabel(r'Median solution step $T_{median}$', fontsize=24)
plt.savefig(f'{folder}/combined_2.png', dpi=300, bbox_inches='tight') ###change file name
# plt.show()
plt.close()


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