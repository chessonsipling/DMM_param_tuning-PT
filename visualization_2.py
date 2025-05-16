import os
import numpy as np
import plt_config
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
color = plt.rcParams['axes.prop_cycle'].by_key()['color']

from scipy.optimize import curve_fit

import sys

def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

prob_type = '3SAT' #sys.argv[1] #prob_type can ONLY take on the values '3SAT', '3R3X', OR '5R5X'

plt.ioff()

os.makedirs(f'results/{prob_type}/Benchmark/wide_param_search', exist_ok=True)

default_interp = 'none' #'mitchell'


#Extracts median TTS (in number of steps)
folder = f'results/{prob_type}/Benchmark/wide_param_search'
files = sorted(os.listdir(folder))
ns = []
step = []
param_vals = []
for i, file in enumerate(files):
    if 'step' not in file or not file.endswith('.txt'):
        continue
    data = np.loadtxt(f'{folder}/{file}')
    ns1, step1 = data[:, 0], data[:, 1]
    beta1 = float(file.split('_')[2])
    zeta1 = float(file.split('_')[3][:-4])
    ns.append(ns1)
    step.append(step1)
    param_vals.append([beta1, zeta1])


#Sorts ns and step based on the parameter values
param_vals, ns, step = zip(*sorted(zip(param_vals, ns, step)))
param_vals = np.array(param_vals)


#Plots different param choices alongside one another
all_param_mults = [0.02, 0.08, 0.2, 0.5, 0.8, 1.5, 3.0, 6.0, 20.0, 50.0]

#Graphing loop for beta
for i in range(len(all_param_mults)):
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.5))
    for j in range(len(param_vals)):
        if param_vals[j][0] == all_param_mults[i] and param_vals[j][1] != 1.0:
            ax.scatter(ns[j][:-1], step[j][:-1], label=f'{param_vals[j][1]}') #cuts off last data point (artificial stop @ 1E6)
            #ax.plot(x_fit[0], y_fit[0])

    #plt.legend(fontsize=24, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(top=2e6)
    plt.tick_params(labelsize=28)
    plt.xlabel(r'Number of variables $N$', fontsize=32)
    plt.ylabel('Median Solution Step', fontsize=32)
    plt.savefig(f'{folder}/beta={all_param_mults[i]}.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

#Graphing loop for zeta
for i in range(len(all_param_mults)):
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.5))
    for j in range(len(param_vals)):
        if param_vals[j][1] == all_param_mults[i] and param_vals[j][0] != 1.0:
            ax.scatter(ns[j][:-1], step[j][:-1], label=f'{param_vals[j][0]}') #cuts off last data point (artificial stop @ 1E6)
            #ax.plot(x_fit[0], y_fit[0])

    #plt.legend(fontsize=24, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(top=2e6)
    plt.tick_params(labelsize=28)
    plt.xlabel(r'Number of variables $N$', fontsize=32)
    plt.ylabel('Median Solution Step', fontsize=32)
    plt.savefig(f'{folder}/zeta={all_param_mults[i]}.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()


#Plots maximum ns before median TTS exceeded 1E6
all_param_mults = [1.0, 3.0, 6.0, 0.02, 0.08, 0.2, 0.5, 0.8, 1.5, 20.0, 50.0]

max_med_TTS = [ni[-2] for ni in ns]
data = list(zip(param_vals, max_med_TTS))

#Replaces parameter multiplier values with indices in a 2D array (from 0 to 10)
for i, mult in enumerate(all_param_mults):
    mask = np.where(param_vals == mult)
    for j in range(len(mask[0])):
        if i == 0:
            data[mask[0][j]][0][mask[1][j]] = 5
        elif i == 1:
            data[mask[0][j]][0][mask[1][j]] = 7
        elif i == 2:
            data[mask[0][j]][0][mask[1][j]] = 8
        elif i <= 7:
            data[mask[0][j]][0][mask[1][j]] = i-3
        elif i == 8:
            data[mask[0][j]][0][mask[1][j]] = 6
        elif i > 8:
            data[mask[0][j]][0][mask[1][j]] = i
data = list(zip([datapoint[0].astype(int) for datapoint in data], max_med_TTS))

plot_array = 10*np.ones((len(all_param_mults), len(all_param_mults)))

#Populates the array with data values
for x, value in data:
    plot_array[x[0]][x[1]] = value

plt.imshow(plot_array, norm=matplotlib.colors.LogNorm(vmin=plot_array.min(), vmax=plot_array.max()), cmap='magma', interpolation=default_interp)
plt.xlabel(r'$\zeta$ (units of $\zeta_{opt}$)')
plt.ylabel(r'$\beta$ (units of $\beta_{opt}$)')
plt.xticks([5, 7, 8, 0, 1, 2, 3, 4, 6, 9, 10], all_param_mults, rotation=45)
plt.yticks([5, 7, 8, 0, 1, 2, 3, 4, 6, 9, 10], all_param_mults)
plt.colorbar(label=r'$N_{max}$ ($T_{median} < 10^6$ steps)')
plt.savefig(f'{folder}/medianTTS_past_1E6_cmap_no_interp.png', dpi=300, bbox_inches='tight')
plt.close()


#Plots avalanche and TTS distributions (across wide params)
folder = f'results/{prob_type}/Benchmark/varied_all_avalanche'

#Avalanche plot
avalanche_plot = np.array([[1.75, 1.75, 1.75, 1.75, 1.75],
                           [2.42, 2.54, 2.60, 1.75, 1.75],
                           [1.75, 1.75, 2.63, 1.75, 1.75],
                           [1.75, 2.37, 2.65, 1.75, 1.75],
                           [1.75, 1.75, 2.72, 2.60, 1.75]])
cmap1 = plt.cm.viridis
cmap1.set_under('white')
plt.imshow(avalanche_plot, norm=matplotlib.colors.Normalize(vmin=2.00, vmax=3.00), cmap=cmap1, interpolation=default_interp)
plt.xlabel(r'$\zeta$ (units of $\zeta_{opt}$)')
plt.ylabel(r'$\beta$ (units of $\beta_{opt}$)')
plt.xticks([0, 1, 2, 3, 4], [0.08, 0.5, 1.0, 3.0, 20.0], rotation=45)
plt.yticks([0, 1, 2, 3, 4], [0.08, 0.5, 1.0, 3.0, 20.0])
plt.colorbar(label='Scale-free exponent')
plt.savefig(f'{folder}/avalanches_no_interp.png', dpi=300, bbox_inches='tight')
plt.close()

#TTS distribution plot
tts_plot = np.array([[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 1], [0, 1, 1, 1, 1], [0, 0, 0, 1, 1]])
cmap2 = matplotlib.colors.ListedColormap(['blue', 'green'])
blue_patch = mpatches.Patch(color='blue', label=r'Peak near $T=0$')
green_patch = mpatches.Patch(color='green', label=r'Peak after $T=0$')
plt.imshow(tts_plot, cmap=cmap2, interpolation=default_interp) #'mitchell'
plt.xlabel(r'$\zeta$ (units of $\zeta_{opt}$)')
plt.ylabel(r'$\beta$ (units of $\beta_{opt}$)')
plt.xticks([0, 1, 2, 3, 4], [0.08, 0.5, 1.0, 3.0, 20.0], rotation=45)
plt.yticks([0, 1, 2, 3, 4], [0.08, 0.5, 1.0, 3.0, 20.0])
plt.legend(handles = [blue_patch, green_patch])
plt.savefig(f'{folder}/tts_comparison_no_interp.png', dpi=300, bbox_inches='tight')
plt.close()

#Avalanche and TTS distribution plots (layered)
plt.imshow(avalanche_plot, norm=matplotlib.colors.Normalize(vmin=2.00, vmax=3.00), cmap=cmap1, interpolation=default_interp)
plt.colorbar(label=r'Scale-free exponent')
plt.imshow(tts_plot, cmap=cmap2, alpha=0.35, interpolation=default_interp)
plt.xlabel(r'$\zeta$ (units of $\zeta_{opt}$)')
plt.ylabel(r'$\beta$ (units of $\beta_{opt}$)')
plt.xticks([0, 1, 2, 3, 4], [0.08, 0.5, 1.0, 3.0, 20.0], rotation=45)
plt.yticks([0, 1, 2, 3, 4], [0.08, 0.5, 1.0, 3.0, 20.0])
plt.savefig(f'{folder}/combined_avalanche_and_tts_no_interp.png', dpi=300, bbox_inches='tight')
plt.close()


#Extracts number of anti-instantons
folder = f'results/{prob_type}/Benchmark/anti-instanton_condensation'
files = sorted(os.listdir(folder))
for file in files[:]:
    if 'instanton' not in file or not file.endswith('.txt'):
        files.remove(file)
files = sorted(files, key=lambda x:int(x.split('_')[3]))
anti_instantons_per_batch = []
param_vals_anti = []
for i, file in enumerate(files):
    anti_instanton1 = np.loadtxt(f'{folder}/{file}')
    if np.size(anti_instanton1) == 1:
        anti_instantons_per_batch.append(anti_instanton1)
    else:
        anti_instantons_per_batch.append(sum(anti_instanton1)/len(anti_instanton1))
    beta1 = float(file.split('_')[4])
    zeta1 = float(file.split('_')[5][:-4])
    param_vals_anti.append([beta1, zeta1])

#Organizes average number of anti-instantons into an array
all_param_mults = [0.02, 0.08, 0.2, 0.5, 0.8, 1.0, 1.5, 3.0, 6.0, 20.0, 50.0]

cmap3 = plt.cm.inferno
cmap3.set_under('white')
anti_instantons_per_batch = np.array(anti_instantons_per_batch).reshape(len(all_param_mults), len(all_param_mults))
anti_instantons_per_batch = np.where(anti_instantons_per_batch == 0, 0.1, anti_instantons_per_batch)
plt.imshow(anti_instantons_per_batch, norm=matplotlib.colors.LogNorm(vmin=1.0, vmax=np.max(anti_instantons_per_batch)), cmap=cmap3, interpolation='none') #'gaussian'
plt.xlabel(r'$\zeta$ (units of $\zeta_{opt}$)')
plt.ylabel(r'$\beta$ (units of $\beta_{opt}$)')
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], all_param_mults, rotation=45)
plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], all_param_mults)
plt.colorbar(label='Anti-Instantons per Batch')
plt.savefig(f'{folder}/anti_instantons_per_batch_no_interp.png', dpi=300, bbox_inches='tight')
plt.close()
