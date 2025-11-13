import matplotlib.pyplot as plt
import numpy as np
import plt_config
from scipy.optimize import curve_fit
import math


def inverse_gaussian(x, mu, lamb):
    return np.sqrt(lamb/(2*math.pi*x**3)) * np.exp(-(lamb*(x-mu)**2)/(2*mu**2*x))

def exponential_decay(x, b):
    return b*np.exp(-b*x)


#Plots time-to-solution distributions given TTS data on a large set of SAT instances
def tts_distribution(solved_step, prob_type, flattened_big_ns, n, name):

    plt.figure(figsize=(3.0, 1.75))
    
    #Extracts total number of instances solved
    with open(f'results/{prob_type}/Benchmark/{flattened_big_ns}/n_solved_{n}_{name}.txt', 'r') as f:
        n_solved = f.readlines()
    total_solved = sum([int(element.strip()) for element in n_solved]) #the total number of solved instances
    solved_step = np.sort(solved_step)

    #Plots all data with linear axes
    prob, bins, patches = plt.hist(solved_step, bins=100, density=True, color='blue')
    #Fits with linear axes
    bin_centers = np.array([(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)])
    #Plots TTS IG fit, as in Fig. 4 of manuscript
    plt.plot(bin_centers, inverse_gaussian(bin_centers, 20000, 1000), color='red', linestyle='dashed') #[20.0, 20.0]
    #Compares TTS IG fit to exponential fit
    plt.plot(bin_centers, exponential_decay(bin_centers, 0.003), color='lime', linestyle='dashed')

    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.xlim(0, 11500)
    #plt.ylim(top=5.2e-4)
    plt.xlabel(r'Solution Step $T$')
    plt.ylabel(r'$P(T)$')
    #plt.legend(fontsize='10')
    plt.savefig(f'results/{prob_type}/Benchmark/{flattened_big_ns}/tts_{n}_{name}_ig_redone_zoomed.png', dpi=300, bbox_inches='tight')
    plt.close()

#Plots TTS distributions
with open(f'results/3SAT/Benchmark/varied_all_avalanche/tts_100_24_20.0_20.0.txt', 'r') as f:
    solved_step = np.array([float(element.strip()) for element in f.readlines()])
tts_distribution(solved_step, '3SAT', 'varied_all_avalanche', 100, '24_20.0_20.0')