import os
import numpy as np
import torch
from solver_PT import Solver_PT
import matplotlib.pyplot as plt
import pandas as pd

import sys

prob_type = '3SAT' #sys.argv[1] #prob_type can ONLY take on the values '3SAT', '3R3X', OR '5R5X'
simple = False #must be False to extract avalanches

if __name__ == '__main__':
    __spec__ = None
    os.makedirs(f'results/{prob_type}', exist_ok=True)
    os.makedirs(f'ckpts/{prob_type}', exist_ok=True)
    #os.makedirs(f'graphs/{prob_type}', exist_ok=True)
    ns = [10, 20, 30]
    instances_per_size = 100
    replicas = 10
    cnf_files = []
    for n in ns:
        cnf_files_n = []
        for i in range(instances_per_size):
            if prob_type == '3SAT':
                file = f'../DMM_param_tuning-main/data/p0_080/ratio_4_30/var_{n}/instances/transformed_barthel_n_{n}_r_4.300_p0_0.080_instance_{i+1:03d}.cnf'
            elif prob_type == '3R3X':
                file = f'../DMM_param_tuning-main/data/XORSAT/3R3X/{n}/problem_{i:04d}.cnf' #f'../DMM_param_tuning-main/data/XORSAT/3R3X/{n}/problem_{i:04d}_XORgates.cnf'
            elif prob_type == '5R5X':
                file = f'../DMM_param_tuning-main/data/XORSAT/5R5X/{n}/problem_{i:04d}.cnf' #f'../DMM_param_tuning-main/data/XORSAT/5R5X/{n}/problem_{i:04d}_XORgates.cnf'
            cnf_files_n.append(file)
        cnf_files.append(cnf_files_n)

    #<E(T)> plot generation
    #instances_per_size = 1000 (or other large number)
    #replicas != 1
    '''for i in range(45): #repeat many times for ensemble average <E>
        solver = Solver_PT(ns, cnf_files, prob_type, simple, replicas, batch=instances_per_size)
        solver.run(max_evals=100) #here, max_evals is the number of temperatures sampled over in annealing process
    with open(f'results/{prob_type}/annealed_energy_{ns}.txt', 'r') as f:
        filedata = [(float(element.strip().split(',')[0]), float(element.strip().split(',')[1])) for element in f.readlines()]
        processed_data = pd.DataFrame(filedata, columns=['x', 'y']).groupby('x')['y'].mean().to_dict() #averages over energies so that <E(T)> can be improved with each run!
        temps = np.array([1/float(element) for element in processed_data.keys()])
        energies = np.array([float(element) for element in processed_data.values()])
    plt.plot(temps, energies)
    plt.xscale('log')
    plt.xlabel('T')
    plt.ylabel('<E>')
    plt.tight_layout()
    plt.savefig(f'results/{prob_type}/annealed_energy_distrb_{ns}.png')
    plt.clf()'''

    #Standard PT
    solver = Solver_PT(ns, cnf_files, prob_type, simple, replicas, batch=instances_per_size)
    solver.run(max_evals=10)
