import os
import numpy as np
from solver_PT import Solver_PT
import matplotlib.pyplot as plt
import pandas as pd
from data_analysis import data_analysis


#############################################################################################################################################################################################
#FREE PARAMETERS DURING TRAINING
eqn_choice = 'sean_choice' #specifies DMM equations; can ONLY take on the values 'sean_choice', 'diventra_choice', 'yuanhang_choice', and 'zeta_zero' (and 'R_zero', 'rudy_choice', or 'rudy_simple' for XORSAT)
prob_type = '3SAT' #specficies type of CO problem to solve; prob_type can ONLY take on the values '3SAT', '3R3X', OR '5R5X'
batch = 100 #number of instances in a batch
max_step = int(5e3) #maximum simulation step
tag = '_testing' #tag to add to file names
big_ns = np.array([[10, 20, 30], [40, 50, 60]]) #array of system sizes N to simulate; include all sizes in which parameters were tuned simultaneously in the same sublist
max_evals = 100 #maximum number of temperatures tested during <E(T)> plot generation, geometrically spaced between lower_T and upper_T
#############################################################################################################################################################################################


flattened_big_ns = str(big_ns.flatten().tolist()) + tag
os.makedirs(f'training/{prob_type}/{flattened_big_ns}', exist_ok=True)

if __name__ == '__main__':
    __spec__ = None

    
    #Collects instance data
    for i, ns in enumerate(big_ns):
        cnf_files = []
        for j, n in enumerate(ns):
            cnf_files_n = []
            for k in range(batch):
                if prob_type == '3SAT':
                    file = f'data/p0_080/ratio_4_30/var_{n}/instances/transformed_barthel_n_{n}_r_4.300_p0_0.080_instance_{k+1:03d}.cnf'
                elif prob_type == '3R3X':
                    file = f'/data/XORSAT/3R3X/{n}/problem_{k:04d}.cnf' #f'/data/XORSAT/3R3X/{n}/problem_{k:04d}_XORgates.cnf'
                elif prob_type == '5R5X':
                    file = f'/data/XORSAT/5R5X/{n}/problem_{k:04d}.cnf' #f'/data/XORSAT/5R5X/{n}/problem_{j=k:04d}_XORgates.cnf'
                cnf_files_n.append(file)
            cnf_files.append(cnf_files_n)

        #<E(T)> plot generation; used to establish T_min and T_max (CHOOSE ONLY ONE, SELECT CORRESPONDING OPTIONS IN run() IN solver_PT.py)
        '''replicas = 1
        for j in range(50): #repeat many times for ensemble average <E>
            solver = Solver_PT(ns, cnf_files, prob_type, True, replicas, big_ns, flattened_big_ns, 0, steps=max_step, batch=batch, lower_T=1e-2, upper_T=1e7) #here, each temperature is tested only once (no swapping)
            solver.run(max_evals=max_evals) #here, max_evals is the number of temperatures sampled over in annealing process
        with open(f'training/{prob_type}/{flattened_big_ns}/annealed_energy_{ns}.txt', 'r') as f:
            filedata = [(float(element.strip().split(',')[0]), float(element.strip().split(',')[1])) for element in f.readlines()]
            processed_data = pd.DataFrame(filedata, columns=['x', 'y']).groupby('x')['y'].mean().to_dict() #averages over energies so that <E(T)> is improved by each run
            temps = np.array([1/float(element) for element in processed_data.keys()])
            energies = np.array([float(element) for element in processed_data.values()])
        plt.plot(temps, energies)
        plt.xscale('log')
        plt.xlabel('T')
        plt.ylabel('<E>')
        plt.tight_layout()
        plt.savefig(f'training/{prob_type}/{flattened_big_ns}/annealed_energy_distrb_{ns}.png')
        plt.clf()'''

        #Applies Parallel Tempering (PT) approach to optimize parameters (CHOOSE ONLY ONE, SELECT CORRESPONDING OPTIONS IN run() IN solver_PT.py)
        replicas = int(0.05*np.average(ns) + 10) #establishes number of replicas in ensemble
        solver = Solver_PT(ns, cnf_files, prob_type, True, replicas, big_ns, flattened_big_ns, i, steps=max_step, batch=batch, lower_T=1e-1, upper_T=1e4) #Initializes Parallel Tempering (PT) device
        solver.run(max_evals=int(100/replicas)) #performs PT to optimize parameters over a given triple of sizes; max_evals*replicas is fixed at 100
        data_analysis(eqn_choice, prob_type, ns, flattened_big_ns) #Analyzes data after PT scheme is complete (extracts optimal parameters to a .json file)
