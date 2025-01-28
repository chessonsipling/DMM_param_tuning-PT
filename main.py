import os
import numpy as np
import torch
from solver import Solver

import sys

eqn_choice = 'sean_choice' #sys.argv[1] #eqn_choice can ONLY take on the values 'sean_choice', 'diventra_choice', 'yuanhang_choice', and 'zeta_zero' (and 'R_zero', 'rudy_choice', or 'rudy_simple' for XORSAT)
prob_type = '3SAT' #sys.argv[1] #prob_type can ONLY take on the values '3SAT', '3R3X', OR '5R5X'
simple = True

if __name__ == '__main__':
    __spec__ = None
    os.makedirs(f'results/{prob_type}', exist_ok=True)
    os.makedirs(f'ckpts/{prob_type}', exist_ok=True)
    #os.makedirs(f'graphs/{prob_type}', exist_ok=True)
    ns = [10, 20, 30]
    instances_per_size = 100
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
    solver = Solver(ns, cnf_files, prob_type, simple)
    solver.run(max_evals=1000)
