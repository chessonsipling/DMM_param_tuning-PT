import os
import numpy as np
import pickle
import json
import torch
import torch.multiprocessing as mp
from model import DMM
from dmm_utils import run_dmm, avalanche_analysis, avalanche_analysis_mp, avalanche_size_distribution
mp.set_start_method('spawn', force=True)
import matplotlib.pyplot as plt
import plt_config

import sys


def param_scaling(param, name, eqn_choice, prob_type, batch, ns, simple):
    max_step = int(1e5)
    avalanche_subprocesses = 5
    avalanche_minibatch = int(np.ceil(batch / avalanche_subprocesses))
    pool = mp.Pool(avalanche_subprocesses)

    # print(param)
    with open(f'results/{prob_type}/Benchmark/{ns}/params_{name}.json', 'w') as f:
        json.dump(param, f)

    spin_traj = []
    time_traj = []
    xl_traj = []
    xs_traj = []
    C_traj = []
    G_traj = []
    R_traj = []
    for n in ns:
        files = []
        for instance_num in range(batch):
            if prob_type == '3SAT':
                file = f'../DMM_param_tuning-main/data/p0_080/ratio_4_30/var_{n}/instances/transformed_barthel_n_{n}_r_4.300_p0_0.080_instance_{instance_num+1:03d}.cnf'
            elif prob_type == '3R3X':
                file = f'../DMM_param_tuning-main/data/XORSAT/3R3X/{n}/problem_{instance_num:04d}.cnf' #f'../DMM_param_tuning-main/data/XORSAT/3R3X/{n}/problem_{instance_num:04d}_XORgates.cnf'
            elif prob_type == '5R5X':
                file = f'../DMM_param_tuning-main/data/XORSAT/5R5X/{n}/problem_{instance_num:04d}.cnf' #f'../DMM_param_tuning-main/data/XORSAT/5R5X/{n}/problem_{instance_num:04d}_XORgates.cnf'
            files.append(file)
        dmm = DMM(files, simple, batch=batch, param=param, eqn_choice=eqn_choice)
        save_steps = 5000
        transient = 0 #1000
        # max_step = int(n ** 4 / 100)
        # max_step = save_steps + transient
        if simple:
            is_solved, solved_step, unsat_moments, step = \
                run_dmm(dmm, max_step, simple, save_steps, transient, break_threshold=0.5)
        else:
            is_solved, solved_step, unsat_moments, spin_traj_n, time_traj_n, xl_traj_n, xs_traj_n, C_traj_n, G_traj_n, R_traj_n, step = \
                run_dmm(dmm, max_step, simple, save_steps, transient, break_threshold=0.5)
            spin_traj.append(spin_traj_n)
            time_traj.append(time_traj_n)
            xl_traj.append(xl_traj_n)
            xs_traj.append(xs_traj_n)
            C_traj.append(C_traj_n)
            G_traj.append(G_traj_n)
            R_traj.append(R_traj_n)
        solved_step[~is_solved] = max_step
        n_solved = is_solved.sum()
        median_step = np.median(solved_step.cpu().numpy()) if n_solved > dmm.batch // 2 \
            else step * (dmm.batch + 1) / (2 * n_solved + 1)
        # cluster_size, out_of_memory_flag = avalanche_analysis(spin_traj, time_traj, dmm.edges_var)
        '''cluster_size, out_of_memory_flag = avalanche_analysis_mp(spin_traj, time_traj, dmm.edges_var, pool, #<<<only needed for avalanche extraction
                                                                 avalanche_minibatch, avalanche_subprocesses)
        avalanche_stats = avalanche_size_distribution(cluster_size, f'/Benchmark/{ns}/{name}_{n}')
        stats = {
            'n': n,
            'is_solved': is_solved,
            'solved_step': solved_step,
            'median_step': median_step,
            'unsat_moments': unsat_moments,
            'avalanche_stats': avalanche_stats
        }
        pickle.dump(stats, open(f'results/{prob_type}/Benchmark/{ns}/stats_{n}_{name}.pkl', 'wb'))''' #<<<only needed for avalanche extraction
        with open(f'results/{prob_type}/Benchmark/{ns}/steps_{name}.txt', 'a') as f:
            f.write(f'{n} {median_step}\n')
        print(f'N = {n} Done')

    if not simple:
        return spin_traj, xl_traj, xs_traj, C_traj, G_traj, R_traj


eqn_choice = 'diventra_choice' #sys.argv[1] #eqn_choice can ONLY take on the values 'sean_choice', 'diventra_choice', 'yuanhang_choice', and 'zeta_zero' (and 'R_zero', 'rudy_choice', or 'rudy_simple' for XORSAT)
prob_type = '3R3X' #sys.argv[2] #prob_type can ONLY take on the values '3SAT', '3R3X', OR '5R5X'

if __name__ == '__main__':
    __spec__ = None
    mp.set_start_method('spawn', force=True)

    params = [{'alpha_by_beta': 0.22524457543545404,
                'beta': 1.2347240851463632,
                'gamma': 0.17787510404154985,
                'delta_by_gamma': 0.4919757315128267,
                'zeta': 0.0003296680064710264,
                'lr': 1.0,
                'alpha_inc': 0}]

    simple = False
    batch = 100
    ns = [40, 50, 60]
    for i, param_i in enumerate(params):
        result_dir = f'results/{prob_type}/Benchmark/{ns}'
        graph_dir = f'graphs/{prob_type}/Benchmark/{ns}'
        os.makedirs(result_dir, exist_ok=True)
        #os.makedirs(graph_dir, exist_ok=True)

        if simple:
            param_scaling(param_i, str(i), eqn_choice, prob_type, batch, ns, simple)
        else:
            spin_traj, xl_traj, xs_traj, C_traj, G_traj, R_traj = param_scaling(param_i, str(i), eqn_choice, prob_type, batch, ns, simple)

    if not simple:
        for i in range(len(ns)): #iterate over variable number
            steps = np.array(list(range(len(spin_traj[i]))))
            for j in range(5): #iterate over batch, could be up to j in range(batch)
                for k in range(len(spin_traj[i][0][j])): #iterate over v, could be up to k in range(len(spin_traj[i][0][j]))
                    spin_traj_to_plot = [element[j][k] for element in spin_traj[i]] #n, step, batch, v/xl/xs
                    #plt.plot(steps, spin_traj_to_plot, label=f'{k}')
                    plt.plot(steps, spin_traj_to_plot)
                plt.xlabel('Steps')
                plt.ylabel('Voltages')
                #plt.legend()
                plt.tight_layout()
                plt.savefig(f'results/{prob_type}/Benchmark/{ns}/n{ns[i]}_batch{j}_v.png')
                plt.clf()

                for k in range(ns[i]): #iterate over n, could be up to k in range(ns[i])
                    xl_traj_to_plot = [element[j][k] for element in xl_traj[i]]
                    #plt.plot(steps, xl_traj_to_plot, label=f'{k}')
                    plt.plot(steps, xl_traj_to_plot)
                plt.xlabel('Steps')
                plt.ylabel('Long Term Memories')
                #plt.legend()
                plt.tight_layout()
                plt.savefig(f'results/{prob_type}/Benchmark/{ns}/n{ns[i]}_batch{j}_xl.png')
                plt.clf()

                for k in range(ns[i]): #iterate over n, could be up to k in range(ns[i])
                    xs_traj_to_plot = [element[j][k] for element in xs_traj[i]]
                    #plt.plot(steps, xs_traj_to_plot, label=f'{k}')
                    plt.plot(steps, xs_traj_to_plot)
                plt.xlabel('Steps')
                plt.ylabel('Short Term Memories')
                #plt.legend()
                plt.tight_layout()
                plt.savefig(f'results/{prob_type}/Benchmark/{ns}/n{ns[i]}_batch{j}_xs.png')
                plt.clf()

                for k in range(ns[i]): #iterate over n, could be up to k in range(ns[i])
                    C_traj_to_plot = [element[j][k] for element in C_traj[i]]
                    #plt.plot(steps, C_traj_to_plot, label=f'{k}')
                    plt.plot(steps, C_traj_to_plot)
                plt.xlabel('Steps')
                plt.ylabel('Clause Functions')
                #plt.legend()
                plt.tight_layout()
                plt.savefig(f'results/{prob_type}/Benchmark/{ns}/n{ns[i]}_batch{j}_C.png')
                plt.clf()

                for k in range(ns[i]): #iterate over n, could be up to k in range(ns[i])
                    for l in range(3): #iterates over 3 voltages in each clause
                        G_traj_to_plot = [element[j][k][l] for element in G_traj[i]]
                        #plt.plot(steps, G_traj_to_plot, label=f'{k}, {l}')
                        plt.plot(steps, G_traj_to_plot)
                plt.xlabel('Steps')
                plt.ylabel('Gradient Terms')
                #plt.legend(ncol=3)
                plt.tight_layout()
                plt.savefig(f'results/{prob_type}/Benchmark/{ns}/n{ns[i]}_batch{j}_G.png')
                plt.clf()

                for k in range(ns[i]): #iterate over n, could be up to k in range(ns[i])
                    for l in range(3): #iterates over 3 voltages in each clause
                        R_traj_to_plot = [element[j][k][l] for element in R_traj[i]]
                        #plt.plot(steps, R_traj_to_plot, label=f'{k}, {l}')
                        plt.plot(steps, R_traj_to_plot)
                plt.xlabel('Steps')
                plt.ylabel('Rigidity Terms')
                #plt.legend(ncol=3)
                plt.tight_layout()
                plt.savefig(f'results/{prob_type}/Benchmark/{ns}/n{ns[i]}_batch{j}_R.png')
                plt.clf()

        '''for n in ns:
            dt_file = open(f'results/{prob_type}/Benchmark/{ns}/n{n}_dt.txt', 'r')
            dt_file_output = dt_file.readlines()
            dts = np.array((np.float_([element.split(', ')[:-1] for element in dt_file_output])))
            #print(dts)
            for i in range(2): #could be up to i in range(batch)
                dt = np.array([element[i] for element in dts])
                #print(dt)
                steps = np.array(list(range(len(dt))))
                plt.plot(steps, dt)
                plt.axhline(0.1, color='red', linestyle='dashed')
                plt.ylim(0, min(10, max(dt)))
                plt.xlabel('Steps')
                plt.ylabel('dt')
                plt.tight_layout()
                plt.savefig(f'results/{prob_type}/Benchmark/{ns}/n{n}_batch{i}_dt.png')
                plt.clf()
                print('Instance ' + str(i+1) + ' dt plotted!')'''
