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
    max_step = int(1e7)
    avalanche_subprocesses = 5
    avalanche_minibatch = int(np.ceil(batch / avalanche_subprocesses))
    pool = mp.Pool(avalanche_subprocesses)

    # print(param)
    with open(f'results/{prob_type}/Benchmark/{ns}/params_{name}.json', 'w') as f:
        json.dump(param, f)

    spin_traj = []
    time_traj = []
    v_traj = []
    xl_traj = []
    xs_traj = []
    C_traj = []
    G_traj = []
    R_traj = []
    dt_traj = []
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
        transient = 0
        # max_step = int(n ** 4 / 100)
        # max_step = save_steps + transient
        if simple:
            is_solved, solved_step, unsat_moments, spin_traj_n, time_traj_n, step = \
                run_dmm(dmm, max_step, simple, save_steps, transient, break_threshold=0.5)
        else:
            is_solved, solved_step, unsat_moments, spin_traj_n, time_traj_n, v_traj_n, xl_traj_n, xs_traj_n, C_traj_n, G_traj_n, R_traj_n, dt_traj_n, step = \
                run_dmm(dmm, max_step, simple, save_steps, transient, break_threshold=0.5)
            spin_traj.append(spin_traj_n)
            time_traj.append(time_traj_n)
            v_traj.append(v_traj_n)
            xl_traj.append(xl_traj_n)
            xs_traj.append(xs_traj_n)
            C_traj.append(C_traj_n)
            G_traj.append(G_traj_n)
            R_traj.append(R_traj_n)
            dt_traj.append(dt_traj_n)
        solved_step[~is_solved] = max_step
        n_solved = is_solved.sum()
        median_step = np.median(solved_step.cpu().numpy()) if n_solved > dmm.batch // 2 \
            else step * (dmm.batch + 1) / (2 * n_solved + 1)
        # cluster_size, out_of_memory_flag = avalanche_analysis(spin_traj, time_traj, dmm.edges_var)
        cluster_size, out_of_memory_flag = avalanche_analysis_mp(spin_traj_n, time_traj_n, dmm.edges_var, pool,
                                                                 avalanche_minibatch, avalanche_subprocesses)
        avalanche_stats = avalanche_size_distribution(cluster_size, f'results/{prob_type}/Benchmark/{ns}/{name}_{n}')
        stats = {
            'n': n,
            'is_solved': is_solved,
            'solved_step': solved_step,
            'median_step': median_step,
            'unsat_moments': unsat_moments,
            'avalanche_stats': avalanche_stats
        }
        pickle.dump(stats, open(f'results/{prob_type}/Benchmark/{ns}/stats_{n}_{name}.pkl', 'wb'))

        with open(f'results/{prob_type}/Benchmark/{ns}/steps_{name}.txt', 'a') as f:
            f.write(f'{n} {median_step}\n')
        print(f'N = {n} Done')

    if not simple:
        return spin_traj, time_traj, v_traj, xl_traj, xs_traj, C_traj, G_traj, R_traj, dt_traj


eqn_choice = 'sean_choice' #sys.argv[1] #eqn_choice can ONLY take on the values 'sean_choice', 'diventra_choice', 'yuanhang_choice', and 'zeta_zero' (and 'R_zero', 'rudy_choice', or 'rudy_simple' for XORSAT)
prob_type = '3SAT' #sys.argv[2] #prob_type can ONLY take on the values '3SAT', '3R3X', OR '5R5X'

if __name__ == '__main__':
    __spec__ = None
    mp.set_start_method('spawn', force=True)

    simple = False
    batch = 100
    ns = np.array([10, 20, 30]) #np.array([10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 210, 250, 300, 350, 400, 450, 500, 600, 700, 900, 1100, 1300, 1500, 1700, 2000]) #3SAT
    #ns = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]) #3R3X

    with open(f'parameters/{prob_type}/{ns}/optimal_param_{ns}.json', 'r') as f:
        all_params = json.load(f)
    params = [{'alpha_by_beta': all_params['alpha_by_beta'],
                'beta': all_params['beta'],
                'gamma': all_params['gamma'],
                'delta_by_gamma': all_params['delta_by_gamma'],
                'zeta': all_params['zeta'],
                'dt_0': all_params['dt_0'],
                'lr': all_params['lr'],
                'alpha_inc': all_params['alpha_inc']}]

    for i, param_i in enumerate(params):
        result_dir = f'results/{prob_type}/Benchmark/{ns}'
        os.makedirs(result_dir, exist_ok=True)

        if simple:
            param_scaling(param_i, str(i), eqn_choice, prob_type, batch, ns, simple)
        else:
            spin_traj, time_traj, v_traj, xl_traj, xs_traj, C_traj, G_traj, R_traj, dt_traj = param_scaling(param_i, str(i), eqn_choice, prob_type, batch, ns, simple)

    if not simple:
        for i in range(len(ns)): #iterate over variable number, could be up to i in range(len(ns))
            #steps = np.array(list(range(len(spin_traj[i]))))

            '''total_active_memories = 0
            for j in range(batch):
                active_memories = 0
                for k in range(ns[i]):
                    xs_traj_physical = [element[j][k] for element in xs_traj[i]]
                    if max(xs_traj_physical) > 0.50:
                        active_memories += 1
                with open(f'results/{prob_type}/Benchmark/{ns}/active_memories_n={ns[i]}_batch={batch}.txt', 'a') as f:
                    f.write(f'{active_memories}\n')
                total_active_memories += active_memories
            with open(f'results/{prob_type}/Benchmark/{ns}/total_active_memories_batch={batch}.txt', 'a') as f:
                f.write(f'{ns[i]} {total_active_memories}\n')
            print(f'Active Memories: {total_active_memories} out of {ns[i]}; {total_active_memories / (batch* int(ns[i]))}')'''

            time_traj_to_plot = time_traj[i][0]
            for j in range(3): #iterate over batch, could be up to j in range(batch)
                for k in range(10): #iterate over v, could be up to k in range(len(spin_traj[i][0][j]))
                    v_traj_to_plot = [element[j][k] for element in v_traj[i]] #n, step, batch, v/xl/xs
                    #plt.plot(time_traj_to_plot, spin_traj_to_plot, label=f'{k}')
                    plt.plot(time_traj_to_plot, v_traj_to_plot)
                plt.xlabel('Time')
                plt.ylabel('Voltages')
                #plt.legend()
                plt.tight_layout()
                plt.savefig(f'results/{prob_type}/Benchmark/{ns}/n{ns[i]}_batch{j}_v.png')
                plt.clf()

                for k in range(10): #iterate over n, could be up to k in range(ns[i])
                    xl_traj_to_plot = [element[j][k] for element in xl_traj[i]]
                    #plt.plot(time_traj_to_plot, xl_traj_to_plot, label=f'{k}')
                    plt.plot(time_traj_to_plot, xl_traj_to_plot)
                plt.xlabel('Time')
                plt.ylabel('Long Term Memories')
                #plt.legend()
                plt.tight_layout()
                plt.savefig(f'results/{prob_type}/Benchmark/{ns}/n{ns[i]}_batch{j}_xl.png')
                plt.clf()

                for k in range(10): #iterate over n, could be up to k in range(ns[i])
                    xs_traj_to_plot = [element[j][k] for element in xs_traj[i]]
                    #plt.plot(time_traj_to_plot, xs_traj_to_plot, label=f'{k}')
                    plt.plot(time_traj_to_plot, xs_traj_to_plot)
                plt.xlabel('Time')
                plt.ylabel('Short Term Memories')
                #plt.legend()
                plt.tight_layout()
                plt.savefig(f'results/{prob_type}/Benchmark/{ns}/n{ns[i]}_batch{j}_xs.png')
                plt.clf()

                for k in range(10): #iterate over n, could be up to k in range(ns[i])
                    C_traj_to_plot = [element[j][k] for element in C_traj[i]]
                    #plt.plot(time_traj_to_plot, C_traj_to_plot, label=f'{k}')
                    plt.plot(time_traj_to_plot, C_traj_to_plot)
                plt.xlabel('Time')
                plt.ylabel('Clause Functions')
                #plt.legend()
                plt.tight_layout()
                plt.savefig(f'results/{prob_type}/Benchmark/{ns}/n{ns[i]}_batch{j}_C.png')
                plt.clf()

                for k in range(10): #iterate over n, could be up to k in range(ns[i])
                    for l in range(3): #iterates over 3 voltages in each clause
                        G_traj_to_plot = [element[j][k][l] for element in G_traj[i]]
                        #plt.plot(time_traj_to_plot, G_traj_to_plot, label=f'{k}, {l}')
                        plt.plot(time_traj_to_plot, G_traj_to_plot)
                plt.xlabel('Time')
                plt.ylabel('Gradient Terms')
                #plt.legend(ncol=3)
                plt.tight_layout()
                plt.savefig(f'results/{prob_type}/Benchmark/{ns}/n{ns[i]}_batch{j}_G.png')
                plt.clf()

                for k in range(10): #iterate over n, could be up to k in range(ns[i])
                    for l in range(3): #iterates over 3 voltages in each clause
                        R_traj_to_plot = [element[j][k][l] for element in R_traj[i]]
                        #plt.plot(time_traj_to_plot, R_traj_to_plot, label=f'{k}, {l}')
                        plt.plot(time_traj_to_plot, R_traj_to_plot)
                plt.xlabel('Time')
                plt.ylabel('Rigidity Terms')
                #plt.legend(ncol=3)
                plt.tight_layout()
                plt.savefig(f'results/{prob_type}/Benchmark/{ns}/n{ns[i]}_batch{j}_R.png')
                plt.clf()

                dt_traj_to_plot = [element[0] for element in dt_traj[i]]
                #plt.plot(time_traj_to_plot, dt_traj_to_plot, label=f'{k}')
                plt.plot(time_traj_to_plot, dt_traj_to_plot)
                plt.axhline(0.1, color='red', linestyle='dashed')
                plt.ylim(0, min(10, max(dt_traj_to_plot)))
                plt.xlabel('Time')
                plt.ylabel('dt')
                plt.tight_layout()
                plt.savefig(f'results/{prob_type}/Benchmark/{ns}/n{ns[i]}_batch{j}_dt.png')
                plt.clf()
