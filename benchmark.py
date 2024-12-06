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

import sys


def param_scaling(param, name, eqn_choice, prob_type, batch, ns, simple):
    max_step = int(1e4)
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

    if eqn_choice == 'rudy_simple':
        params = [{"alpha": 0.04971315994791988, #0.04971315994791988
                   "delta": 0.6836765386231144, #0.6836765386231144
                   "chi": 0.3920514767288729, #0.3920514767288729
                   "zeta": 1.0086447902802087, #1.0086447902802087
                   "lr": 0.5774245666426205, #0.5774245666426205
                   "alpha_inc": 0}] #setting to 0 removes the ability for alpha to be incremented based on the value of the long-term memory (one less hyperparameter)
    elif eqn_choice == 'rudy_choice':
        params = [{"alpha_by_beta": 0.6673803458867835, #0.6673803458867835
                   "beta": 0.12025788113438217, #0.12025788113438217
                   "delta": 0.23882315104794455, #0.23882315104794455
                   "chi": 0.7666130790024149, #0.7666130790024149
                   "zeta": 4.185850844132004, #4.185850844132004
                   "lr": 3.207904660978736, #3.207904660978736
                   "alpha_inc": 0}]
    elif eqn_choice == 'zeta_zero' or eqn_choice == 'R_zero':
        params = [{"alpha_by_beta": 0.0016847839733506176, #0.0016847839733506176 #0.2934482659380352
                   "beta": 8356.965862561136, #8356.965862561136 #5307.693062806705
                   "gamma": 0.008267617243376338, #0.008267617243376338 #0.2990065533858563
                   "delta_by_gamma": 0.7723606265496351, #0.7723606265496351 #0.9149742827027212
                   "lr": 2.3764017040725935, #2.3764017040725935 #1.3390921599920842
                   "alpha_inc": 0}]
    else: #sean_choice, diventra_choice, and yuanhang_choice
        #[10, 20, 30] 3SAT
        '''params = [{"alpha_by_beta": 0.5924294731644022,
                   "beta": 3.377467133778823,
                   "gamma": 0.02352653872127089,
                   "delta_by_gamma": 0.8944393148402235,
                   "zeta": 7.940151660134392e-05,
                   "lr": 1.0,
                   "alpha_inc": 0}]'''
        #[50, 70, 90] 3SAT
        '''params = [{"alpha_by_beta": 0.7930523072512615,
                   "beta": 9.944653030715514,
                   "gamma": 0.046993375175827884,
                   "delta_by_gamma": 0.7925839173698284,
                   "zeta": 0.03730299264560244,
                   "lr": 1.0,
                   "alpha_inc": 0}]'''
        #[100, 120, 150] 3SAT
        '''params = [{"alpha_by_beta": 0.8629877746071739,
                   "beta": 9.899769142352335,
                   "gamma": 0.04845602220107946,
                   "delta_by_gamma": 0.617285687929963,
                   "zeta": 0.6184799259157281,
                   "lr": 1.0,
                   "alpha_inc": 0}]'''
        #[200, 300, 400] 3SAT
        '''params = [{"alpha_by_beta": 0.6064391671677962,
                   "beta": 9.8725033824004,
                   "gamma": 0.050292721988009256,
                   "delta_by_gamma": 0.45486557754853946,
                   "zeta": 0.5513270812050881,
                   "lr": 1.0,
                   "alpha_inc": 0}]'''
        #3SAT from previous (green dot) 3SAT tests #diventra_choice
        '''params = [{"alpha_by_beta": 0.628339734416646,
                   "beta": 1304.0063531328492,
                   "delta_by_gamma": 0.19512045666519495,
                   "gamma": 0.20977267203687128,
                   "lr": 3.2373030092084756,
                   "zeta": 0.050745372628985166}]'''
        #[10, 20, 30] 3R3X #diventra_choice
        '''params = [{"alpha_by_beta": 0.0876601792699714,
                   "beta": 56791.17970403271,
                   "gamma": 0.11247833010649586,
                   "delta_by_gamma": 0.8494656365136101,
                   "zeta": 0.0008807016987375788,
                   "lr": 1.0,
                   "alpha_inc": 0}]'''
        #[40, 60, 80] 3R3X #diventra_choice
        '''params = [{"alpha_by_beta": 0.9900045691821819,
                   "beta": 1.152386945113294,
                   "gamma": 0.09578333549454954,
                   "delta_by_gamma": 0.6963182254224278,
                   "zeta": 0.7518624914100748,
                   "lr": 1.0,
                   "alpha_inc": 0}]'''
        #[10, 20, 30] 3R3X (literal-specific memories) #sean_choice
        '''params = [{"alpha_by_beta": 0.7438250509121647,
                   "beta": 478.72915344318625,
                   "gamma": 0.04431169598526268,
                   "delta_by_gamma": 0.9788885953618904,
                   "zeta": 0.0059093190306424535,
                   "lr": 1.0,
                   "alpha_inc": 0}]'''
        #[40, 60, 80] 3R3X (literal-specific memories) #diventra_choice
        '''params = [{"alpha_by_beta": 0.057246190146539574,
                   "beta": 368.53637767096967,
                   "gamma": 0.32632951899661117,
                   "delta_by_gamma": 0.9056902101841241,
                   "zeta": 2.8551877398828256,
                   "lr": 1.0,
                   "alpha_inc": 0}]'''
        #[10, 20, 30] 3R3X, PT #diventra_choice
        params = [{"alpha_by_beta": 0.3,
                   "beta": 1.0,
                   "gamma": 0.2,
                   "delta_by_gamma": 0.2,
                   "zeta": 1,
                   "lr": 1.0,
                   "alpha_inc": 0}]
    simple = False
    batch = 100
    ns = [10, 20, 30]
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
                    plt.plot(steps, spin_traj_to_plot)
                plt.xlabel('Steps')
                plt.ylabel('Voltages')
                plt.tight_layout()
                plt.savefig(f'results/{prob_type}/Benchmark/{ns}/n{ns[i]}_batch{j}_v.png')
                plt.clf()

                for k in range(ns[i]): #iterate over n, could be up to k in range(len(xl_traj[i][0][j])) = ns[i]
                    xl_traj_to_plot = [element[j][k] for element in xl_traj[i]]
                    plt.plot(steps, xl_traj_to_plot)
                plt.xlabel('Steps')
                plt.ylabel('Long Term Memories')
                plt.tight_layout()
                plt.savefig(f'results/{prob_type}/Benchmark/{ns}/n{ns[i]}_batch{j}_xl.png')
                plt.clf()

                for k in range(ns[i]): #iterate over n, could be up to k in range(len(xs_traj[i][0][j])) = ns[i]
                    xs_traj_to_plot = [element[j][k] for element in xs_traj[i]]
                    plt.plot(steps, xs_traj_to_plot)
                plt.xlabel('Steps')
                plt.ylabel('Short Term Memories')
                plt.tight_layout()
                plt.savefig(f'results/{prob_type}/Benchmark/{ns}/n{ns[i]}_batch{j}_xs.png')
                plt.clf()

                for k in range(ns[i]): #iterate over n, could be up to k in range(len(C_traj[i][0][j])) = ns[i]
                    C_traj_to_plot = [element[j][k] for element in C_traj[i]]
                    plt.plot(steps, C_traj_to_plot)
                plt.xlabel('Steps')
                plt.ylabel('Clause Functions')
                plt.tight_layout()
                plt.savefig(f'results/{prob_type}/Benchmark/{ns}/n{ns[i]}_batch{j}_C.png')
                plt.clf()

                for k in range(ns[i]): #iterate over n, could be up to k in range(len(G_traj[i][0][j])) = ns[i]
                    G_traj_to_plot = [element[j][k] for element in G_traj[i]]
                    plt.plot(steps, G_traj_to_plot)
                plt.xlabel('Steps')
                plt.ylabel('Gradient Terms')
                plt.tight_layout()
                plt.savefig(f'results/{prob_type}/Benchmark/{ns}/n{ns[i]}_batch{j}_G.png')
                plt.clf()

                for k in range(ns[i]): #iterate over n, could be up to k in range(len(R_traj[i][0][j])) = ns[i]
                    R_traj_to_plot = [element[j][k] for element in R_traj[i]]
                    plt.plot(steps, R_traj_to_plot)
                plt.xlabel('Steps')
                plt.ylabel('Rigidity Terms')
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
