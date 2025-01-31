import os
import pickle
import time
import numpy as np
import torch
import torch.multiprocessing as mp
from model import DMM
from scipy.stats import linregress, skew, kurtosis
from tqdm import trange
import json
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from dmm_utils import run_dmm, avalanche_analysis_mp, avalanche_size_distribution
#from benchmark import param_scaling


torch.set_default_tensor_type(torch.cuda.FloatTensor
                              if torch.cuda.is_available()
                              else torch.FloatTensor)
mp.set_start_method('spawn', force=True)


# def prob_explore(pool_size):
#     return max(np.exp(-(pool_size - 5) / 10), 0.1)


# def prob_explore(min_loss):
#     return max(np.exp(2 * min_loss + 2), 0.1)


class Solver:
    def __init__(self, ns, cnf_files, prob_type, simple, max_step=int(1e8), steps=None, batch=100):
        self.best_eqn_choice = ''
        self.space = hp.choice('choice', [
            {
            'eqn_choice': 'sean_choice',
            'alpha_by_beta': hp.uniform('alpha_by_beta-sean', 0, 1),
            'beta': hp.loguniform('beta-sean', np.log(1e-5), np.log(1e5)),
            'gamma': hp.uniform('gamma-sean', 0, 0.5),
            'delta_by_gamma': hp.uniform('delta_by_gamma-sean', 0, 1),
            'zeta': hp.loguniform('zeta-sean', np.log(1e-5), np.log(1)),
            'dt_0': hp.loguniform('dt_0-sean', np.log(1e-2), np.log(1)),
            }])#,
            #{
            #'eqn_choice': 'diventra_choice',
            #'alpha_by_beta': hp.uniform('alpha_by_beta-diventra', 0, 1),
            #'beta': hp.loguniform('beta-diventra', np.log(1e-5), np.log(1e2)),
            #'gamma': hp.uniform('gamma-diventra', 0, 0.5),
            #'delta_by_gamma': hp.uniform('delta_by_gamma-diventra', 0, 1),
            #'zeta': hp.loguniform('zeta-diventra', np.log(1e-5), np.log(1e1)),
            #'dt_0': hp.loguniform('dt_0-diventra', np.log(1e-2), np.log(1)),
            #}])
        #CHANGE param_0 FOR EACH NEW VARIABLE NUMBER RANGE
        self.param_0 = {
            'alpha_by_beta': 0.5,
            'beta': 1,
            'gamma': 0.2,
            'delta_by_gamma': 0.2,
            'zeta': 0.0001,
            'dt_0': 1,
            'lr': 1.0,
            'alpha_inc': 0,
            'jump_thrs': 0,
            'jump_mag': 2.1
            }
        '''self.eqn_choice = eqn_choice
        if self.eqn_choice == 'rudy_simple':
            self.space = {
                'alpha': hp.loguniform('alpha', np.log(1e-2), np.log(1e4)),
                'delta': hp.uniform('delta', 0, 1),
                'chi': hp.loguniform('chi', np.log(1e-4), np.log(1e1)),
                'zeta': hp.loguniform('zeta', np.log(1e-4), np.log(1e1)),
                'dt_0': hp.loguniform('dt_0', np.log(1e-2), np.log(1)),
                # 'alpha_inc': hp.loguniform('alpha_inc', np.log(1e-6), np.log(1e-2)),
                # 'jump_thrs': hp.uniform('jump_thrs', 0, 1),
                # 'jump_mag': hp.uniform('jump_mag', 2, 2.5)
            }
            self.param_0 = {
                'alpha': 1,
                'delta': 0.5,
                'chi': 1e-1,
                'zeta': 1e-1,
                'dt_0': 1,
                'lr': 1.0,
                'alpha_inc': 0,
                'jump_thrs': 0,
                'jump_mag': 2.1
            }
        elif self.eqn_choice == 'rudy_choice':
            self.space = {
                'alpha_by_beta': hp.uniform('alpha_by_beta', 0, 1),
                'beta': hp.loguniform('beta', np.log(1e-2), np.log(1e4)),
                'delta': hp.uniform('delta', 0, 1),
                'chi': hp.loguniform('chi', np.log(1e-4), np.log(1e1)),
                'zeta': hp.loguniform('zeta', np.log(1e-4), np.log(1e1)),
                'dt_0': hp.loguniform('dt_0', np.log(1e-2), np.log(1)),
                # 'alpha_inc': hp.loguniform('alpha_inc', np.log(1e-6), np.log(1e-2)),
                # 'jump_thrs': hp.uniform('jump_thrs', 0, 1),
                # 'jump_mag': hp.uniform('jump_mag', 2, 2.5)
            }
            self.param_0 = {
                'alpha_by_beta': 1,
                'beta': 5,
                'delta': 0.5,
                'chi': 1e-1,
                'zeta': 1e-1,
                'dt_0': 1,
                'lr': 1.0,
                'alpha_inc': 0,
                'jump_thrs': 0,
                'jump_mag': 2.1
            }
        elif self.eqn_choice == 'zeta_zero' or self.eqn_choice == 'R_zero':
            self.space = {
                'alpha_by_beta': hp.uniform('alpha_by_beta', 0, 1),
                'beta': hp.loguniform('beta', np.log(1e-2), np.log(1e4)),
                'gamma': hp.uniform('gamma', 0, 0.5),
                'delta_by_gamma': hp.uniform('delta_by_gamma', 0, 1),
                'dt_0': hp.loguniform('dt_0', np.log(1e-2), np.log(1)),
                # 'alpha_inc': hp.loguniform('alpha_inc', np.log(1e-6), np.log(1e-2)),
                # 'jump_thrs': hp.uniform('jump_thrs', 0, 1),
                # 'jump_mag': hp.uniform('jump_mag', 2, 2.5)
            }
            self.param_0 = {
                'alpha_by_beta': 0.25,
                'beta': 4000,
                'gamma': 0.15,
                'delta_by_gamma': 0.15,
                'lr': 1.0,
                'dt_0': 1,
                'alpha_inc': 0,
                'jump_thrs': 0,
                'jump_mag': 2.1
            }
        else: #sean_choice, diventra_choice, and yuanhang_choice
            self.space = {
                'alpha_by_beta': hp.uniform('alpha_by_beta', 0, 1),
                'beta': hp.loguniform('beta', np.log(1), np.log(1e6)),
                'gamma': hp.uniform('gamma', 0, 0.5),
                'delta_by_gamma': hp.uniform('delta_by_gamma', 0, 1),
                'zeta': hp.loguniform('zeta', np.log(1e-5), np.log(1)),
                'dt_0': hp.loguniform('dt_0', np.log(1e-2), np.log(1)),
                # 'alpha_inc': hp.loguniform('alpha_inc', np.log(1e-6), np.log(1e-2)),
                # 'jump_thrs': hp.uniform('jump_thrs', 0, 1),
                # 'jump_mag': hp.uniform('jump_mag', 2, 2.5)
            }
            self.param_0 = {
                'alpha_by_beta': 0.5,
                'beta': 20,
                'gamma': 0.25,
                'delta_by_gamma': 0.2,
                'zeta': 1e-3,
                'dt_0': 1,
                'lr': 1.0,
                'alpha_inc': 0,
                'jump_thrs': 0,
                'jump_mag': 2.1
            }'''
        self.max_step = max_step
        self.ns = np.array(ns)
        self.cnf_files = cnf_files
        # self.dmm = DMM(self.cnf_files[0], param=self.param_0)
        self.file_pointer = 0
        self.avalanche_subprocesses = 10
        self.batch = batch
        self.avalanche_minibatch = int(np.ceil(self.batch / self.avalanche_subprocesses))
        self.pool_avalanches = mp.Pool(self.avalanche_subprocesses)
        if steps is None:
            # self.steps = int(np.ceil(100000 / np.sqrt(self.dmm.n_clause)))
            self.steps = int(5e3)
        else:
            self.steps = steps
        self.best_param = None
        self.best_metric = 1e6
        self.prob_type = prob_type
        self.simple = simple

    def run(self, max_evals=10000):
        save_interval = 100
        p = None
        for i in range(max_evals // save_interval):
            try:
                with open(f'results/{self.prob_type}/tpe_trials_{self.ns}.pkl', 'rb') as f:
                    tpe_trials = pickle.load(f)
            except FileNotFoundError:
                tpe_trials = Trials()
            max_evals_i = len(tpe_trials.trials) + save_interval
            best = fmin(fn=self.objective, space=self.space, algo=tpe.suggest,
                        max_evals=max_evals_i, trials=tpe_trials)
            with open(f'results/{self.prob_type}/tpe_trials_{self.ns}.pkl', 'wb') as f:
                pickle.dump(tpe_trials, f)
            #if p is not None:
                #p.terminate()
            #p = mp.Process(target=param_scaling, args=(self.best_param, i))
            #p.start()
            self.file_pointer += 1
            if self.file_pointer >= len(self.cnf_files):
                self.file_pointer = 0
            self.best_param = None
            self.best_metric = 1e6

    def avalanche_analysis(self, dmm, spin_traj, time_traj, time_window=0.5):
        cluster_sizes, memory_flag = avalanche_analysis_mp(spin_traj, time_traj, dmm.edges_var,
                                                           self.pool_avalanches, self.avalanche_minibatch,
                                                           self.avalanche_subprocesses, time_window)
        avalanche_stats = avalanche_size_distribution(cluster_sizes, f'training/{self.prob_type}/{self.ns}/{dmm.n_var}')
        if memory_flag:
            self.steps = self.steps // 2
        return avalanche_stats

    def metric(self, avalanche_stats, unsolved_stats, tts_stats):
        metric = 0

        #LRO Contribution
        lro_metric = 0
        '''# slope, intercept, r, avl_max = stats.transpose()
        target_stats = np.concatenate([np.tile(np.array([-1.5, 0, -0.98]), (len(self.ns), 1)),
                                       np.log10(self.ns).reshape(-1, 1)], axis=1)
        target_std = np.tile(np.array([1, 0.5, 0.02, 0.2]), (len(self.ns), 1))
        lro_metric = np.abs((avalanche_stats - target_stats) / (target_std)) - 0.5
        lro_metric = np.maximum(lro_metric, 0)
        lro_metric = np.exp(-np.sum(lro_metric ** 2, axis=1) / 2)
        lro_metric = np.sum(1 - lro_metric)
        lro_metric += 4 * target_stats[:, 0].std()
        print('LRO contribution: ' + str(metric))'''

        #UnSAT Contribution
        unsat_metric = 0
        '''unsat_mean = [ns_stats[0] for ns_stats in unsolved_stats]
        unsat_std = [ns_stats[1] for ns_stats in unsolved_stats]
        unsat_m_3 = [ns_stats[2] for ns_stats in unsolved_stats]
        unsat_moments_metric = np.mean(unsat_mean) - np.mean(unsat_std) + np.mean(unsat_m_3)
        print('UnSAT moments contribution: ' + str(unsat_moments_metric))
        metric += 3*unsat_moments_metric #incorporates mean and SD of distribution of unsat instances into metric'''

        #TTS Contribution
        tts_metric = 0
        #THESE ARE ONLY DISTRIBUTION STATS ON THE FIRST 51 INSTANCES TO BE SOLVED
        '''for i in range(len(tts_stats)): #len(tts_stats) = len(self.ns)
            tts_mean = tts_stats[i][0]
            #print('TTS Mean: ' + str(tts_mean))
            tts_var = tts_stats[i][1]
            #print('TTS Var: ' + str(tts_var))
            tts_skewness = tts_stats[i][2]
            tts_kurt = tts_stats[i][3]
            tts_metric += 1*tts_mean# + 1*(np.abs(tts_mean/tts_var - 9/(tts_skewness**2)) + np.abs(tts_mean/tts_var - 15/tts_kurt))'''

        instances_solved = [len(tts_stats[i]) for i in range(len(tts_stats))]
        print('Instances Solved: ' + str(instances_solved))
        best_percentile = min(instances_solved)
        #print('Best Percentile: ' + str(best_percentile))
        if best_percentile == 0:
            print('No instances solved at some n :(')
            tts_metric += 1e6
        elif best_percentile < 50:
            print('Too few instances solved at some n')
            tts_metric += 1e5/best_percentile
        else:
            small_n_bp_tts = np.log10(tts_stats[0][best_percentile-1])
            print('Small n Best Percentile TTS: 10^' + str(small_n_bp_tts))
            tts_slope = (np.log10(tts_stats[-1][best_percentile-1]) - np.log10(tts_stats[0][best_percentile-1]))/(np.log10(self.ns[-1]) - np.log10(self.ns[0]))
            print('TTS Slope: ' + str(tts_slope))
            tts_slope_upper = (np.log10(tts_stats[-1][best_percentile-1]) - np.log10(tts_stats[-2][best_percentile-1]))/(np.log10(self.ns[-1]) - np.log10(self.ns[-2]))
            #print('TTS Slope Upper: ' + str(tts_slope_upper))
            tts_slope_lower = (np.log10(tts_stats[1][best_percentile-1]) - np.log10(tts_stats[0][best_percentile-1]))/(np.log10(self.ns[1]) - np.log10(self.ns[0]))
            #print('TTS Slope Lower: ' + str(tts_slope_lower))
            tts_concavity = tts_slope_upper - tts_slope_lower
            print('TTS Concavity: ' + str(tts_concavity))
            tts_metric += small_n_bp_tts + 2*tts_slope + 0.5*tts_concavity

        metric += lro_metric + unsat_metric + tts_metric
        print('Total metric: ' + str(metric))
        return metric

    def objective(self, param):
        tts_stats = []
        unsolved_stats = []
        avalanche_stats = []
        eqn_choice = param['eqn_choice']
        for i, n in enumerate(self.ns):
            # cnf_file = self.cnf_files[i][self.file_pointer]
            dmm = DMM(self.cnf_files[i], simple=self.simple, batch=self.batch, param=param, eqn_choice=eqn_choice, prob_type=self.prob_type)
            transient = 100
            break_threshold = 0.5
            if self.simple:
                is_solved, solved_step, unsat_moments, spin_traj, time_traj, current_step = \
                    run_dmm(dmm, self.steps+transient, self.steps, transient, break_threshold)
            else:
                is_solved, solved_step, unsat_moments, spin_traj, time_traj,  xl_traj, xs_traj, C_traj, G_traj, R_traj, current_step = \
                    run_dmm(dmm, self.steps+transient, self.steps, transient, break_threshold)
            solved_step[~is_solved] = current_step
            #print('Solved Step: ' + str(solved_step))
            #print('Current Step: ' + str(current_step))

            #TTS Stats
            mask = solved_step != current_step
            true_solved_step = solved_step[mask]
            true_solved_step = sorted(true_solved_step.tolist())
            print('True Solved Steps: ' + str(true_solved_step))
            #THESE ARE ONLY DISTRIBUTION STATS ON THE FIRST 51 INSTANCES TO BE SOLVED (doesn't work in all cases)
            '''tts_mean = np.mean(true_solved_step)
            tts_var = np.var(true_solved_step)
            tts_skewness = skew(true_solved_step)
            tts_kurt = kurtosis(true_solved_step)
            #print('TTS Mean: ' + str(tts_mean))
            #print('TTS Var: ' + str(tts_var))
            #print('TTS Skew: ' + str(tts_skewness))
            #print('TTS Kurtosis: ' + str(tts_kurt))
            tts_stats.append([tts_mean, tts_var, tts_skewness, tts_kurt])'''

            tts_stats.append(true_solved_step)

            '''#UnSAT Stats
            unsat_moments_sample_size = (solved_step + 1 - transient)
            unsat_moments_sample_size[is_solved] -= 1
            unsat_moments_sample_size.clamp_(min=1)
            moment_1, moment_2, moment_3, moment_4 = unsat_moments.unbind(dim=1)
            unsat_mean = moment_1 / unsat_moments_sample_size
            unsat_var = moment_2 / unsat_moments_sample_size - unsat_mean ** 2
            unsat_std = unsat_var.sqrt()
            unsat_skewness = (((moment_3 - 3 * unsat_mean * moment_2) / unsat_moments_sample_size) + 2 * unsat_mean ** 3) / unsat_std ** 3
            unsat_kurt = ((moment_4 - 4 * unsat_mean * moment_3 + 6 * unsat_mean ** 2 * moment_2) / unsat_moments_sample_size
                        - 3 * unsat_mean ** 4) / unsat_std ** 4
            unsat_skewness[unsat_std < 1e-6] = 0
            #unsat_m_3 = torch.sign(unsat_skewness) * torch.abs(unsat_skewness).pow(1/3)
            unsat_kurt[unsat_std < 1e-6] = 3
            #unsat_m_4 = torch.sign(unsat_kurt) * torch.abs(unsat_kurt).pow(1/4)
            unsat_moments = torch.stack([unsat_mean, unsat_var, unsat_skewness, unsat_kurt], dim=1).cpu().numpy()
            unsolved_stats.append(unsat_moments)

            #Avalanche Stats
            avalanche_stat_i = self.avalanche_analysis(dmm, spin_traj, time_traj)
            avalanche_stats.append(avalanche_stat_i)'''

        unsolved_stats = np.array(unsolved_stats)
        avalanche_stats = np.array(avalanche_stats)

        metric = self.metric(avalanche_stats, unsolved_stats, tts_stats)
        if metric < self.best_metric:
            self.best_metric = metric
            self.best_param = param
            self.best_eqn_choice = eqn_choice

        stats = {
            'n': self.ns,
            'param': param,
            'unsat_moments': unsolved_stats,
            'avalanche_stats': avalanche_stats,
            'metric': metric,
        }
        with open(f'results/{self.prob_type}/stats_{self.ns}.p', 'ab') as f:
            pickle.dump(stats, f)
        self.file_pointer += 1
        if self.file_pointer >= len(self.cnf_files[0]):
            self.file_pointer = 0
        # self.reinitialize()

        return {
            'loss': metric,
            'param': param,
            'status': STATUS_OK
        }

    # def reinitialize(self):


