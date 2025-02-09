import pickle
import numpy as np
import torch
import torch.multiprocessing as mp
from model import DMM
from scipy.stats import skew, kurtosis
import json
from hyperopt import STATUS_OK
from dmm_utils import run_dmm, avalanche_analysis_mp, avalanche_size_distribution


torch.set_default_tensor_type(torch.cuda.FloatTensor
                              if torch.cuda.is_available()
                              else torch.FloatTensor)
mp.set_start_method('spawn', force=True)


#Initializes the Parallel Tempering (PT) device used to optimize parameters
class Solver_PT:
    def __init__(self, ns, cnf_files, prob_type, simple, replicas, max_step=int(1e8), steps=None, batch=100):
        self.best_eqn_choice = ''
        self.ns = np.array(ns)
        self.cnf_files = cnf_files
        # self.dmm = DMM(self.cnf_files[0], param=self.param_0)
        self.file_pointer = 0
        self.avalanche_subprocesses = 10
        self.batch = batch
        self.lower_T = lower_T
        self.upper_T = upper_T
        self.avalanche_minibatch = int(np.ceil(self.batch / self.avalanche_subprocesses))
        self.pool_avalanches = mp.Pool(self.avalanche_subprocesses)
        if steps is None:
            self.steps = int(5e3) #max number of allowable timesteps (arbitrary units, NOT necessarily of width dt) after transient
        else:
            self.steps = steps
        self.best_param = None
        self.best_metric = 1e6
        self.prob_type = prob_type
        self.simple = simple
        self.replicas = replicas

    #Runs the PT procedure
    def run(self, max_evals=10000):
        param_mask = [[0, 1],
                      [1e-5, 1e2],
                      [0, 0.5],
                      [0, 1],
                      [1e-5, 1e1],
                      [1e-2, 1],
                      [1.0, 1.0],
                      [0, 0],
                      [0, 0],
                      [2.1, 2.1]]
        starting_params = {'alpha_by_beta': 0.06805874059816672,
                           'beta': 4.850707467528604,
                           'gamma': 0.01108894541776315,
                           'delta_by_gamma': 0.5617086818983189,
                           'zeta': 0.0005372553919645686,
                           'dt_0': 1,
                           'lr': 1.0,
                           'alpha_inc': 0,
                           'jump_thrs': 0,
                           'jump_mag': 2.1} #initial values for alpha_by_beta, beta, gamma, delta_by_gamma, zeta
        current_replica_details = [self.objective(starting_params)] * self.replicas #initializes details for all replicas, which start at the same point in parameter space
        distrb_params = {'alpha_by_beta': 0.005,
                    'beta': 10,
                    'gamma': 0.003,
                    'delta_by_gamma': 0.003,
                    'zeta': 10,
                    'dt_0': 3,
                    'time_window': 3} #for draws from mixed gaussian and log-normal distributions with the given variances
    
        #For E<T> plot generation (CHOOSE ONLY ONE, SELECT CORRESPONDING OPTION IN benchmark.py)
        '''inverse_temps = np.geomspace(self.lower_T, self.upper_T, max_evals) #for <E(T)> plot generation (simulated annealing)'''
        #For normal PT (CHOOSE ONLY ONE, SELECT CORRESPONDING OPTION IN benchmark.py)
        inverse_temps = np.geomspace(self.lower_T, self.upper_T, self.replicas) #creates a list of inverse temps for each replica from 1/ln(mean(N)) to 10, spaced geometrically
    
        for i in range(max_evals):
            new_replica_details = [0] * self.replicas
            for j in range(self.replicas):
                new_param_values = [0] * len(starting_params)
                #Metropolis Update (w/ a multivariate normal as the generating distribution)
                #Determines a new set of parameters from the old ones, if they exist
                for k, key in enumerate(current_replica_details[j]['param'].keys()): #drawn from mixed gaussian & log-normal distributions
                    if key == 'alpha_by_beta' or key == 'gamma' or key == 'delta_by_gamma': #draw from gaussians
                        new_param_values[k] = np.random.normal(current_replica_details[j]['param'][key], np.sqrt(distrb_params[key]))
                    elif key == 'beta' or key == 'zeta' or key == 'dt_0' or key == 'time_window': #draw from log-normals
                        new_param_values[k] = np.random.lognormal(np.log(current_replica_details[j]['param'][key]), np.log(distrb_params[key]))
                new_param_values = [new_param_values[k] if (new_param_values[k] >= param_mask[k][0] and new_param_values[k] <= param_mask[k][1])
                                  else list(current_replica_details[j]['param'].values())[k] for k in range(len(new_param_values))] #verifies that params are within physical bounds
                new_params = dict(zip(list(current_replica_details[j]['param'].keys()), new_param_values))
                #Metropolis Update cont. (accepting/denying the proposed param values based on an Acceptance distribution)
                #A(x_j(t), x_j(t+dt), T_j)
                new_replica_details[j] = self.objective(new_params) #runs with new parameters and determines how effective they are (produces metric)

                #For E<T> plot generation (CHOOSE ONLY ONE, SELECT CORRESPONDING OPTION IN benchmark.py)
                '''acceptance_prob = min(1, np.exp(-1 * (new_replica_details[j]['loss'] - current_replica_details[j]['loss']) * inverse_temps[i]))'''
                #For normal PT (CHOOSE ONLY ONE, SELECT CORRESPONDING OPTION IN benchmark.py)
                acceptance_prob = min(1, np.exp(-1 * (new_replica_details[j]['loss'] - current_replica_details[j]['loss']) * inverse_temps[j])) #k_B = 1

                if acceptance_prob >= np.random.uniform(): #only updates the parameters if the probability of acceptance is high enough
                    current_replica_details[j] = new_replica_details[j]
                    print(f'New, accepted parameters: {new_params}!')
                else:
                    print('New parameters not accepted.')
                print(f'Replica {j} Simulated')
                if j > 0 and ((j % 2 == 1) and (i % 2 == 0)) or ((j % 2 == 0) and (i % 2 == 1)): #will not run for the case of a single replica (simulated annealing), and proposes swaps only between every other pair, alternating (Metropolis sweep 1: 0<->1, 2<-3>, ..., Sweep 2: 1<->2, 3<->4, ...)
                    #Parallel Tempering Update (proposes a single swap between adjacent layers)
                    #S(x_j(t), x_{j-1}(t), T_j, T_{j-1})
                    swap_acceptance_prob = min(1, np.exp((inverse_temps[j] - inverse_temps[j-1]) * (current_replica_details[j]['loss'] - current_replica_details[j-1]['loss'])))
                    if swap_acceptance_prob >= np.random.uniform():
                        current_replica_details[j], current_replica_details[j-1] = current_replica_details[j-1], current_replica_details[j] #swaps the parameters of neighboring replicas (NOTE:temps do not swap!)
                        print(f'Replicas {j} and {j-1} swapped!')
                    else:
                        print(f'Replicas {j} and {j-1} did not swap.')
                    #with open(f'training/{self.prob_type}/{self.flattened_big_ns}/swap_acceptance_prob_{self.ns}_{self.replicas}.txt', 'a') as f:
                        #f.write(f'{j},{swap_acceptance_prob}\n')

            #For E<T> plot generation (ONLY UNCOMMENT WITH CORRESPONDING OPTION IN benchmark.py IS SELECTED)
            '''with open(f'training/{self.prob_type}/{self.flattened_big_ns}/annealed_energy_{self.ns}.txt', 'a') as f:
                f.write(f'{inverse_temps[i]},{current_replica_details[j]['loss']}\n')'''

        print(f'Optimal params: {self.best_param}')
        print(f'Best metric: {self.best_metric}')

    #Collects statistics on avalanches, for training purpose
    def avalanche_analysis(self, dmm, spin_traj, time_traj, time_window):
        cluster_sizes, memory_flag = avalanche_analysis_mp(spin_traj, time_traj, dmm.edges_var,
                                                           self.pool_avalanches, self.avalanche_minibatch,
                                                           self.avalanche_subprocesses, time_window)
        avalanche_stats = avalanche_size_distribution(cluster_sizes, f'training/{self.prob_type}/{self.ns}/{dmm.n_var}')
        if memory_flag:
            self.steps = self.steps // 2
        return avalanche_stats

    def metric(self, avalanche_stats, unsolved_stats, tts_stats):
        metric = 0

        #LRO Contribution (currently not implemented in metric)
        lro_metric = 0
        '''target_stats = np.concatenate([np.tile(np.array([-1.5, 0, -0.98]), (len(self.ns), 1)),
                                       np.log10(self.ns).reshape(-1, 1)], axis=1)
        target_std = np.tile(np.array([1, 0.5, 0.02, 0.2]), (len(self.ns), 1))
        lro_metric = np.abs((avalanche_stats - target_stats) / (target_std)) - 0.5
        lro_metric = np.maximum(lro_metric, 0)
        lro_metric = np.exp(-np.sum(lro_metric ** 2, axis=1) / 2)
        lro_metric = np.sum(1 - lro_metric)
        lro_metric += 4 * target_stats[:, 0].std()
        print('LRO contribution: ' + str(lro_metric))'''

        #UnSAT Contribution (currently not implemented in metric)
        unsat_metric = 0
        '''unsat_mean = [ns_stats[0] for ns_stats in unsolved_stats]
        unsat_std = [ns_stats[1] for ns_stats in unsolved_stats]
        unsat_m_3 = [ns_stats[2] for ns_stats in unsolved_stats]
        unsat_moments_metric = np.mean(unsat_mean) - np.mean(unsat_std) + np.mean(unsat_m_3)
        print('UnSAT moments contribution: ' + str(unsat_moments_metric))
        metric += 3*unsat_moments_metric #incorporates mean and SD of distribution of unsat instances into metric'''

        #TTS Contribution
        tts_metric = 0
        #Extracts the y-intercept, slope, and concavity of the T(N) (time to solve a median of instances as a function of size) when plotted in log10
        #Uses this information to generate a metric (tries to minimize all 3 simultaneously)
        instances_solved = [len(tts_stats[i]) for i in range(len(tts_stats))]
        print('Instances Solved: ' + str(instances_solved))
        best_percentile = min(instances_solved)
        #print('Best Percentile: ' + str(best_percentile))
        if best_percentile == 0:
            print('No instances solved at some n :(')
            tts_metric += 1e6
        elif best_percentile < 45: #gives some leeway, in case especially small instances solve only 49/50 before stopping themselves
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
            tts_metric += small_n_bp_tts + 2*tts_slope + 0.5*tts_concavity #c_1=2 and c_2=0.5 were selected to prioritize slope, the y-intercept, then concavity

        metric += lro_metric + unsat_metric + tts_metric
        print('Total metric: ' + str(metric))
        return metric

    #Runs DMM 
    def objective(self, param):
        tts_stats = []
        unsolved_stats = []
        avalanche_stats = []
        eqn_choice = 'sean_choice' #param['eqn_choice'] #need to fix this so the multi-equation hyperopt functionality is restored
        for i, n in enumerate(self.ns):
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
            tts_stats.append(true_solved_step)

            #Collects some supplementary statistics (only necessary when avalanche_stats, unsolved_stats, and tts_stats contribute to the metric)
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
            avalanche_stat_i = self.avalanche_analysis(dmm, spin_traj, time_traj, param['time_window'])
            avalanche_stats.append(avalanche_stat_i)'''

        unsolved_stats = np.array(unsolved_stats)
        avalanche_stats = np.array(avalanche_stats)

        metric = self.metric(avalanche_stats, unsolved_stats, tts_stats)
        if metric < self.best_metric:
            self.best_metric = metric
            self.best_param = param
            self.best_eqn_choice = eqn_choice

        #Extracts statistics to a .p file
        stats = {
            'n': self.ns,
            'param': param,
            'unsat_moments': unsolved_stats,
            'avalanche_stats': avalanche_stats,
            'metric': metric,
        }
        with open(f'training/{self.prob_type}/{self.ns}/stats_{self.ns}.p', 'ab') as f:
            pickle.dump(stats, f)
        self.file_pointer += 1
        if self.file_pointer >= len(self.cnf_files[0]):
            self.file_pointer = 0

        return {
            'loss': metric,
            'param': param,
            'status': STATUS_OK
        }
