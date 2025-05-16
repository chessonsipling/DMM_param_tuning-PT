import os
import numpy as np
from numpy.fft import fft, fftfreq
import pickle
import json
import torch
import torch.multiprocessing as mp
from scipy.optimize import curve_fit
from model import DMM
from dmm_utils import run_dmm, avalanche_analysis, avalanche_analysis_mp, avalanche_size_distribution
mp.set_start_method('spawn', force=True)
import matplotlib.pyplot as plt
import plt_config
import math

import sys


def power_law_decay(x, a, b, c):
    return a * x**(-b) + c

def inverse_gaussian(x, mu, lamb):
    return np.sqrt(lamb/(2*math.pi*x**3)) * np.exp(-(lamb*(x - mu)**2)/(2*mu**2*x))

def inverse_gaussian_fit_to_peak(x, C_star, x_star, mu):
    lamb = (-3 * mu**2 * x_star) / (x_star**2 - mu**2)
    return C_star * (x_star / x)**(3/2) * np.exp((lamb/(2*mu**2))*(((x_star - mu)**2/x_star) - ((x - mu)**2/x)))

def exponential_decay(x, b):
    return b*np.exp(-b*x)

def log_normal(x, mu, sigma):
    return (1/(x*sigma*np.sqrt(2*math.pi))) * np.exp(-(np.log(x) - mu)**2/(2*sigma**2))


def tts_distribution(solved_step, prob_type, flattened_big_ns, n, break_t, name):

    plt.figure(figsize=(3.0, 1.75))
    
    #Extracts total number of instances solved
    with open(f'results/{prob_type}/Benchmark/{flattened_big_ns}/n_solved_{n}_{name}.txt', 'r') as f:
        n_solved = f.readlines()
    total_solved = sum([int(element.strip()) for element in n_solved])
    solved_step = np.sort(solved_step)
    max_percentile = total_solved/len(n_solved)
    percentile = 7 #51

    #Plotting vertical line at a particular percentile (set just above)
    '''tts_comparison = solved_step[int(len(solved_step)*(percentile/max_percentile)) - 1]
    plt.axvline(x = tts_comparison, color='red', linestyle='dashed')'''

    #Plotting with logarithmic axes
    '''log_bins = np.logspace(np.log10(min(solved_step)), np.log10(max(solved_step)), 50)
    prob, bins, patches = plt.hist(solved_step, bins=log_bins, density=True, color='blue', label=f'Proportion solved: {total_solved/500}')'''
    #Fitting with logarithmic axes
    '''bin_centers = np.array([(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)])
    popt, pcov = curve_fit(inverse_gaussian, prob, bin_centers, p0=[37000/total_solved, 10000, 400])
    plt.plot(bin_centers, inverse_gaussian(bin_centers, popt[0], popt[1], popt[2]), color='red', linestyle='dashed') # 37000/total_solved, 10000, 400'''

    #Plotting all data with linear axes
    prob, bins, patches = plt.hist(solved_step, bins=200, density=True, color='blue', label=f'Max Percentile:\n{percentile}')
    #Fitting with linear axes
    bin_centers = np.array([(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)])
    #popt, pcov = curve_fit(choose_a_distribution, prob, bin_centers) #play around with some heavy-tailed distributions! Try different p0 #[0.0005, 0.0005] for exp [0.08, 20.0], [0.0003, 0.0007] for exp [1.0, 0.08]
    #plt.plot(bin_centers, exponential_decay(bin_centers, 0.0005), color='red', linestyle='dashed') #[0.08, 20.0]
    #plt.plot(bin_centers, exponential_decay(bin_centers, 0.0005), color='red', linestyle='dashed') #[1.0, 0.08]
    #plt.plot(bin_centers, inverse_gaussian(bin_centers, 10000, 900), color='red', linestyle='dashed') #[0.08, 20.0]
    #plt.plot(bin_centers, inverse_gaussian(bin_centers, 300, 600), color='red', linestyle='dashed') #[1.0, 0.08] #OLD
    #plt.plot(bin_centers, inverse_gaussian(bin_centers, 1000, 500), color='red', linestyle='dashed') #[3.0, 1.0]
    plt.plot(bin_centers, inverse_gaussian(bin_centers, 1150, 4400), color='red', linestyle='dashed') #[20.0, 0.08]
    #plt.plot(bin_centers, inverse_gaussian(bin_centers, 20000, 1000), color='red', linestyle='dashed') #[20.0, 20.0]
    #plt.plot(bin_centers, log_normal(bin_centers, 6.15, 1.25), color='red', linestyle='dashed') #[3.0, 1.0]
    #plt.plot(bin_centers, log_normal(bin_centers, 7.30, 1.30), color='red', linestyle='dashed') #[20.0, 20.0]

    #Fits (when y-axis was "Counts"; additional first variable is from lack of normalization)
    '''#plt.plot(bin_centers, inverse_gaussian(bin_centers, 370000, 2100, 400), color='red', linestyle='dashed') #[3.0, 1.0]
    #plt.plot(bin_centers, inverse_gaussian(bin_centers, 600000, 100000, 750), color='red', linestyle='dashed') #[20.0, 20.0]
    #plt.plot(bin_centers, inverse_gaussian(bin_centers, 1400000, 1000000, 2000), color='red', linestyle='dashed') #[20.0, 20.0] w/ smaller dt_0
    #plt.plot(bin_centers, inverse_gaussian_fit_to_peak(bin_centers, 512, 156.25, 1200), color='red', linestyle='dashed') #[3.0, 1.0]
    #plt.plot(bin_centers, inverse_gaussian_fit_to_peak(bin_centers, 288, 231.5, 1000000), color='red', linestyle='dashed') #[20.0, 20.0]
    #plt.plot(bin_centers, inverse_gaussian_fit_to_peak(bin_centers, 188, 737.5, 1000000), color='red', linestyle='dashed') #[20.0, 20.0] w/ smaller dt_0'''

    plt.xlim(0, solved_step[int(len(solved_step)*(percentile/max_percentile)) - 1]) #median_tts) #50000)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel(r'Solution Step $T$')
    plt.ylabel(r'$P(T)$')
    plt.legend(fontsize='10')
    #plt.savefig(f'results/{prob_type}/Benchmark/{flattened_big_ns}/tts_{n}_{name}_no_y_lim.png', dpi=300, bbox_inches='tight')
    #plt.ylim(0, 2.5e-5)
    plt.savefig(f'results/{prob_type}/Benchmark/{flattened_big_ns}/tts_{n}_{name}_ig.png', dpi=300, bbox_inches='tight')
    plt.close()

def param_scaling(param, name, eqn_choice, prob_type, batch, ns, simple, flattened_big_ns, last_iteration, time_window, max_step):
    avalanche_subprocesses = 5
    avalanche_minibatch = int(np.ceil(batch / avalanche_subprocesses))
    pool = mp.Pool(avalanche_subprocesses)

    # print(param)
    with open(f'results/{prob_type}/Benchmark/{flattened_big_ns}/params_{ns}_{name}.json', 'w') as f:
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
        save_steps = 6000 #5900
        transient = 0 #100
        break_t = 0.5 #0.85, 0.75, 0.60, and 0.55 for optimal, bigger_beta, smaller_beta, smaller_zeta (when extracting TTS distrbs)
        # max_step = int(n ** 4 / 100)
        # max_step = save_steps + transient
        '''if simple:
            is_solved, solved_step, unsat_moments, spin_traj_n, time_traj_n, step = \
                run_dmm(dmm, max_step, simple, save_steps, transient, break_threshold=break_t)
        else:
            is_solved, solved_step, unsat_moments, spin_traj_n, time_traj_n, v_traj_n, xl_traj_n, xs_traj_n, C_traj_n, G_traj_n, R_traj_n, dt_traj_n, step = \
                run_dmm(dmm, max_step, simple, save_steps, transient, break_threshold=break_t)
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
            else step * (dmm.batch + 1) / (2 * n_solved + 1)'''
        #Extracts avalanche size distributions
        # cluster_size, out_of_memory_flag = avalanche_analysis(spin_traj, time_traj, dmm.edges_var)
        '''cluster_size, out_of_memory_flag = avalanche_analysis_mp(spin_traj_n, time_traj_n, dmm.edges_var, pool,
                                                                 avalanche_minibatch, avalanche_subprocesses, time_window)
        with open(f'results/{prob_type}/Benchmark/{flattened_big_ns}/cluster_sizes_{n}_{name}_{time_window}.txt', 'a') as f:
            for cluster in cluster_size:
                f.write(f'{cluster}\n')'''
        #Extract number of anti_instantons
        '''with open(f'results/{prob_type}/Benchmark/{flattened_big_ns}/anti_instantons_{n}_{name}.txt', 'a') as f:
            f.write(f'{anti_instantons}\n')'''
        #Extracts TTS distributions
        '''with open(f'results/{prob_type}/Benchmark/{flattened_big_ns}/tts_{n}_{name}.txt', 'a') as f:
            for tts in solved_step:
                if tts < max_step:
                    f.write(f'{tts}\n')'''
        #Keeps track of n_solved over all batches
        '''with open(f'results/{prob_type}/Benchmark/{flattened_big_ns}/n_solved_{n}_{name}.txt', 'a') as f:
            f.write(f'{n_solved}\n')'''
        if last_iteration == True:
            print('Last Iteration!')
            #Plots avalanche size distributions
            '''with open(f'results/{prob_type}/Benchmark/{flattened_big_ns}/cluster_sizes_{n}_{name}_{time_window}.txt', 'r') as f:
                cluster_size = np.array([float(element.strip()) for element in f.readlines()])
            avalanche_stats = avalanche_size_distribution(cluster_size, f'results/{prob_type}/Benchmark/{flattened_big_ns}/{name}_{n}_{time_window}')'''
            #Plots TTS distributions
            with open(f'results/{prob_type}/Benchmark/{flattened_big_ns}/tts_{n}_{name}.txt', 'r') as f:
                solved_step = np.array([float(element.strip()) for element in f.readlines()])
            tts_distribution(solved_step, prob_type, flattened_big_ns, n, break_t, name + '')
        '''stats = {
            'n': n,
            'is_solved': is_solved,
            'solved_step': solved_step,
            'median_step': median_step,
            'unsat_moments': unsat_moments,
            'avalanche_stats': avalanche_stats
        }
        pickle.dump(stats, open(f'results/{prob_type}/Benchmark/{flattened_big_ns}/stats_{n}_{name}.pkl', 'wb'))'''

        '''with open(f'results/{prob_type}/Benchmark/{flattened_big_ns}/steps_{name}.txt', 'a') as f:
            f.write(f'{n} {median_step}\n')
        print(f'N = {n} Done')'''

    if not simple:
        return spin_traj, time_traj, v_traj, xl_traj, xs_traj, C_traj, G_traj, R_traj, dt_traj


eqn_choice = 'sean_choice' #sys.argv[1] #eqn_choice can ONLY take on the values 'sean_choice', 'diventra_choice', 'yuanhang_choice', and 'zeta_zero' (and 'R_zero', 'rudy_choice', or 'rudy_simple' for XORSAT)
prob_type = '3SAT' #sys.argv[2] #prob_type can ONLY take on the values '3SAT', '3R3X', OR '5R5X'

if __name__ == '__main__':
    __spec__ = None
    mp.set_start_method('spawn', force=True)

    simple = True
    batch = 100
    #big_ns = np.array([[10], [20], [30], [40], [50], [60], [70], [80], [90], [100], [110], [120], [130], [140], [150], [160], [170], [180], [190], [200], [210], [220], [230], [240], [250], [260], [270], [280], [290], [300], [350], [400], [450], [500], [550], [600], [650], [700], [750], [800], [850], [900], [950]])
    #big_ns = np.array([[10], [20], [30], [40], [50], [60], [80], [100], [120], [150], [180], [210], [250], [300], [350], [400], [450], [500], [600], [700], [900]])
    big_ns = np.array([[100]])
    #big_ns = np.array([[10], [20], [30]])
    num_iterations = 1 #100
    tag = '_new_tts_1e-5_clamp'
    max_step = int(1e4)
    '''param_list = [0.02, 0.08, 0.2, 0.5, 0.8, 1.0, 1.5, 3, 6, 20, 50] #[0.08, 0.5, 1.0, 3.0, 20.0]
    total_param_list = []
    for item in param_list:
        item = [item] * len(param_list)
        total_param_list.append(list(zip(item, param_list)))
    #total_param_list = total_param_list[:2]
    #total_param_list = total_param_list[2:5]
    #total_param_list = total_param_list[5:8]
    #total_param_list = total_param_list[8:]
    total_param_list = [element for sublist in total_param_list for element in sublist]'''
    '''param_list = total_param_list  #[[1, 10], [1, 0.1], [10, 1], [0.1, 1], [1, 1]]
    time_window_list = [0.001, 0.002, 0.0035, 0.006, 0.01, 0.02, 0.035, 0.06, 0.1, 0.2, 0.35, 0.6, 1.0]
    total_param_list = []
    for item in param_list:
        item = [tuple(item)] * len(time_window_list)
        total_param_list.append(list(zip(item, time_window_list)))
    total_param_list = [element for sublist in total_param_list for element in sublist]
    total_param_list = [(a, b, c) for (a, b), c in total_param_list]'''
    #total_param_list = [[0.08, 0.08, 0.35], [0.08, 0.5, 0.35], [0.08, 1.0, 0.35], [0.08, 3.0, 0.6], [0.08, 20.0, 1.0],
                        #[0.5, 0.08, 0.035], [0.5, 0.5, 0.035], [0.5, 1.0, 0.035], [0.5, 3.0, 0.06], [0.5, 20.0, 0.35],
                        #[1.0, 0.08, 0.002], [1.0, 0.5, 0.0035], [1.0, 1.0, 0.02], [1.0, 3.0, 0.035], [1.0, 20.0, 0.2],
                        #[3.0, 0.08, 0.001], [3.0, 0.5, 0.006], [3.0, 1.0, 0.006], [3.0, 3.0, 0.01], [3.0, 20.0, 0.1],
                        #[20.0, 0.08, 0.0006], [20.0, 0.5, 0.001], [20.0, 1.0, 0.002], [20.0, 3.0, 0.006], [20.0, 20.0, 0.035]] #[[1, 10, 0.1], [1, 0.1, 0.0035], [10, 1, 0.0035], [0.1, 1, 0.35], [1, 1, 0.02]]
    total_param_list = [[20.0, 0.08, 0.0006]]

    flattened_big_ns = str(big_ns.flatten().tolist()) + tag
    result_dir = f'results/{prob_type}/Benchmark/{flattened_big_ns}'
    os.makedirs(result_dir, exist_ok=True)

    '''params = [{'alpha_by_beta': 0.45313481433413916,
                'beta': 78.88305080020264,
                'gamma': 0.3635604327568345,
                'delta_by_gamma': 0.21883211263830715,
                'zeta': 0.06294441488786634,
                'dt_0': 1,
                'time_window': 0.5,
                'lr': 1,
                'alpha_inc': 1}]''' #placeholder values

    for param_index in range(len(total_param_list)):
        last_iteration = False
        for iter in range(num_iterations): #, param_i in enumerate(params):
            if iter == num_iterations-1:
                last_iteration = True
            for ns in big_ns:
                if ns != big_ns[0]:
                    with open(f'{result_dir}/steps_{param_index}_{total_param_list[param_index][0]}_{total_param_list[param_index][1]}.txt', 'r') as f:
                        last_tts = f.readlines()[-1].split()[-1]
                    if float(last_tts) >= max_step:
                        break
                try:
                    with open(f'parameters/{prob_type}/[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]_optimal/optimal_param_{ns}.json', 'r') as f:
                        all_params = json.load(f)
                except:
                    print(f'No new parameters for n = {ns}')
                param_i = {'alpha_by_beta': 0.45313481433413916, #all_params['alpha_by_beta'], #0.45313481433413916
                            'beta': total_param_list[param_index][0]*78.83050800202636, #*all_params['beta'], #total_param_list[param_index][0]*78.83050800202636
                            'gamma': 0.3635604327568345, #all_params['gamma'], #0.3635604327568345
                            'delta_by_gamma': 0.21883211263830715, #all_params['delta_by_gamma'], #0.21883211263830715
                            'zeta': total_param_list[param_index][1]*0.06294441488786634, #*all_params['zeta'], #total_param_list[param_index][1]*0.06294441488786634
                            'dt_0': 0.01, #0.0898215588038146 for all "non-optimal" parameter choices, 0.06792404000784787 for "optimal" parameter choices
                            'time_window': total_param_list[param_index][2], #all_params['time_window'], #0.04095860826223421 was found to be "optimal" for small (up to n=100) sizes
                            'lr': 1.0, #all_params['lr'], #1.0
                            'alpha_inc': 0} #all_params['alpha_inc']} #0

                if simple:
                    param_scaling(param_i, str(param_index) + '_' + str(total_param_list[param_index][0]) + '_' + str(total_param_list[param_index][1]), eqn_choice, prob_type, batch, ns, simple, flattened_big_ns, last_iteration, param_i['time_window'], max_step)
                else:
                    spin_traj, time_traj, v_traj, xl_traj, xs_traj, C_traj, G_traj, R_traj, dt_traj = param_scaling(param_i, str(param_index), eqn_choice, prob_type, batch, ns, simple, flattened_big_ns, last_iteration, param_i['time_window'], max_step)

                    for i in range(len(ns)): #iterate over variable number, could be up to i in range(len(ns))
                        #steps = np.array(list(range(len(spin_traj[i]))))

                        for j in range(5): #iterate over batch, could be up to j in range(batch)
                            time_traj_to_plot = time_traj[i][j]
                            transient = 100

                            #Calculate power spectrum
                            '''time_traj_for_derivs = [(time_traj_to_plot[l+1] + time_traj_to_plot[l])/2 for l in range(len(time_traj_to_plot)-1)]
                            frequencies = np.fft.fftfreq(len(time_traj_for_derivs), d=np.mean([element[j] for element in dt_traj[i]]))
                            freq_to_plot = frequencies[:len(time_traj_for_derivs)//2]
                            total_spectrum_avg = np.zeros_like(frequencies)
                            mag_vderivs = np.zeros_like(frequencies)
                            for k in range(ns[i]): #iterate over v, could be up to k in range(ns[i])
                                v_traj_to_plot = [element[j][k] for element in v_traj[i]] #n, step, batch, v/xl/xs
                                v_deriv = [(v_traj_to_plot[l+1] - v_traj_to_plot[l])/(time_traj_to_plot[l+1] - time_traj_to_plot[l]) for l in range(len(time_traj_to_plot)-1)]
                                mag_vderivs += np.square(np.array(v_deriv))
                                #Individual power spectra
                                v_spectrum = np.abs(np.fft.fft(np.array(v_deriv)))**2
                                total_spectrum_avg += v_spectrum
                                if k < 10:
                                    plt.plot(freq_to_plot, v_spectrum[:len(time_traj_for_derivs)//2]) #only plots positivr frequencies
                                    #plt.yscale('log')
                                    plt.xlabel('Hz')
                                    plt.ylabel(r'$P(\dot{v}_i)$')
                                    #plt.legend()
                                    plt.tight_layout()
                                    plt.savefig(f'results/{prob_type}/Benchmark/{flattened_big_ns}/n{ns[i]}_batch{j}_power_spectrum{k}.png')
                                    plt.clf()
                            #Total Spectrum (avg)
                            total_spec_avg_to_plot = total_spectrum_avg[:len(time_traj_for_derivs)//2]
                            #popt, pcov = curve_fit(power_law_decay, freq_to_plot, total_spec_to_plot, p0=[100, 1, 0])
                            #a, b, c = popt
                            plt.plot(freq_to_plot, total_spec_avg_to_plot/k) #only plots positive frequencies
                            #plt.plot(freq_to_plot, power_law_decay(freq_to_plot, a, b, c), 'r-', label=f'f^-{b}')
                            #plt.yscale('log')
                            plt.xlabel('Hz')
                            plt.ylabel(r'$\frac{1}{N} \sum_i P(\dot{v}_i)$')
                            plt.legend()
                            plt.tight_layout()
                            plt.savefig(f'results/{prob_type}/Benchmark/{flattened_big_ns}/n{ns[i]}_batch{j}_power_spectrum_avg.png')
                            plt.clf()
                            #Total Spectrum (mag vdot)
                            mag_vderivs = mag_vderivs**(1/2)
                            total_spectrum_mag_vderivs = np.abs(np.fft.fft(np.array(mag_vderivs)))**2
                            total_spec_mag_vderivs_to_plot = total_spectrum_mag_vderivs[:len(time_traj_for_derivs)//2]
                            plt.plot(freq_to_plot, total_spec_mag_vderivs_to_plot)
                            #plt.yscale('log')
                            plt.xlabel('Hz')
                            plt.ylabel(r'$P(|\dot{v}|)$')
                            plt.legend()
                            plt.tight_layout()
                            plt.savefig(f'results/{prob_type}/Benchmark/{flattened_big_ns}/n{ns[i]}_batch{j}_power_spectrum_mag_vderivs.png')
                            plt.clf()'''
                        
                            #Plot relevant trajectories
                            for k in range(10): #iterate over v
                                v_traj_to_plot = [element[j][k] for element in v_traj[i]] #n, step, batch, v/xl/xs
                                #plt.plot(time_traj_to_plot, spin_traj_to_plot, label=f'{k}')
                                plt.plot(time_traj_to_plot, v_traj_to_plot)
                            plt.xlabel('Time')
                            plt.ylabel('Voltages')
                            #plt.legend()
                            plt.tight_layout()
                            plt.savefig(f'results/{prob_type}/Benchmark/{flattened_big_ns}/n{ns[i]}_batch{j}_v.png')
                            plt.xlim(left = time_traj_to_plot[transient], right=time_traj_to_plot[transient+int(len(time_traj_to_plot)/10)])
                            plt.savefig(f'results/{prob_type}/Benchmark/{flattened_big_ns}/n{ns[i]}_batch{j}_v_zoomed.png')
                            plt.clf()

                            for k in range(10): #iterate over n
                                xl_traj_to_plot = [element[j][k] for element in xl_traj[i]]
                                #plt.plot(time_traj_to_plot, xl_traj_to_plot, label=f'{k}')
                                plt.plot(time_traj_to_plot, xl_traj_to_plot)
                            plt.xlabel('Time')
                            plt.ylabel('Long Term Memories')
                            #plt.legend()
                            plt.tight_layout()
                            plt.savefig(f'results/{prob_type}/Benchmark/{flattened_big_ns}/n{ns[i]}_batch{j}_xl.png')
                            plt.xlim(left = time_traj_to_plot[transient], right=time_traj_to_plot[transient+int(len(time_traj_to_plot)/10)])
                            plt.savefig(f'results/{prob_type}/Benchmark/{flattened_big_ns}/n{ns[i]}_batch{j}_xl_zoomed.png')
                            plt.clf()

                            for k in range(10): #iterate over n
                                xs_traj_to_plot = [element[j][k] for element in xs_traj[i]]
                                #plt.plot(time_traj_to_plot, xs_traj_to_plot, label=f'{k}')
                                plt.plot(time_traj_to_plot, xs_traj_to_plot)
                            plt.xlabel('Time')
                            plt.ylabel('Short Term Memories')
                            #plt.legend()
                            plt.tight_layout()
                            plt.savefig(f'results/{prob_type}/Benchmark/{flattened_big_ns}/n{ns[i]}_batch{j}_xs.png')
                            plt.xlim(left = time_traj_to_plot[transient], right=time_traj_to_plot[transient+int(len(time_traj_to_plot)/10)])
                            plt.savefig(f'results/{prob_type}/Benchmark/{flattened_big_ns}/n{ns[i]}_batch{j}_xs_zoomed.png')
                            plt.clf()

                            for k in range(10): #iterate over n
                                C_traj_to_plot = [element[j][k] for element in C_traj[i]]
                                #plt.plot(time_traj_to_plot, C_traj_to_plot, label=f'{k}')
                                plt.plot(time_traj_to_plot, C_traj_to_plot)
                            plt.xlabel('Time')
                            plt.ylabel('Clause Functions')
                            #plt.legend()
                            plt.tight_layout()
                            plt.savefig(f'results/{prob_type}/Benchmark/{flattened_big_ns}/n{ns[i]}_batch{j}_C.png')
                            plt.xlim(left = time_traj_to_plot[transient], right=time_traj_to_plot[transient+int(len(time_traj_to_plot)/10)])
                            plt.savefig(f'results/{prob_type}/Benchmark/{flattened_big_ns}/n{ns[i]}_batch{j}_C_zoomed.png')
                            plt.clf()

                            for k in range(10): #iterate over n
                                for l in range(3): #iterates over 3 voltages in each clause
                                    G_traj_to_plot = [element[j][k][l] for element in G_traj[i]]
                                    #plt.plot(time_traj_to_plot, G_traj_to_plot, label=f'{k}, {l}')
                                    plt.plot(time_traj_to_plot, G_traj_to_plot)
                            plt.xlabel('Time')
                            plt.ylabel('Gradient Terms')
                            #plt.legend(ncol=3)
                            plt.tight_layout()
                            plt.savefig(f'results/{prob_type}/Benchmark/{flattened_big_ns}/n{ns[i]}_batch{j}_G.png')
                            plt.xlim(left = time_traj_to_plot[transient], right=time_traj_to_plot[transient+int(len(time_traj_to_plot)/10)])
                            plt.savefig(f'results/{prob_type}/Benchmark/{flattened_big_ns}/n{ns[i]}_batch{j}_G_zoomed.png')
                            plt.clf()

                            for k in range(10): #iterate over n
                                for l in range(3): #iterates over 3 voltages in each clause
                                    R_traj_to_plot = [element[j][k][l] for element in R_traj[i]]
                                    #plt.plot(time_traj_to_plot, R_traj_to_plot, label=f'{k}, {l}')
                                    plt.plot(time_traj_to_plot, R_traj_to_plot)
                            plt.xlabel('Time')
                            plt.ylabel('Rigidity Terms')
                            #plt.legend(ncol=3)
                            plt.tight_layout()
                            plt.savefig(f'results/{prob_type}/Benchmark/{flattened_big_ns}/n{ns[i]}_batch{j}_R.png')
                            plt.xlim(left = time_traj_to_plot[transient], right=time_traj_to_plot[transient+int(len(time_traj_to_plot)/10)])
                            plt.savefig(f'results/{prob_type}/Benchmark/{flattened_big_ns}/n{ns[i]}_batch{j}_R_zoomed.png')
                            plt.clf()

                            dt_traj_to_plot = [element[j] for element in dt_traj[i]]
                            #plt.plot(time_traj_to_plot, dt_traj_to_plot, label=f'{k}')
                            plt.plot(time_traj_to_plot, dt_traj_to_plot)
                            plt.axhline(0.1, color='red', linestyle='dashed')
                            plt.axhline(1e-5, color='red', linestyle='dashed')
                            plt.ylim(5e-6, min(10, max(dt_traj_to_plot)))
                            plt.yscale('log')
                            plt.xlabel('Time')
                            plt.ylabel('dt')
                            plt.tight_layout()
                            plt.savefig(f'results/{prob_type}/Benchmark/{flattened_big_ns}/n{ns[i]}_batch{j}_dt.png')
                            plt.clf()
