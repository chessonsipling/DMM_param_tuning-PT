import os
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.optimize import curve_fit
import plt_config
import math


def linear_func(x, m, b):
    return m * x + b

def fit_to_lin(x, y):
    opt_params, cov = curve_fit(linear_func, x, y)
    m_opt, b_opt = opt_params
    x_fit = np.linspace(np.min(x), np.max(x), 1000)
    y_fit = linear_func(x_fit, m_opt, b_opt)
    return x_fit, y_fit, m_opt


plt.rcParams["figure.figsize"] = (5, 6)

with open('experiment_results_YH.json', 'r') as file:
    data = json.load(file)

Ns = data['Ns']

sim_cim_data = data['results']['sim_cim']['medians'][:-2]
plt.scatter(Ns[:-2], sim_cim_data, color='red')
sim_cim_xfit, sim_cim_yfit, sim_cim_m_opt = fit_to_lin(Ns[:-2], np.log(sim_cim_data))
plt.plot(sim_cim_xfit, math.e**sim_cim_yfit, color='red', linestyle='dashed', label='Coherent\nIsing\nMachine')

gd_data = data['results']['gd']['medians'][:-1]
plt.scatter(Ns[:-1], gd_data, color='purple')
gd_xfit, gd_yfit, gd_m_opt = fit_to_lin(Ns[:-1], np.log(gd_data))
plt.plot(gd_xfit, math.e**gd_yfit, color='purple', linestyle='dashed', label='Gain-\nDissipative\nDynamics')

sbm_data = data['results']['sbm']['medians'][:-1]
plt.scatter(Ns[:-1], sbm_data, color='hotpink')
sbm_xfit, sbm_yfit, sbm_m_opt = fit_to_lin(Ns[:-1], np.log(sbm_data))
plt.plot(sbm_xfit, math.e**sbm_yfit, color='hotpink', linestyle='dashed', label='Simulated\nBifurcation\nMachine')

plt.legend(fontsize='13.5')
plt.yscale('log')
plt.xticks([5, 10, 15, 20, 25], ['5', '10', '15', '20', '25'], fontsize='28')
plt.yticks(fontsize='28')
plt.xlabel(r'$N$', fontsize='28')
plt.ylabel(r'$T_{median}$', fontsize='28')
plt.savefig(f'other_solver_inset.png', dpi=300, bbox_inches='tight')
plt.close()
