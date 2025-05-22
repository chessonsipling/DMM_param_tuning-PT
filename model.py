# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 12:52:05 2021

@author: Yuanhang Zhang
"""

import numpy as np
import torch
import torch.nn as nn
from operators import OR
from dataset import import_data
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # for running on server without GUI


#Defines DMM class
class DMM(nn.Module):
    def __init__(self, filenames, simple=True, batch=100, param=None, plot_graph=False, eqn_choice='sean_choice', prob_type='3SAT'):
        super().__init__()
        # This version cannot deal with mixed SAT, where each clause may have different number of literals
        #It does work with general XORSAT instances
        self.clause_idx = []
        self.clause_sign = []
        self.n_var = None
        self.n_clause = None
        self.batch = len(filenames)
        assert self.batch == batch, 'Batch size does not match the number of files'
        for filename in filenames:
            clause_idx, clause_sign, n_var, n_clause, n_sat, n_sat_count = import_data(filename)
            if self.n_var is None:
                self.n_var = n_var
                self.n_clause = n_clause
                self.n_sat = n_sat
                self.n_sat_count = n_sat_count
            else:
                assert self.n_var == n_var, 'Number of variables does not match'
                assert self.n_clause == n_clause, 'Number of clauses does not match'
            self.clause_idx.append(clause_idx[0])
            self.clause_sign.append(clause_sign[0])
        self.clause_idx = torch.stack(self.clause_idx, dim=0)
        self.clause_sign = torch.stack(self.clause_sign, dim=0)
        # if self.n_sat[0] == 1:
        #     self.one_sat_idx = self.clause_idx[0].squeeze(1)
        #     self.one_sat_sign = self.clause_sign[0].squeeze(1)
        #     self.clause_idx = self.clause_idx[1:]
        #     self.clause_sign = self.clause_sign[1:]
        #     self.n_clause -= len(self.one_sat_idx)
        # else:
        self.one_sat_sign = None
        self.one_sat_idx = None
        self.plot_graph = plot_graph
        if self.plot_graph:
            self.graph = nx.Graph()
            v_nodes = list(range(self.n_var))
            c_nodes = list(range(self.n_var, self.n_var + self.n_clause))
            self.graph.add_nodes_from(v_nodes)
            self.graph.add_nodes_from(c_nodes)
            c_node_pointer = self.n_var
            for i in range(len(self.clause_idx)):
                nodes_0 = torch.arange(c_node_pointer, c_node_pointer + self.clause_idx[i].shape[0])\
                    .unsqueeze(1).expand_as(self.clause_idx[i]).reshape(-1).detach().cpu().numpy()
                nodes_1 = self.clause_idx[i].reshape(-1).detach().cpu().numpy()
                self.graph.add_edges_from(zip(nodes_0, nodes_1))
                c_node_pointer += self.clause_idx[i].shape[0]
            pos = nx.spring_layout(self.graph)
            nx.set_node_attributes(self.graph, pos, 'pos')
        self.eqn_choice = eqn_choice
        self.prob_type = prob_type
        self.simple = simple

        # self.batch = int(np.ceil(1e6 / self.n_clause))
        # self.batch = batch
        #Provides some default parameters (these will be overwritten if particular params are provided as argyments to the DMM)
        if self.eqn_choice == 'rudy_simple':
            self.param = {
                'alpha': 1,
                'delta': 0.5,
                'chi': 1e-1,
                'zeta': 1e-1,
                'dt_0': 1.0,
                'alpha_inc': 0,
                'jump_thrs': 0,
                'jump_mag': 2.1
            }
        elif self.eqn_choice == 'rudy_choice':
            self.param = {
                'alpha_by_beta': 0.25,
                'beta': 5,
                'delta': 0.5,
                'chi': 1e-1,
                'zeta': 1e-1,
                'dt_0': 1.0,
                'alpha_inc': 0,
                'jump_thrs': 0,
                'jump_mag': 2.1
            }
        elif self.eqn_choice == 'zeta_zero' or self.eqn_choice == 'R_zero':
            self.param = {
                'alpha_by_beta': 0.25,
                'beta': 4000,
                'gamma': 0.15,
                'delta_by_gamma': 0.15,
                'dt_0': 1.0,
                'alpha_inc': 0,
                'jump_thrs': 0,
                'jump_mag': 2.1
            }
        else: #sean_choice, diventra_choice, and yuanhang_choice
            self.param = {
                'alpha_by_beta': 0.5,
                'beta': 20,
                'gamma': 0.25,
                'delta_by_gamma': 0.2,
                'zeta': 1e-3,
                'dt_0': 1.0,
                'alpha_inc': 0,
                'jump_thrs': 0,
                'jump_mag': 2.1
            }
        if param is not None:
            self.param.update(param)
        # self.lr = self.param['lr']
        #self.lr = 50 / self.n_var #<<<Yuanhang's learning rate
        self.lr = 1.0 #<<<Sean's learning rate
        self.max_xl = 1e4 * self.n_var

        self.ORs = nn.ModuleList()
        self.ORs.append(OR(self.clause_idx, self.clause_sign, self.simple))
        # for i in range(len(self.clause_idx)):
        #     OR_i = OR(self.clause_idx[i], self.clause_sign[i])
        #     self.ORs.append(OR_i)
            
        #Initializes voltages to random values between -1 and 1 (SELECT ONLY ONE)
        '''self.v = nn.Parameter(2 * torch.rand(self.batch, self.n_var) - 1)'''
        #Initializes voltages to the solution, if such a solution has already been found previously (SELECT ONLY ONE)
        input_voltage = torch.tensor([])
        for i in range(self.batch):
            try:
                with open(f'results/{self.prob_type}/Benchmark/n{self.n_var}_solution_{i:04d}.txt', 'r') as file:
                    solution = file.readlines()[0]
                print(f'Instance {i} already solved: {solution}')
                batch_input_voltage = torch.tensor([2*int(voltage)-1 for voltage in solution])
            except FileNotFoundError:
                print(f'Instance {i} not yet solved')
                batch_input_voltage = 2 * torch.rand(self.n_var) - 1
            finally:
                batch_input_voltage = torch.unsqueeze(batch_input_voltage, 0)
                input_voltage = torch.cat((input_voltage, batch_input_voltage), 0)
        self.v = nn.Parameter(input_voltage)
    
        self.v.grad = torch.zeros_like(self.v)

        for OR_i in self.ORs:
            OR_i.init_memory(self.v)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        '''try:
            self.compiled_step = torch.compile(self.step)
        except RuntimeError:
            self.compiled_step = self.step'''
        self.compiled_step = self.step

        clause_pointer = self.n_var
        clause_nodes = []
        for clause_idx_k in self.clause_idx:
            clause_nodes.append(torch.arange(clause_pointer, clause_pointer + clause_idx_k.shape[0]))
            clause_pointer += clause_idx_k.shape[0]

        self.edges_var = []
        for clause_idx_k in self.clause_idx:
            edges_var_k = []
            k_sat = clause_idx_k.shape[1]
            for i in range(k_sat):
                for j in range(i + 1, k_sat):
                    edges_var_k.append(torch.stack([clause_idx_k[:, i], clause_idx_k[:, j]], dim=1))
            edges_var_k = torch.cat(edges_var_k, dim=0)
            edges_var_k = torch.unique(edges_var_k, dim=0)
            self.edges_var.append(edges_var_k)
        # self.edges_var = torch.cat(self.edges_var, dim=0)
        # self.edges_var = torch.unique(self.edges_var, dim=0)

        # self.edges_var_clause = []
        # for k, clause_idx_k in enumerate(self.clause_idx):
        #     edges_k = torch.stack([clause_idx_k, clause_nodes[k].unsqueeze(1).expand_as(clause_idx_k)], dim=2)
        #     self.edges_var_clause.append(edges_k.reshape(-1, 2))
        # self.edges_var_clause = torch.cat(self.edges_var_clause, dim=0)
        # self.edges_var_clause = torch.unique(self.edges_var_clause, dim=0)

    #Reinitializes DMM logical variables
    def reinitialize(self):
        self.v.data = 2 * torch.rand(self.batch, self.n_var) - 1
        for OR_i in self.ORs:
            OR_i.init_memory(self.v)

    #Performs a single DMM integration step of duration dt
    def step(self):
        self.optimizer.zero_grad()
        if self.simple:
            unsat_clauses, dt = self.backward()
        else:
            unsat_clauses, v, xl, xs, dt, C, G, R = self.backward()

        # v_last = self.v.data.clone()
        self.optimizer.step()
        # self.apply_jump(v_last)
        self.clip_weights()
        #self.adjust_alpha()
        if self.simple:
            return unsat_clauses, dt
        else:
            return unsat_clauses, v, xl, xs, dt, C, G, R

    #Updates parameter values
    def update_param(self, param):
        self.param.update(param)

    #Calculates how much the relevant DMM variables/functions change after a single integration timestep
    def backward(self):
        if self.eqn_choice == 'rudy_simple':
            param = [self.param['alpha'], self.param['delta'], self.param['chi'],
                         self.param['zeta'], self.param['dt_0']]
        elif self.eqn_choice == 'rudy_choice':
            param = [self.param['alpha_by_beta'], self.param['beta'], self.param['delta'],
                         self.param['chi'], self.param['zeta'], self.param['dt_0']]
        elif self.eqn_choice == 'zeta_zero' or self.eqn_choice == 'R_zero':
            param = [self.param['alpha_by_beta'], self.param['beta'], self.param['gamma'],
                         self.param['delta_by_gamma'], self.param['dt_0']]
        else: #sean_choice, diventra_choice, and yuanhang_choice
            param = [self.param['alpha_by_beta'], self.param['beta'], self.param['gamma'],
                         self.param['delta_by_gamma'], self.param['zeta'], self.param['dt_0']]
        unsat_clauses = torch.zeros(self.batch, dtype=torch.int64)

        if not self.simple:
            v = torch.empty(0)
            xl = torch.empty(0)
            xs = torch.empty(0)
            C = torch.empty(0)
            G = torch.empty(0)
            R = torch.empty(0)
        for OR_i in self.ORs:
            #Calculates the change (gradient) in DMM variables/functions for each instance
            if self.simple:
                Ci = OR_i.calc_grad(self.v, param, self.eqn_choice)
            else:
                Ci, Gi, Ri, xli, xsi = OR_i.calc_grad(self.v, param, self.eqn_choice)
                v = torch.cat((v, self.v))
                xl = torch.cat((xl, xli))
                xs = torch.cat((xs, xsi))
                C = torch.cat((C, Ci))
                G = torch.cat((G, Gi))
                R = torch.cat((R, Ri))
            unsat_clauses += (Ci >= 0.5).sum(dim=1)

        #Sean's adaptive time step
        max_dv = torch.max(torch.abs(self.v.grad), dim=1)[0] + 1e-6
        dt = (self.param['dt_0'] / max_dv).clamp(1e-5, 0.1) #explicitly bounds dt (mostly to prevent it from becoming too large)
        for param in self.parameters():
            param.grad.data *= dt.view((-1, ) + (1, )*(len(param.shape)-1))
        if self.simple:
            return unsat_clauses, dt
        else:
            return unsat_clauses, v, xl, xs, dt, C, G, R

    #Clamps the logical variables between -1 and 1 (negative values correspond to the boolean FALSE, positive to the boolean TRUE)
    def clamp(self, max_xl):
        self.v.data.clamp_(-1, 1)
        if self.one_sat_idx is not None:
            self.v.data[:, self.one_sat_idx] = self.one_sat_sign

    #Clamps the long-term memories in each instance
    def clip_weights(self):
        self.clamp(self.max_xl)
        for OR_i in self.ORs:
            OR_i.clamp(self.max_xl)
