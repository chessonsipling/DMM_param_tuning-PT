# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 12:43:57 2021

Bidirectional logic operations used for building DMMs
Each "logic variable" is a continuous variable between [-1, 1]

@author: Yuanhang Zhang
"""

import numpy as np
import torch
import torch.nn as nn


def dec2bin(x, bits):
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(torch.get_default_dtype())


def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)


class OR(nn.Module):
    def __init__(self, input_idx, input_sign, simple):
        super(OR, self).__init__()
        self.shape_in = input_idx.shape
        self.n_sat = self.shape_in[-1]
        self.input_idx = input_idx
        self.input_sign = input_sign
        self.simple = simple

    def init_memory(self, v):
        batch0 = v.shape[0]
        input = v[torch.arange(batch0).view((batch0, 1, 1)),
            self.input_idx]
        input = input * self.input_sign
        C = torch.max(input, dim=-1)[0]
        C = (1 - C) / 2

        self.xl = nn.Parameter(torch.ones(self.shape_in[:-1]))
        self.xs = nn.Parameter(C)
        self.xl.grad = torch.zeros_like(self.xl)
        self.xs.grad = torch.zeros_like(self.xs)
        self.alpha_multiplier = torch.ones_like(self.xl)

    def calc_grad(self, v, param, eqn_choice):
        # param: (n_param, batch)
        # batch = v.shape[0]
        batch, n_clause, n_sat = self.shape_in
        v_input = v[torch.arange(batch).view(batch, 1, 1),
        self.input_idx]
        v_input = v_input * self.input_sign  # (batch, n_clause, n_sat)

        if eqn_choice == 'rudy_simple':
            alpha, delta, chi, zeta, _ = param
        elif eqn_choice == 'rudy_choice':
            alpha_by_beta, beta, delta, chi, zeta, _ = param
        elif eqn_choice == 'zeta_zero' or eqn_choice == 'R_zero':
            alpha_by_beta, beta, gamma, delta_by_gamma, _ = param
        else:
            alpha_by_beta, beta, gamma, delta_by_gamma, zeta, _ = param
        epsilon = 1e-3

        v_top, v_top_idx = torch.topk(v_input, 2, dim=-1)


        ###################################################################################
        #SAT
        v_top = (1 - v_top) / 2

        C = v_top[:, :, 0]
        G = C.unsqueeze(-1).repeat(1, 1, n_sat)
        G[torch.arange(batch).reshape(batch, 1),
        torch.arange(n_clause).reshape(1, n_clause),
        v_top_idx[:, :, 0]] = v_top[:, :, 1]
        G *= (self.xl * self.xs).unsqueeze(-1)

        R = torch.zeros(batch, n_clause, n_sat)
        R[torch.arange(batch).reshape(batch, 1),
        torch.arange(n_clause).reshape(1, n_clause),
        v_top_idx[:, :, 0]] = C

        if eqn_choice == 'zeta_zero':
            R *= (1 - self.xs).unsqueeze(-1)
        elif eqn_choice == 'sean_choice':
            R *= ((1 + zeta * self.xl) * (1 - self.xs)).unsqueeze(-1)
        elif eqn_choice == 'diventra_choice':
            R *= ((zeta * self.xl) * (1 - self.xs)).unsqueeze(-1)
        elif eqn_choice == 'yuanhang_choice':
            R *= ((zeta * torch.log(self.xl)) * (1 - self.xs)).unsqueeze(-1)

        dv = -(G + R) * self.input_sign
        #Linear x_{l,m}
        #dxl = -(alpha_by_beta * beta * self.alpha_multiplier * (C - delta_by_gamma*gamma))
        #Linear growth, exponential decay x_{l,m}
        dxl = torch.where(C >= delta_by_gamma*gamma, -(alpha_by_beta * beta * self.alpha_multiplier * (C - delta_by_gamma*gamma)), -(10 * (C - delta_by_gamma*gamma) * (self.xl - 1)))
        #Exponential x_{s,m}
        #dxs = -(beta * (self.xs + epsilon) * (C - gamma))
        #Linear x_{s, m}
        dxs = -(beta * (C - gamma))
        ###################################################################################


        ###################################################################################
        #XORSAT
        '''C = (1 - torch.prod(v_input, dim=-1)) / 2
        G_below = torch.ones(batch, n_clause, n_sat, dtype=v_input.dtype)
        G_above = torch.ones(batch, n_clause, n_sat, dtype=v_input.dtype)

        for i in range(1, n_sat):
            G_below[:, :, i] = G_below[:, :, i - 1] * v_input[:, :, i - 1]
    
        for i in range(n_sat - 2, -1, -1):
            G_above[:, :, i] = G_above[:, :, i + 1] * v_input[:, :, i + 1]
    
        G = G_below * G_above * self.input_sign
        #G = (-v_input + G_below * G_above)/2

        if eqn_choice == 'rudy_simple':
            R = torch.zeros(batch, n_clause, n_sat)
            dv = -(chi * G - (zeta * self.xs).unsqueeze(-1) * v_input * self.input_sign)
            dxs = -(alpha * (C - delta))
            dxl = 0
        elif eqn_choice == 'rudy_choice':
            R = torch.zeros(batch, n_clause, n_sat)
            dv = -(chi * G - (zeta * self.xs).unsqueeze(-1) * v_input * self.input_sign)
            dxs = -((beta * C) - self.xl)
            dxl = -((alpha_by_beta * beta * self.xs) - delta)
        elif eqn_choice == 'R_zero' or eqn_choice == 'zeta_zero' or eqn_choice == 'sean_choice' or eqn_choice == 'diventra_choice' or eqn_choice == 'yuanhang_choice':
            G *= (self.xl * self.xs / 2).unsqueeze(-1)

            if eqn_choice != 'R_zero':
                R_1 = (1 - v_input)/2
                R_2 = -(1 + v_input)/2

                voltages_to_make_satisfied = G_below * G_above >= 0
                voltages_to_make_unsatisfied = G_below * G_above < 0

                R = (R_1*voltages_to_make_satisfied + R_2*voltages_to_make_unsatisfied)
                #R = torch.zeros(batch, n_clause, n_sat)
                #R += v_input
            else:
                R = torch.zeros(batch, n_clause, n_sat)

            if eqn_choice == 'zeta_zero':
                R *= (1 - self.xs).unsqueeze(-1)
            elif eqn_choice == 'sean_choice':
                R *= ((1 + zeta * self.xl) * (1 - self.xs)).unsqueeze(-1)
            elif eqn_choice == 'diventra_choice':
                R *= ((zeta * self.xl) * (1 - self.xs)).unsqueeze(-1)
            elif eqn_choice == 'yuanhang_choice':
                R *= ((zeta * torch.log(self.xl)) * (1 - self.xs)).unsqueeze(-1)

            dv = -(G + R) * self.input_sign
            dxl = -(alpha_by_beta * beta * self.alpha_multiplier * (C - delta_by_gamma*gamma))
            dxs = -(beta * (self.xs + epsilon) * (C - gamma))'''
        ###################################################################################
        

        if v.grad is None:
            v.grad = torch.zeros_like(v)
        if self.xl.grad is None:
            self.xl.grad = torch.zeros_like(self.xl)
        if self.xs.grad is None:
            self.xs.grad = torch.zeros_like(self.xs)
        # v.grad.data.index_add_(1, self.input_idx.reshape(-1), dv.reshape(batch, -1))
        v.grad.data.scatter_add_(1, self.input_idx.reshape(batch, -1), dv.reshape(batch, -1))
        self.xl.grad.data += dxl
        self.xs.grad.data += dxs

        if self.simple:
            return C
        else:
            return C, G, R, self.xl, self.xs

    def calc_C(self, v): ###only valid for SAT problems (not XORSAT)
        batch0 = v.shape[0]
        input = v[torch.arange(batch0).view((batch0, 1, 1)), self.input_idx]
        input = input * self.input_sign
        C = torch.max(input, dim=-1)[0]
        C = (1 - C) / 2
        return C

    def clamp(self, max_xl):
        self.xl.data.clamp_(1, max_xl)
        self.xs.data.clamp_(0, 1)
