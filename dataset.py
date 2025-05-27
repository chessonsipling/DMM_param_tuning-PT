# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 16:07:55 2022

@author: Yuanhang Zhang
"""

import os
import numpy as np
import torch
import logging


#############################################################################################################################################################################################
#FREE PARAMETERS DURING DATA GENERATION
ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
num_instances = 100
#############################################################################################################################################################################################


#Defines the class used to generate 3SAT instances; batch gives number of unique instances, n is number of logical variables per instance
#r is the clause-to-variable ratio (r = 4.3 are the hardest instances)
class SAT_dataset:
    def __init__(self, batch, n, r=4.3, p0=0.08):
        self.batch = batch
        self.n = n
        self.r = r
        self.p0 = p0
        
    #Generates 3SAT instances via a Clause Distribution Control (CDC) procedure (Barthel et al. 2002)
    @torch.no_grad()
    def generate_instances(self, output_to_file=False, folder=None):
        batch = self.batch
        n = self.n
        r = self.r
        p0 = self.p0
        n_clause = int(n*r)
        planted_solution = 2*torch.randint(0, 2, [batch, n]) - 1

        # Generate a little bit more clauses than needed
        # Extra ~ mean + 10*std
        n_generation = int(n_clause + 3 * r + 10 * np.sqrt(3 * r))
        while True:
            clause_idx = torch.randint(0, n, (batch * n_generation, 3))
            is_equal = (clause_idx[:, 0] == clause_idx[:, 1]) | \
                       (clause_idx[:, 0] == clause_idx[:, 2]) | \
                       (clause_idx[:, 1] == clause_idx[:, 2])
            # print(f'is_equal: {is_equal.sum()}  mean: {batch*3*r}  std: {batch*np.sqrt(3*r)}')
            clause_idx = clause_idx[~is_equal]
            if clause_idx.shape[0] >= batch * n_clause:
                break
            else:
                print('Not enough clauses generated, retrying...')
        clause_idx = clause_idx[:batch*n_clause, :].reshape(batch, n_clause, 3)

        planted_flip = torch.gather(planted_solution, 1, clause_idx.reshape(batch, n_clause*3))\
                            .reshape(batch, n_clause, 3)
        # the probability that some variable does not show up in any of the clauses
        # is roughly 1 - (1-e^(-3r))^n, practically negligible when n is not too large
        # ignore the check of whether all variables at least show up once
        
        clause_sign = torch.ones(batch, n_clause, 3)
        flip_prob = torch.tensor([p0, (1-4*p0)/2, (1+2*p0)/2]).expand(batch*n_clause, 3)
        flip_choice = torch.multinomial(flip_prob, 1, replacement=False).reshape(batch, n_clause)
        mask_1 = flip_choice==1
        mask_2 = flip_choice==2
        n1 = mask_1.sum()
        n2 = mask_2.sum()
        flip_idx_1 = torch.multinomial(torch.ones(n1, 3), 1, replacement=False)
        flip_idx_2 = torch.multinomial(torch.ones(n2, 3), 2, replacement=False)
        
        temp_1 = clause_sign[mask_1].clone()
        temp_2 = clause_sign[mask_2].clone()
        temp_1[torch.arange(n1).reshape(n1, 1), flip_idx_1] *= -1
        temp_2[torch.arange(n2).reshape(n2, 1), flip_idx_2] *= -1

        clause_sign[mask_1] = temp_1
        clause_sign[mask_2] = temp_2
        clause_sign *= planted_flip
        
        if output_to_file:
            if folder==None:
                folder = './'
            folder = folder + 'p0_{:03d}/ratio_{}_{}/var_{}/instances/'\
                                .format(int(p0*1000), int(r), round(100*(r-int(r))), n)
            try:
                os.makedirs(folder)
            except:
                pass
            clauses = ((clause_idx+1) * clause_sign).to(torch.int64)
            for i in range(batch):
                name = 'transformed_barthel_n_{}_r_{:4.3f}_p0_{:4.3f}_instance_{:03d}.cnf'\
                        .format(n, r, p0, i+1)
                with open(folder + name, 'w') as f:
                    # f.write('c ' + str((torch.arange(1, n+1)*planted_solution[i])\
                    #                    .detach().cpu().numpy()) + ' \n')
                    f.write('c r={:4.3f} p0={:4.3f}\n'.format(r, p0))
                    f.write('p cnf {} {}\n'.format(n, n_clause))
                    for j in range(n_clause):
                        f.write('{} {} {} 0\n'\
                        .format(clauses[i, j, 0], clauses[i, j, 1], clauses[i, j, 2]))

        return clause_idx, clause_sign


#Returns statistics on a particular instance (clause_idx, clause_sign, n_var, n_clause, n_sat, and n_sat_count) given its filename
@torch.no_grad()
def import_data(file):
    clauses = []
    n_sat = []
    with open(file, 'r') as f:
        for line in f:
            if line.startswith('c'):
                continue
            if line.startswith('p'):
                n_var, n_clause = map(int, line.split(' ')[2:4])
                #n_clause = int(n_clause/4) #including for XORgated .cnf files
                continue
            clause = list(map(int, line.strip().split(' ')[:-1]))
            clauses.append(clause)
            n_sat.append(len(clause))
    # assert np.concatenate(clauses).max() == n_var, 'Number of variables does not match'
    # assert len(clauses) == n_clause, 'Number of clauses does not match'
    max_var = abs(np.concatenate(clauses)).max() #works for OR or XORgated .cnf files
    if max_var != n_var:
        logging.warning(f'{file} Number of variables does not match, expected {n_var}, got {max_var}')
    if len(clauses) != n_clause:
        logging.warning(f'{file} Number of clauses does not match, expected {n_clause}, got {len(clauses)}')
    n_sat = np.array(n_sat)
    if len(np.unique(n_sat)) == 1:
        clauses = torch.tensor(clauses)
        clause_idx = [torch.abs(clauses) - 1]
        clause_sign = [2 * (clauses > 0).to(torch.get_default_dtype()) - 1]
    else:
        rearranged_idx = np.argsort(n_sat)
        n_sat = n_sat[rearranged_idx]
        clauses = np.array(clauses, dtype=object)
        clauses = clauses[rearranged_idx]
        clause_idx = []
        clause_sign = []
        diff = np.diff(n_sat)
        diff_idx = np.where(diff)[0]
        diff_idx += 1
        diff_idx = np.concatenate([[0], diff_idx, [len(clauses)]])
        for i in range(len(diff_idx)-1):
            temp = clauses[diff_idx[i]:diff_idx[i+1]].tolist()
            temp = torch.tensor(temp)
            clause_idx.append(torch.abs(temp) - 1)
            clause_sign.append(2 * (temp > 0).to(torch.get_default_dtype()) - 1)

    n_sat, n_sat_count = np.unique(n_sat, return_counts=True)
    return clause_idx, clause_sign, n_var, n_clause, n_sat, n_sat_count


if __name__ == '__main__':
    for n in ns:
        print(f'Generating instances with {n} variables...')
        dataset = SAT_dataset(num_instances, n)
        dataset.generate_instances(True, folder='data/')
