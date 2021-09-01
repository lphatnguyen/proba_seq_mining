#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:13:16 2020

@author: lnguyen
"""

import torch
import numpy as np
import time
from tqdm import tqdm

def expanse(t):
    nb_el = len(t[0].split())
    assert nb_el>=2 , 'Number of elements should be greater than 1'
    new_t = []
    for motif_1 in t:
        motif_1 = motif_1.split()
        tail_1 = motif_1[1:]
        for motif_2 in t:
            motif_2 = motif_2.split()
            head_2 = motif_2[:-1]
            tail_2 = [motif_2[-1]]
            if tail_1==head_2:
                motif = ' '.join(motif_1+tail_2)
                new_t.append(motif)
    return new_t

def calc_supp_torch(t,D,g):
    t = t.split()
    len_t = len(t)
    len_p = D.size(1)
    nb_data = D.size(0)
    tables = torch.zeros((nb_data,len_t,len_p),device='cuda').float()
    tables[:,0,:] = D[:,:,int(t[0])]
    for i in range(1,len_t):
        for j in range(1,len_p):
            tables[:,i,j] = D[:,j,int(t[i])]*torch.max(tables[:,i-1,max(0,j-g):j],dim=1)[0]
    support = torch.sum(torch.max(tables[:,tables.shape[1]-1,:],dim=1)[0])/nb_data
    support = support.data.tolist()
    return support

def key_motif_mining_gpu(D,g,epsilon):
    k=1
    num_clusters = D.shape[2]
    D = torch.tensor(D,device='cuda')
    T_1 = [str(i) for i in range(num_clusters)]
    supports = []
    for t in tqdm(T_1):
        supports.append(calc_supp_torch(t,D,g))
    T = {}
    T_1_bis = []
    for i in range(len(supports)):
        if supports[i] >= epsilon:
            T_1_bis.append((T_1[i],supports[i]))
    T[str(k)] = T_1_bis
    print(T_1_bis)
    ret = True
    while ret:
        k+=1
        if k==2:
            T_k_motif = []
            for t_1 in T[str(k-1)]:
                t_1 = t_1[0]
                for t_2 in T[str(k-1)]:
                    t_2 = t_2[0]
                    T_k_motif.append(t_1 + ' ' + t_2)
        else:
            T_k = T[str(k-1)]
            T_k_motif = []
            for t in T_k:
                T_k_motif.append(t[0])
            T_k_motif = expanse(T_k_motif)

        supports = []
        for t in tqdm(T_k_motif):
            supports.append(calc_supp_torch(t,D,g))

        T_k_bis = []
        for i in range(len(supports)):
            if supports[i] >= epsilon:
                T_k_bis.append((T_k_motif[i],supports[i]))
        if T_k_bis != []:
            T[str(k)] = T_k_bis
            print(T_k_bis)
        else:
            ret = False
            
        if k==D.shape[1]-1:
            ret=False
    
    return T

if __name__ == "__main__":
    from glob import glob
    device = torch.device("cuda:0")
    fns = glob("data/train_20_clusters_p8_setidx_0/*/*.pt")
    D = []
    for fn in fns:
        item_data = torch.load(fn)['p_sequences']
        D += item_data
    
    D = np.array(D[:1000])
    # t = "17 1 17"
    
    time_0 = time.time()
    T_2 = key_motif_mining_gpu(D, 5, 0.1)
    print(time.time()-time_0)