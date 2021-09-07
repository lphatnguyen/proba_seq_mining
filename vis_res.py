#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 13:36:15 2020

@author: lnguyen
"""


import torch
from glob import glob
import os
from sklearn.metrics import confusion_matrix as cm
from mlxtend.plotting import plot_confusion_matrix
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cmaps
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--coef', type = int, default=20,
                    help="The subdataset index of the DynTex dataset. This is used for crossvalidation. Default: 0")
parser.add_argument('--gamma', type = int, default=20,
                    help="The subdataset index of the DynTex dataset. This is used for crossvalidation. Default: 0")
args = parser.parse_args()

def view_cm(path,num_clusters,patch_size):
    res_fns = []
    for i in range(4):
        res_fns.append(path + '/c{}_p{}_setidx_{}_result.pt'.format(num_clusters,patch_size,i))
    res = [torch.load(res_fn) for res_fn in res_fns]
    cms = {}
    epsilons = list(res[0].keys())
    for eps in epsilons:
        cms_eps = []
        for i in range(4):
            conf_mat = cm(res[i][eps][1],res[i][eps][0])
            if conf_mat.shape[0]<=2:
                print("Size error!")
            cms_eps.append(conf_mat)
        cms[eps] = [conf_mat.trace()/conf_mat.sum() for conf_mat in cms_eps]# sum(cms_eps)
    
    results = []
    for eps in epsilons:
        results.append(cms[eps])
    
    return results

if __name__ == "__main__":
    res = {}
    for gap in [5]:
        for ps in [8,12]:
            for nc in [20,50]:
                all_cmats = []
                
                path = 'results_gap{}_n_nearest/'.format(gap)
                dirs = []
                for i in range(1,args.coef+1):
                    for j in range(1,args.gamma+1):
                        dirs.append('gamma_{}_coef_{}'.format(j,i))

                cmats = []
                for sub_dir in tqdm(dirs):
                    subpath = path + sub_dir
                    cmats.append(view_cm(subpath,nc,ps))
                
                cmats = np.array(cmats)
                all_cmats.append(cmats)
                
                res['gap={},num_clusters={}, patch_size={}'.format(gap,nc,ps)] = np.array(all_cmats)

    torch.save(res,'results.pt')
    
    for key in res.keys():
        mean = res[key].squeeze().mean(axis=2).max(axis=0)
        print(key,mean)