#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 10:23:39 2020

@author: lnguyen
"""


import torch
import os

def get_union_set(num_clusters,
                  patch_size,
                  set_idx,
                  gap,
                  epsilon,
                  loop):
    path = './mined_motif/'
    motif_path = path +'{}_clusters_p{}_setidx_{}_gap{}_loop{}/'.format(num_clusters,
                                                           patch_size,
                                                           set_idx,
                                                           gap,loop)
    classes = os.listdir(motif_path)
    classes = [classe[:-3] for classe in classes]
    motifs = {}
    
    for classe in classes:
        class_motifs = torch.load(motif_path+classe+'.pt')
        class_motifs_list = []
        for motif in class_motifs.values():
            class_motifs_list += motif
        
        motifs[classe] = set(class_motifs_list)
    
    union_motifs = set()
    for classe in classes:
        union_motifs = union_motifs.union(motifs[classe])
    
    all_motifs = {}
    for motif in union_motifs:
        current_motif = motif[0]
        current_support = motif[1]
        if all_motifs.get(current_motif)==None:
            all_motifs[current_motif] = current_support
        else:
            if all_motifs[current_motif] < current_support:
                all_motifs[current_motif] = current_support
    all_motifs = sorted(all_motifs.items())
    motifs = []
    supports = []
    for motif in all_motifs:
        motifs.append(motif[0])
        supports.append(motif[1])
    return motifs,supports

if __name__ == "__main__":
    motifs = get_union_set(num_clusters=20,
                           patch_size=8,
                           set_idx=0,
                           gap=5,
                           epsilon=0.05)