#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:53:06 2020

@author: lnguyen
"""

from sequence_data import sequence_data
from joblib import Parallel,delayed
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
from clustering import kmeans
from mining_motifs import mining_motif
from train_val import histogram_dyntex
from motifs_filtering import filtering_motifs
import argparse
import torch
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--patch_size',type=int,default=8, 
                    help="Patch size, default: 16.")
parser.add_argument('--num_clusters', type = int, default=50,
                    help="Number of clusters in a codebook, default: 20")
parser.add_argument('--set_idx', type = int, default=0,
                    help="The index of the subset, default: 0")
parser.add_argument('--gap', type = int, default=5,
                    help="The maximum gap constraint, default: 5")
parser.add_argument('--loop', type = int, default=1,
                    help="The loop number iteration, default: 1")
parser.add_argument('--coef', type = int, default=1,
                    help="The subdataset index of the DynTex dataset. This is used for crossvalidation. Default: 0")
parser.add_argument('--gamma', type = int, default=1,
                    help="The subdataset index of the DynTex dataset. This is used for crossvalidation. Default: 0")
args = parser.parse_args()

dataset_path = '../../datasets/UCLA dataset/'
print(args)

def main():
    result = one_iter(args.set_idx)
    if not os.path.exists('results_gap{}_loop{}/gamma_{}_coef_{}'.format(args.gap,args.loop,args.gamma,
                                                            args.coef)):
        os.makedirs('results_gap{}_loop{}/gamma_{}_coef_{}'.format(args.gap,args.loop,args.gamma,
                                                      args.coef))
    torch.save(result,'results_gap{}_loop{}/gamma_{}_coef_{}/c{}_p{}_setidx_{}_result.pt'.format(args.gap,args.loop,args.gamma,
                                                                                    args.coef,
                                                                                    args.num_clusters,
                                                                                    args.patch_size,
                                                                                    args.set_idx))

def one_iter(set_idx):
    train_hist_path = 'histogram_data/train_diff_clip_gap{}_loop{}/c{}_p{}_setidx_{}.pt'.format(args.gap,args.loop,args.num_clusters,
                                                                                   args.patch_size,
                                                                                   set_idx)
    
    test_hist_path = 'histogram_data/test_diff_clip_gap{}_loop{}/c{}_p{}_setidx_{}.pt'.format(args.gap,args.loop,args.num_clusters,
                                                                                 args.patch_size,
                                                                                 set_idx)

    # Constructing codebook if necessary
    codebook_path = './codebooks/codebook_{}_p{}_setidx_{}_loop{}.joblib'.format(args.num_clusters,
                                                                          args.patch_size,
                                                                          set_idx,args.loop)
    
    if not os.path.exists(codebook_path):
        kmeans(dataset_path = dataset_path,
               set_idx = set_idx,
               patch_size = args.patch_size,
               num_clusters = args.num_clusters,
               loop = args.loop)
    
    # Computing probabilistic sequences and deterministic sequences
    data_path = 'data/train_{}_clusters_p{}_setidx_{}/'.format(args.num_clusters,
                                                               args.patch_size,
                                                               set_idx)
    if not os.path.exists(data_path):
        computing_ps = sequence_data(path = dataset_path,
                                     patch_size=args.patch_size,
                                     num_clusters=args.num_clusters,
                                     set_idx=set_idx,
                                     sequence_length=15,
                                     loop = args.loop)
        if not os.path.exists(train_hist_path):
            # Parallel(n_jobs=12,backend="multiprocessing")(delayed(computing_ps.each_item)(idx) for idx in tqdm(range(computing_ps.__len__())))
            for idx in tqdm(range(computing_ps.__len__())):
                computing_ps.each_item(idx)
    
    # Mining key motifs from the Dyntex dataset
    motif_path = 'mined_motif/{}_clusters_p{}_setidx_{}_gap{}_loop{}/'.format(args.num_clusters,
                                                                 args.patch_size,
                                                                 set_idx,args.gap,args.loop)
    if not os.path.exists(motif_path):
        motifs_mining = mining_motif(patch_size = args.patch_size,
                                     num_clusters = args.num_clusters,
                                     set_idx = set_idx,gap=args.gap,epsilon=0.05,
                                     loop=args.loop)
        motifs_mining.mining()
    
    # Getting vectors of supports using the mined key motifs
    
    histogram_construction = histogram_dyntex(dataset_path = dataset_path,
                                              patch_size = args.patch_size,
                                              num_clusters = args.num_clusters,
                                              set_idx = set_idx,
                                              sequence_length = 15,
                                              gap=args.gap,
                                              loop=args.loop)
    
    classes = histogram_construction.classes
    if not os.path.exists(train_hist_path):
        print('Computing train histogram for the set index {} dataset!'.format(args.set_idx))
        histogram_construction.getting_hist_data()
        
    if not os.path.exists(test_hist_path):
        print('Computing test histogram for the set index {} dataset!'.format(args.set_idx))
        histogram_construction.get_test_hist()
        
    train_hists_data = np.array(torch.load(train_hist_path)[0])
    train_labels_data = np.array(torch.load(train_hist_path)[1])
    supports = np.load('histogram_data/supports_c{}_p{}_setidx_{}_gap{}_loop{}.npy'.format(args.num_clusters,
                                                                                    args.patch_size,
                                                                                    args.set_idx,
                                                                                    args.gap,args.loop))
    
    epsilon_list = np.arange(0.05,0.16,0.02).tolist()
    results = {}
    index2keep = filtering_motifs(num_clusters=args.num_clusters,
                                  patch_size=args.patch_size,
                                  set_idx=args.set_idx,
                                  gap=args.gap,
                                  loop=args.loop,
                                  percentage=0.9)
    
    train_hists_data = train_hists_data[:,index2keep]
    supports = supports[index2keep]
    
    for epsilon in epsilon_list:
        train_hists = train_hists_data[:,supports>=epsilon]
        random_idx = np.random.permutation(train_hists.shape[0])
        train_hists = train_hists[random_idx,:]
        train_labels = train_labels_data[random_idx]
        train_chi_distance = chi2_kernel(X = train_hists,gamma=args.gamma/100)
        clf = SVC(C=args.coef, kernel='precomputed').fit(X=train_chi_distance,
                                                         y=train_labels)
        
        test_data = torch.load(test_hist_path)
        predicted_vid_labels = []
        ground_truth = []
        for item in test_data:
            video_hists = test_data[item][0]
            video_label = test_data[item][1]
            ground_truth.append(video_label)
            test_vid_hists = np.array(video_hists)
            test_vid_hists = test_vid_hists[:,index2keep]
            test_vid_hists = test_vid_hists[:,supports>=epsilon]
            test_video_distances = chi2_kernel(X = train_hists, Y = test_vid_hists, gamma=args.gamma/100)
            output_res = clf.predict(test_video_distances.transpose())
            output_res = list(output_res)
            predicted_vid_labels.append(classes[max(output_res,key=output_res.count)])
        cm = confusion_matrix(ground_truth,predicted_vid_labels)
        print(cm)
        print('The classification rate of this set is:',cm.trace()/cm.sum())
        results[str(epsilon)] = (predicted_vid_labels,ground_truth)
    return results

if __name__ == "__main__":
    main()