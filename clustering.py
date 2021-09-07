#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:01:43 2020

@author: lnguyen
"""
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from ucla_dataset import ucla_dataset
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from joblib import dump
import os

def kmeans(dataset_path,
           set_idx = 0,
           patch_size=16,
           num_clusters=20,
           loop=1):
    num_samples = 4
    dataset = ucla_dataset(path = dataset_path,
                           set_idx = set_idx,
                           patch_size = patch_size)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8192, shuffle=True)
    print("Number of samples:",len(dataset))
    print("Finished loading!")
    print("Starting calculating initial codebook!")
    
    nb_iter = 10
    Kinit = num_clusters
    K = Kinit
    N = dataset.patch_size*dataset.patch_size # Number of features
    patchSize = dataset.patch_size
    
    allPatches = torch.empty((0,N))
    for datum in train_loader:
        patch,_ = datum
        allPatches = torch.cat((allPatches, patch), dim=0)
    
    nbSamplesTotal = allPatches.shape[0]
    nb_samples_iter = 100000
    nb_blocks = nbSamplesTotal//nb_samples_iter
    
    centroids = torch.zeros((K,N))
    
    for k in range(K):
        idx_k = [np.random.randint(0, len(dataset)-1) for i in range(20)]
        centroids[k] = allPatches[idx_k].mean(dim=0)
        
    del dataset,train_loader
    
    centroids = centroids.data.numpy()
    data_reshape = [centroids[i,:].reshape(patchSize,patchSize) for i in range(centroids.shape[0])]
    plt.figure(figsize=[12.8,9.6])
    for i in range(len(data_reshape)):
        plt.subplot((K+9)//10,10,i+1)
        plt.axis('off')
        dat = data_reshape[i]
        plt.imshow(dat,cmap=cm.gray)
        
    if not os.path.isdir('visualize/p'+str(patch_size)+'_c'+str(num_clusters)+'/'):
        os.makedirs('visualize/p'+str(patch_size)+'_c'+str(num_clusters)+'/')
    plt.savefig('visualize/p'+str(patch_size)+'_c'+str(num_clusters)+'/epoch_0_p'+str(patch_size)+'.png',
                format = 'png')
    plt.close()
    
    gmm = GaussianMixture(n_components=num_clusters)
    
    allPatches = np.array(allPatches)
    gmm.fit(allPatches)
    
    clusters = gmm.means_
    
    data_reshape = [clusters[i,:].reshape(patchSize,patchSize) for i in range(clusters.shape[0])]
    plt.figure(figsize=[12.8,9.6])
    for i in range(len(data_reshape)):
        plt.subplot((K+9)//10,10,i+1)
        plt.axis('off')
        dat = data_reshape[i]
        plt.imshow(dat,cmap=cm.gray)
    plt.savefig('visualize/p'+str(patch_size)+'_c'+str(num_clusters)+'/afterCL_p'+str(patch_size)+'.png',
                format = 'png')
    plt.close()
    
    print('Is the GMM converged? {}'.format(gmm.converged_))
    
    if not os.path.exists('codebooks/'):
        os.makedirs('codebooks/')
    # torch.save(clusters,'codebooks/'+dataset_type+'/codebook_'+str(num_clusters)+'_p'+str(patch_size)+'_setidx_'+str(set_idx)+'.pt')
    dump(gmm,'codebooks/codebook_'+str(num_clusters)+'_p'+str(patch_size)+'_setidx_'+str(set_idx)+'_loop{}.joblib'.format(loop))
    
if __name__ == "__main__": 
    kmeans(dataset_path='../../datasets/UCLA dataset/',
           patch_size=8,
           set_idx = 0,
           num_clusters=20)