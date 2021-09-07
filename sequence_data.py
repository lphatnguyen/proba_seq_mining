#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:03:52 2020

@author: lnguyen
"""
from joblib import Parallel,delayed
import torch
import numpy as np
import cv2
import os
from joblib import load
from tqdm import tqdm
from scipy.io import loadmat

class sequence_data(torch.utils.data.Dataset):
    def __init__(self,
                 path,
                 patch_size = 48,
                 num_clusters = 20,
                 set_idx = 0,
                 sequence_length = 25,
                 loop=1):
        super(sequence_data,self).__init__()
        self.patch_size = patch_size
        self.num_clusters = num_clusters
        self.set_idx = set_idx
        self.sequence_length = sequence_length
        self.loop=loop
        self.path = path
        split_fns = torch.load(path+'ucla8_split.pt')[set_idx]
        self.fns = split_fns['train_fns']
        self.fns = [path+self.fns[i] for i in range(len(self.fns))]
        self.lbs = split_fns['train_lbs']
        codebook_path = './codebooks/codebook_{}_p{}_setidx_{}_loop{}.joblib'.format(num_clusters,
                                                                              patch_size,
                                                                              set_idx,loop)
        self.codebook = load(codebook_path)
        self.saving_path = 'data/train_{}_clusters_p{}_setidx_{}/'.format(num_clusters,
                                                                          patch_size,
                                                                          set_idx)
        self.clip_maker()
        
    def clip_maker(self):
        self.clip_fns = []
        self.idx_frame = []
        self.clip_length = []
        self.clip_lbs = []
        for i in range(len(self.fns)):
            fn = self.fns[i]
            lb = self.lbs[i]
            cap = cv2.VideoCapture(fn)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            num_clip = int(frame_count//self.sequence_length)
            for j in range(num_clip):
                (self.clip_fns).append(fn)
                (self.clip_lbs).append(lb)
                (self.idx_frame).append(j*self.sequence_length)
                (self.clip_length).append(self.sequence_length)
            cap.release()
            
    def __len__(self):
        return len(self.clip_fns)
    
    def each_item(self,idx):
        fn = self.clip_fns[idx]
        lb = self.clip_lbs[idx]
        cap = cv2.VideoCapture(fn)
        cap.set(cv2.CAP_PROP_POS_FRAMES,self.idx_frame[idx])
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        fc = 0
        fc_1 = 0
        
        frame_block = np.empty((0,frame_height,frame_width,1))
        
        while fc < frame_count and fc_1 < self.sequence_length:
            _,frame = cap.read()
            if np.array(frame).all() != None:
                frame = cv2.resize(frame,(frame_width,frame_height), interpolation = cv2.INTER_LINEAR)
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                frame = np.expand_dims(frame,axis=2)
                frame = np.expand_dims(frame,axis=0)
                frame_block = np.concatenate((frame_block,frame),axis=0)
                fc_1 += 1
            fc += 1
        
        cap.release()
        
        frame_block = frame_block.squeeze().astype(np.float32)
        clip_data = []
        for y in range(frame_height//self.patch_size):
            for x in range(frame_width//self.patch_size):
                ystart = y*self.patch_size
                xstart = x*self.patch_size
                patches = frame_block[:,ystart:ystart+self.patch_size,xstart:xstart+self.patch_size].squeeze()
                patches = patches.reshape(patches.shape[0],self.patch_size*self.patch_size)
                clip_data.append(patches)
                
        clip_d_sequence = []
        clip_p_sequence = []
        
        for patch in clip_data:
            p_seq = self.codebook.predict_proba(patch)
            p_seq = [p_seq[i,:] for i in range(p_seq.shape[0])]
            clip_p_sequence.append(p_seq)
        
        data = {'p_sequences': clip_p_sequence}
        
        if not os.path.exists(self.saving_path + lb):
            os.makedirs(self.saving_path + lb)
        
        torch.save(data,self.saving_path + lb + '/clip_{}_{}.pt'.format(lb,idx))

if __name__ == "__main__":
    data_extract = sequence_data(path='../../datasets/UCLA dataset/',
                                 patch_size=16,
                                 num_clusters=20)
    
    data_extract.each_item(0)
