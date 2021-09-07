from calc_supps import key_motif_mining_gpu
import os
import numpy as np
from glob import glob
import torch
from tqdm import tqdm

class mining_motif(object):
    def __init__(self,
                 patch_size=16,
                 num_clusters=20,
                 set_idx=0,
                 gap=5,
                 epsilon=0.05,
                 loop=1):
        self.data_path = 'data/train_{}_clusters_p{}_setidx_{}/'.format(num_clusters,
                                                                        patch_size,
                                                                        set_idx)
        self.save_path = 'mined_motif/{}_clusters_p{}_setidx_{}_gap{}_loop{}/'.format(num_clusters,
                                                                         patch_size,
                                                                         set_idx,
                                                                         gap,loop)
        self.patch_size = patch_size
        self.num_clusters = num_clusters
        self.gap = gap
        self.loop=loop
        self.epsilon = epsilon
        self.set_idx = set_idx
        self.classes = sorted(os.listdir(self.data_path))
        
    def mining(self):
        for i in range(len(self.classes)):
            self.mining_one_class(i)
                
    def mining_one_class(self,i):
        classe = self.classes[i]
        print('Getting key motifs for the class of',classe)
        p_sequences = []
        fns = glob(self.data_path+'/'+classe+'/*')
        print('Concatenate probabilistic sequences!')
        for fn in tqdm(fns):
            clip_data = torch.load(fn)
            p_sequences += clip_data['p_sequences']
        
        p_sequences = np.array(p_sequences)
        print('Mining key motifs starting!')
        
        print('Mining key motifs for dynamic textures with the epsilon of {}'.format(self.epsilon))
        T = key_motif_mining_gpu(D = p_sequences,g=self.gap,
                                 epsilon=self.epsilon)
        
        print('Mining key motifs finished!')
        if os.path.exists(self.save_path):
            torch.save(T,self.save_path+classe+'.pt')
        else:
            os.makedirs(self.save_path)
            torch.save(T,self.save_path+classe+'.pt')