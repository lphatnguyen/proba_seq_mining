from scipy.io import loadmat
import numpy as np
import torch
from tqdm import tqdm
from joblib import Parallel,delayed
from glob import glob
import cv2

class ucla_dataset(torch.utils.data.Dataset):
    def __init__(self,path,
                 eval_type = '4_fold',
                 set_idx = 0,
                 patch_size = 16,
                 num_samples=4):
        super(ucla_dataset,self).__init__()
        self.path = path
        self.patch_size = patch_size
        self.num_samples = num_samples
        # self.fns = sorted(glob(path + 'avi_vids/*.avi'))
        split_fns = torch.load(path+'ucla8_split.pt')[set_idx]
        self.fns = split_fns['train_fns']
        self.fns = [path+self.fns[i] for i in range(len(self.fns))]
        self.lbs = split_fns['train_lbs']
        self.patches = []
        self.labels = []
        # self.preprocess()
        self.patch_maker()
    
    def get_patches(self,idx):
        patches = []
        labels = []
        
        fn = self.fns[idx]
        classe = self.lbs[idx]
        cap = cv2.VideoCapture(fn)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        fc = 0
        nb_patch_per_frame = self.num_samples
        while fc < frame_count:
            _,frame = cap.read()
            if np.array(frame).all() != None:
                frame = cv2.resize(frame.astype(np.float32), (frame_width,frame_height), interpolation = cv2.INTER_LINEAR)
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                for iPatch in range(0, nb_patch_per_frame):
                    x = np.random.randint(0, frame_width-self.patch_size)
                    y = np.random.randint(0, frame_height-self.patch_size)
                    patch = frame[y:y+self.patch_size, x:x+self.patch_size].reshape(self.patch_size**2)
                    patch = patch.reshape(-1)
                    patches.append(torch.tensor(patch))
                    labels.append(classe)
                
            fc += 1
        
        cap.release()
        
        return patches,labels
    
    def patch_maker(self):
        data = Parallel(n_jobs=12,backend='multiprocessing')(delayed(self.get_patches)(idx) for idx in tqdm(range(len(self.fns))))
        # data = []
        # for idx in tqdm(self.trainind):
        #     data.append(self.get_patches(idx))
        
        for dat in data:
            self.patches += dat[0]
            self.labels += dat[1]
            
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        return self.patches[idx],self.labels[idx]
                

if __name__=="__main__":
    dataset = ucla_dataset(path = '../../datasets/UCLA dataset/')
    data,_ = dataset.__getitem__(10)
