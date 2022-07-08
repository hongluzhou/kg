import os
import pdb
import sys
from pathlib import Path 
import pickle
import numpy as np
import random
import copy
import json
import time
import glob
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.utils.data import Dataset


class HT100M(Dataset):
    def __init__(self, args, logger):
        
        self.args = args
        self.logger = logger
        
        # self.sample_gt_files = list(
        #     glob.glob(
        #         os.path.join(
        #             args.segment_wikistep_sim_scores_v_path,
        #             '*', 
        #             'segment_*.npy'))) # NOTE!!!
        # the above line will take 15 min to run
        self.sample_gt_files = np.load(
            os.path.join(args.segment_wikistep_sim_scores_v_path, 'sample_paths.npy')
        )
        # the above line will take 11 s to run 
        
        # self.__getitem__(0)
        # for index in tqdm(range(self.__len__())):
        #     self.__getitem__(index)
        # pdb.set_trace()
        
        
    def __len__(self):
        return len(self.sample_gt_files)

    
    def __getitem__(self, index):
        sample_gt_path = self.sample_gt_files[index]
        
        video_sid = sample_gt_path.split('/')[-2]
        segment_iid = int(sample_gt_path.split('/')[-1].split('.')[0].split('segment_')[1])
        
        segment_video_feat = np.mean(
            np.load(os.path.join(self.args.segment_feat_dir, video_sid, 'video.npy'))[segment_iid], axis=0)
        # (512,)
        
        step_scores = np.load(sample_gt_path)
        # (10588,)
        
        gt_step_classid = np.argmax(step_scores)
        # if not isinstance(gt_step_classid, np.int64):
        #     print('not isinstance(gt_step_classid, np.int64)!!!!')
        #     print(index)
        # # pdb.set_trace()
        return torch.FloatTensor(segment_video_feat), gt_step_classid
        
        