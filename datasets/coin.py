import os
import pdb
import sys
from pathlib import Path 
import pickle
import numpy as np
import random
import copy
import json
from tqdm import tqdm
import time
import glob
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class COIN(Dataset):
    def __init__(self, args, logger, split='train'):
        
        self.args = args
        self.logger = logger
        
        with open(args.coin_annoataion_json, 'r') as f:
            self.coin_json = json.load(f)
        
        # obtain the task name to id mapping and the list of video samples
        self.task_sid2iid = defaultdict()
        self.task_iid2sid = defaultdict()
        self.sample_video_to_task = defaultdict()
        self.sample_video_paths = []
        for video_sid in self.coin_json['database']:
            task_sid = self.coin_json['database'][video_sid]['class']
            if task_sid not in self.task_sid2iid:
                self.task_sid2iid[task_sid] = len(self.task_sid2iid)
            self.task_iid2sid[self.task_sid2iid[task_sid]] = task_sid
            self.sample_video_to_task[video_sid] = self.task_sid2iid[task_sid]
            
            video_feat_path = os.path.join(args.coin_s3d_feat_dir, video_sid)
            if split == 'train' and self.coin_json['database'][video_sid]['subset'] == 'training':
                self.sample_video_paths.append(video_feat_path)
            elif split == 'test' and self.coin_json['database'][video_sid]['subset'] == 'testing':
                self.sample_video_paths.append(video_feat_path)
                
        # logger.info('The COIN dataset has {} tasks in total'.format(
        #     len(self.task_sid2iid)))
        # The COIN dataset has 180 tasks in total 
        # The COIN dataset has 9030 training videos, 2797 testing videos
        
        
        # self.__getitem__(0) 
        # for index in tqdm(range(self.__len__())):
        #     self.__getitem__(index)
        # pdb.set_trace()
        
        
        
    def __len__(self):
        return len(self.sample_video_paths)
        
    
    
    @staticmethod  
    def custom_collate(batch):
        # https://androidkt.com/create-dataloader-with-collate_fn-for-variable-length-input-in-pytorch/
        segments_list, label_list = [], []
        max_length = 0
        sequence_dim = None
        for (_segments, _label) in batch:
            if not sequence_dim:
                sequence_dim = _segments.shape[-1]
            max_length = max(max_length, len(_segments))
            segments_list.append(_segments)
            label_list.append(_label)
             
        mask_list = []
        for i in range(len(segments_list)):
            if len(segments_list[i]) < max_length:
                pad_length = max_length - len(segments_list[i])
                mask_list.append(torch.tensor([0]*len(segments_list[i])+[1]*pad_length))
                segments_list[i] = torch.cat(
                    [segments_list[i], torch.zeros((pad_length, sequence_dim))], dim=0)
            else:
                mask_list.append(torch.tensor([0]*max_length))
        return torch.stack(segments_list), torch.LongTensor(label_list), torch.stack(mask_list).bool()

    
    
    def __getitem__(self, index):
        sample_video_path = self.sample_video_paths[index]
        video_sid = sample_video_path.split('/')[-1]
        task_iid = self.sample_video_to_task[video_sid]
        
        
        video_feats = np.load(
            os.path.join(sample_video_path, 'video.npy'))
        video_feats = np.mean(video_feats, axis=1)
        
        return torch.FloatTensor(video_feats), task_iid
        