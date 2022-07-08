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
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def get_train_test_splits(cross_task_video_dir, train_ratio=0.8):
    task_sids = os.listdir(cross_task_video_dir)
    
    train_split = defaultdict()
    test_split = defaultdict()
        
    for task_sid in task_sids:
        videos_this_task = glob.glob(os.path.join(cross_task_video_dir, task_sid, '*.*'))

        train_split[task_sid] = np.random.choice(
            videos_this_task, int(len(videos_this_task)*train_ratio), replace=False).tolist()
        test_split[task_sid] = [
            vid for vid in videos_this_task if vid not in train_split[task_sid]]
    return train_split, test_split
             
        

class CrossTask(Dataset):
    def __init__(self, args, logger, split='train'):
        
        self.args = args
        self.logger = logger
        
        self.task_sid2iid = defaultdict()
        self.task_iid2sid = defaultdict()
        self.get_task_ids()
        # logger.info('The CrossTask dataset has {} tasks in total'.format(
        #     len(self.task_sid2iid)))
        # The CrossTask dataset has 83 tasks in total
        
        with open(os.path.join(args.cross_task_s3d_feat_dir, '{}_split.pickle'.format(split)), 
                  'rb') as f:
            split = pickle.load(f)
        self.sample_video_paths = []
        for t in split:
            self.sample_video_paths += split[t] 
        
        
        # self.__getitem__(0) 
        # for index in range(self.__len__()):
        #     self.__getitem__(index)
        # pdb.set_trace()
        
        
        
    def __len__(self):
        return len(self.sample_video_paths)
        
        
        
    def get_task_ids(self):
        task_sids = os.listdir(self.args.cross_task_video_dir)
        for i in range(len(task_sids)):
            task_sid = task_sids[i]
            self.task_sid2iid[task_sid] = i
            self.task_iid2sid[i] = task_sid
            # self.logger.info('CrossTask - Task {} has {} vidoes.'.format(
            #     task_sid, 
            #     len(os.listdir(os.path.join(self.args.cross_task_video_dir, task_sid)))))
        return
    
    
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
        task_id = self.task_sid2iid[sample_video_path.split('/')[-2]]
        video_id = sample_video_path.split('/')[-1].split('.')[0]
       
        video_feats = np.load(
            os.path.join(self.args.cross_task_s3d_feat_dir, video_id, 'video.npy'))
        video_feats = np.mean(video_feats, axis=1)
        
        return torch.FloatTensor(video_feats), task_id
        