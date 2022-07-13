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

from utils.common_utils import numpy_topk_indices


class HT100M(Dataset):
    def __init__(self, args, logger):
        
        self.args = args
        self.logger = logger
        
        if args.adapter_pseudo_label_form == 'step_narraion_matching_mpnet':
            gt_source_path = args.segment_wikistep_sim_scores_n_path
        elif args.adapter_pseudo_label_form == 'step_video_matching_s3d_text':
            gt_source_path = args.segment_wikistep_sim_scores_v_path
        else:
            logger.info('The adapter_pseudo_label_form is not implemented!')
            os._exit(0)
            
        
        if not os.path.exists(os.path.join(gt_source_path, 'sample_paths.npy')):
            self.sample_gt_files = list(
                glob.glob(
                    os.path.join(
                        gt_source_path,
                        '*', 
                        'segment_*.npy'))) # NOTE!!!
            np.save(os.path.join(gt_source_path, 'sample_paths.npy'), self.sample_gt_files)
            # the above line will take 15~40 min to run
        else:
            self.sample_gt_files = np.load(
                os.path.join(gt_source_path, 'sample_paths.npy')
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
        
        if self.args.adapter_objective == 'step_cls_with_bg':
            pseudo_label = np.argmax(step_scores)
            if step_scores[pseudo_label] < self.args.bg_cls_pseudo_label_threshold:
                pseudo_label = self.args.adapter_num_classes - 1 
                # the last class id is for the background class
            return torch.FloatTensor(segment_video_feat), pseudo_label
        
        elif self.args.adapter_objective == 'step_cls_without_bg':
            pseudo_label = np.argmax(step_scores)
            return torch.FloatTensor(segment_video_feat), pseudo_label
        
        elif self.args.adapter_objective == 'step_kl_distribution_matching':
            if self.args.adapter_kl_topk > 0:
                topk_step_indices = numpy_topk_indices(step_scores, self.args.adapter_kl_topk)
                topk_step_confidence = torch.nn.functional.softmax(
                    torch.FloatTensor(step_scores[topk_step_indices]), dim=0)
                pseudo_label = torch.zeros(step_scores.shape)
                for i in range(len(topk_step_indices)):
                    pseudo_label[topk_step_indices[i]] = topk_step_confidence[i]
            else:
                pseudo_label = torch.nn.functional.softmax(torch.FloatTensor(step_scores), dim=0)
            return torch.FloatTensor(segment_video_feat), pseudo_label
        
        else:
            self.logger.info('The adapter_objective is not implemented!\nFunc: {}\nFile:{}'.format(__name__, __file__))
            os._exit(0)
            
            