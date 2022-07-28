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
        
        # determine the directory to obtain sample sim scores
        if args.adapter_pseudo_label_form == 'step_narraion_matching_mpnet':
            gt_source_path = args.segment_wikistep_sim_scores_n_path
        elif args.adapter_pseudo_label_form == 'step_video_matching_s3d_text':
            gt_source_path = args.segment_wikistep_sim_scores_v_path
        else:
            self.logger.info(
                'The adapter_pseudo_label_form is not implemented!\nFunc: {}\nFile:{}'.format(
                    __name__, __file__))
            os._exit(0)
            
        # get all sample paths (samples' sim score file paths)    
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
            
        
        # preparation based on adapter objective
        if 'ours' in self.args.adapter_objective:
            
            from datasets.build_kg_no_edges import get_nodes_by_removing_step_duplicates
            self.node2step, self.step2node = get_nodes_by_removing_step_duplicates(args, logger)
            if self.args.num_nodes == -1:
                self.args.num_nodes = len(self.node2step)  # modify the num_nodes in args!
            else:
                assert self.args.num_nodes == len(self.node2step) 
            logger.info('The number of nodes is: {}'.format(self.args.num_nodes))
            
            from datasets.build_kg_no_edges import (
                get_edges_between_wikihow_steps_in_wikihow, get_edges_between_wikihow_steps_in_howto100m)
            self.G_wikihow = get_edges_between_wikihow_steps_in_wikihow(args, logger)
            self.G_howto100m = get_edges_between_wikihow_steps_in_howto100m(args, logger, len(self.step2node))
            
            
        else:  # baseline objectives
            
            if self.args.adapter_objective == 'step_regression' and self.args.step_regression_func == 'mse':
                from datasets.build_kg_no_edges import get_step_des_feats
                if args.adapter_pseudo_label_form == 'step_narraion_matching_mpnet':
                    self.step_embed = get_step_des_feats(args, logger, language_model="MPNet")
                elif args.adapter_pseudo_label_form == 'step_video_matching_s3d_text':
                    self.step_embed = get_step_des_feats(args, logger, language_model="S3D")
                else:
                    logger.info('The adapter_pseudo_label_form is not implemented!\nFunc: {}\nFile:{}'.format(
                            __name__, __file__))
                    os._exit(0)
                # self.step_embd: (S, d)
                self.step_embed = torch.tensor(self.step_embed)
            
        
        
        # self.__getitem__(0)
        # for index in tqdm(range(self.__len__())):
        #     self.__getitem__(index)
        # pdb.set_trace()
        
        
    def __len__(self):
        return len(self.sample_gt_files)
    
    
    def get_pseudo_label_Q1(self, step_scores):
        if self.args.adapter_kl_topk > 0:
            topk_step_indices = numpy_topk_indices(step_scores, self.args.adapter_kl_topk)
            topk_node_indices = [self.step2node[step_id] for step_id in topk_step_indices]

            # obtain node scores
            node_scores = dict()
            for node_id in topk_node_indices:
                node_scores[node_id] = 0
            for i in range(self.args.adapter_kl_topk):
                step_id = topk_step_indices[i]
                node_id = self.step2node[step_id]
                node_scores[node_id] = max(node_scores[node_id], step_scores[step_id])

            # obtain normalized topk_node_confidence
            node_scores_normalized = []
            node_indices = []
            for node_id in node_scores:
                node_indices.append(node_id)
                node_scores_normalized.append(node_scores[node_id])
            node_scores_normalized = torch.nn.functional.softmax(
                torch.FloatTensor(node_scores_normalized), dim=0)
            pseudo_label = torch.zeros(len(self.node2step))
            for i in range(len(node_indices)):
                pseudo_label[node_indices[i]] = node_scores_normalized[i]

        else:  # not just consider topk for kl

            # obtain node scores
            node_scores = dict()
            for node_id in range(len(self.node2step)):
                node_scores[node_id] = 0
            for step_id in range(len(step_scores)):
                node_id = self.step2node[step_id]
                node_scores[node_id] = max(node_scores[node_id], step_scores[step_id])

            # obtain normalized node_scores_normalized
            node_scores_normalized = []
            node_indices = []
            for node_id in node_scores:
                node_indices.append(node_id)
                node_scores_normalized.append(node_scores[node_id])
            pseudo_label = torch.nn.functional.softmax(
                torch.FloatTensor(node_scores_normalized), dim=0)
        return pseudo_label
                     
    
    def __getitem__(self, index):
        sample_gt_path = self.sample_gt_files[index]
        
        video_sid = sample_gt_path.split('/')[-2]
        segment_iid = int(sample_gt_path.split('/')[-1].split('.')[0].split('segment_')[1])
        
        segment_video_feat = np.mean(
            np.load(
                os.path.join(self.args.segment_feat_dir, video_sid, 'video.npy')
            )[segment_iid], axis=0)
        # (512,)
        
        step_scores = np.load(sample_gt_path)
        # (10588,)
        
        if 'ours' in self.args.adapter_objective:
            # obtain answers for various adapter question types
            if 'Q1' in self.args.adapter_objective:
                pseudo_label_Q1 = self.get_pseudo_label_Q1(step_scores)
                
            if self.args.adapter_objective == 'ours_Q1':
                return torch.FloatTensor(segment_video_feat), pseudo_label_Q1
            

        else:  # baseline adapter objectives
            if self.args.adapter_objective == 'step_cls_with_bg':
                pseudo_label = np.argmax(step_scores)
                if step_scores[pseudo_label] < self.args.bg_cls_pseudo_label_threshold:
                    pseudo_label = self.args.adapter_num_classes - 1 
                    # the last class id is for the background class
                return torch.FloatTensor(segment_video_feat), pseudo_label

            elif self.args.adapter_objective in {'step_cls_without_bg', 'step_regression'}:
                if self.args.adapter_objective == 'step_regression' and self.args.step_regression_func == 'mse':
                    pseudo_label = np.argmax(step_scores)
                    return torch.FloatTensor(segment_video_feat), pseudo_label, self.step_embed[pseudo_label, :]
                else:
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
                self.logger.info(
                    'The adapter_objective is not implemented!\nFunc: {}\nFile:{}'.format(
                        __name__, __file__))
                os._exit(0)
            
            