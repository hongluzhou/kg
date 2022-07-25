import os
import pdb
import sys
from pathlib import Path 
import pickle
import numpy as np
import random
import copy
import json
import glob
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose, ToTensor, Normalize



def return_dataset(args, logger, dataset_name, dataset_split='train'):
    
    if args.need_external_knowledge and not args.external_knowledge_ready:
        from datasets.build_kg_no_edges import obtain_external_knowledge
        obtain_external_knowledge(args, logger)

    if dataset_name == 'HowTo100M-subset_9.6s-visual-segments':
        from datasets.ht100m import HT100M
        return HT100M(args, logger)
    
    elif dataset_name == 'CrossTask':
        if (not os.path.exists(
            os.path.join(args.cross_task_s3d_feat_dir, 'train_split.pickle'))) or (
            not os.path.exists(os.path.join(args.cross_task_s3d_feat_dir, 'test_split.pickle'))):
            
            from datasets.cross_task import get_train_test_splits
            train_split, test_split = get_train_test_splits(
                args.cross_task_video_dir, train_ratio=0.8)
            with open(os.path.join(args.cross_task_s3d_feat_dir, 'train_split.pickle'), 'wb') as f:
                pickle.dump(train_split, f)
            with open(os.path.join(args.cross_task_s3d_feat_dir, 'test_split.pickle'), 'wb') as f:
                pickle.dump(test_split, f)
         
        from datasets.cross_task import CrossTask
        return CrossTask(args, logger, split=dataset_split)
    
    
    elif dataset_name == 'COIN':
        from datasets.coin import COIN
        return COIN(args, logger, split=dataset_split)
    
