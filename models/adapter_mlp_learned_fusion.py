import numpy as np
import math
import copy
import os
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.misc import build_mlp
from models import check_adapter_objective


class Adapter(nn.Module):
    def __init__(self, args, logger):
        super(Adapter, self).__init__()
        
        check_adapter_objective(args, logger)
        
        self.args = args
        self.logger = logger
        
        assert self.args.adapter_refined_feat_dim == self.args.model_video_pretrained_dim
        
        adapter_layers = []
        adapter_layers.append(
            nn.Linear(args.model_video_pretrained_dim, args.bottleneck_dim))
        adapter_layers.append(
            nn.ReLU(inplace=True))
        adapter_layers.append(
            nn.Linear(args.bottleneck_dim, args.adapter_refined_feat_dim))
        self.adapter = nn.Sequential(*adapter_layers)
        
        self.fusion = build_mlp(input_dim=args.adapter_refined_feat_dim + args.model_video_pretrained_dim, 
                hidden_dims=[int((args.adapter_refined_feat_dim + args.model_video_pretrained_dim)/1.5),
                             int((args.adapter_refined_feat_dim + args.model_video_pretrained_dim)//2)], 
                output_dim=args.adapter_refined_feat_dim)
        
        if self.args.adapter_objective in {'step_cls_with_bg', 'step_cls_without_bg', 
                                           'step_kl_distribution_matching'}:
            self.classifier = build_mlp(
                input_dim=args.adapter_refined_feat_dim, 
                hidden_dims=[args.adapter_num_classes//4, args.adapter_num_classes//2], 
                output_dim=args.adapter_num_classes)
            
        elif self.args.adapter_objective == 'step_regression':
            if args.adapter_pseudo_label_form == 'step_narraion_matching_mpnet':
                if args.adapter_refined_feat_dim != args.mpnet_hidden_dim:
                    self.adapter_fix_hidden_dim_layer = nn.Linear(args.adapter_refined_feat_dim, args.mpnet_hidden_dim)
                else:
                    self.adapter_fix_hidden_dim_layer = None
            elif args.adapter_pseudo_label_form == 'step_video_matching_s3d_text':
                if args.adapter_refined_feat_dim != args.s3d_hidden_dim:
                    self.adapter_fix_hidden_dim_layer = nn.Linear(args.adapter_refined_feat_dim, args.s3d_hidden_dim)
                else:
                    self.adapter_fix_hidden_dim_layer = None
        else:
            self.logger.info('The adapter_objective is not implemented!\nFunc: {}\nFile:{}'.format(
                    __name__, __file__))
            os._exit(0)
        
        
        
    def forward(self, segment_feat, prediction=True):
        """
        - segment_feat: (B, 512)
        """
        refined_segment_feat = self.fusion(
            torch.cat([self.adapter(segment_feat), segment_feat], dim=1))
         
        if prediction:
            if self.args.adapter_objective in {'step_cls_with_bg', 'step_cls_without_bg'}:
                return self.classifier(refined_segment_feat)
            elif self.args.adapter_objective in {'step_kl_distribution_matching'}:
                return F.log_softmax(self.classifier(refined_segment_feat), dim=1)
            elif self.args.adapter_objective == 'step_regression':
                if self.adapter_fix_hidden_dim_layer:
                    refined_segment_feat = self.adapter_fix_hidden_dim_layer(refined_segment_feat)
                return refined_segment_feat
            else:
                self.logger.info('The adapter_objective is not implemented!\nFunc: {}\nFile:{}'.format(
                    __name__, __file__))
                os._exit(0)
        else:
            return refined_segment_feat
        
    
        