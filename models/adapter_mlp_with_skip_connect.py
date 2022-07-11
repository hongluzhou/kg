import numpy as np
import math
import copy
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
        
        
        self.classifier = build_mlp(
            input_dim=args.adapter_refined_feat_dim, 
            hidden_dims=[args.adapter_num_classes//4, args.adapter_num_classes//2], 
            output_dim=args.adapter_num_classes)
        
        
        
    def forward(self, segment_feat, prediction=True):
        """
        - segment_feat: (B, 512)
        """
        
        # skip connection
        refined_segment_feat = self.args.skip_connection_refined_feat_ratio * self.adapter(
            segment_feat) + (1 - self.args.skip_connection_refined_feat_ratio) * segment_feat

        if prediction:
            return self.classifier(refined_segment_feat)
        else:
            return refined_segment_feat
        
    
        