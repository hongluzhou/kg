import numpy as np
import math
import copy
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.misc import build_mlp
from models import check_adapter_objective


class Adapter(nn.Module):
    def __init__(self, args, logger):
        super(Adapter, self).__init__()
        
        check_adapter_objective(args, logger)
        
        self.args = args
        self.logger = logger
        
        assert self.args.adapter_refined_feat_dim == self.args.model_video_pretrained_dim
        
        adapter_layer_block_1 = []
        adapter_layer_block_1.append(nn.Linear(512, 384))
        adapter_layer_block_1.append(nn.Dropout(p=0.1))
        adapter_layer_block_1.append(nn.BatchNorm1d(384))
        adapter_layer_block_1.append(nn.ReLU(inplace=True))
        self.adapter_layer_block_1 = nn.Sequential(*adapter_layer_block_1)
        
        adapter_layer_block_2 = []
        adapter_layer_block_2.append(nn.Linear(384, 256))
        adapter_layer_block_2.append(nn.Dropout(p=0.1))
        adapter_layer_block_2.append(nn.BatchNorm1d(256))
        adapter_layer_block_2.append(nn.ReLU(inplace=True))
        self.adapter_layer_block_2 = nn.Sequential(*adapter_layer_block_2)
        
        adapter_layer_block_3 = []
        adapter_layer_block_3.append(nn.Linear(256, 128))
        adapter_layer_block_3.append(nn.Dropout(p=0.1))
        adapter_layer_block_3.append(nn.BatchNorm1d(128))
        adapter_layer_block_3.append(nn.ReLU(inplace=True))
        self.adapter_layer_block_3 = nn.Sequential(*adapter_layer_block_3)
        
        adapter_layer_block_4 = []
        adapter_layer_block_4.append(nn.Linear(128, 64))
        adapter_layer_block_4.append(nn.Dropout(p=0.1))
        adapter_layer_block_4.append(nn.BatchNorm1d(64))
        adapter_layer_block_4.append(nn.ReLU(inplace=True))
        self.adapter_layer_block_4 = nn.Sequential(*adapter_layer_block_4)
        
        adapter_layer_block_4_up = []
        adapter_layer_block_4_up.append(nn.Linear(64, 128))
        adapter_layer_block_4_up.append(nn.Dropout(p=0.1))
        adapter_layer_block_4_up.append(nn.BatchNorm1d(128))
        adapter_layer_block_4_up.append(nn.ReLU(inplace=True))
        self.adapter_layer_block_4_up = nn.Sequential(*adapter_layer_block_4_up)
        
        adapter_layer_block_3_up = []
        adapter_layer_block_3_up.append(nn.Linear(128, 256))
        adapter_layer_block_3_up.append(nn.Dropout(p=0.1))
        adapter_layer_block_3_up.append(nn.BatchNorm1d(256))
        adapter_layer_block_3_up.append(nn.ReLU(inplace=True))
        self.adapter_layer_block_3_up = nn.Sequential(*adapter_layer_block_3_up)
        
        adapter_layer_block_2_up = []
        adapter_layer_block_2_up.append(nn.Linear(256, 384))
        adapter_layer_block_2_up.append(nn.Dropout(p=0.1))
        adapter_layer_block_2_up.append(nn.BatchNorm1d(384))
        adapter_layer_block_2_up.append(nn.ReLU(inplace=True))
        self.adapter_layer_block_2_up = nn.Sequential(*adapter_layer_block_2_up)
        
        adapter_layer_block_1_up = []
        adapter_layer_block_1_up.append(nn.Linear(384, 512))
        adapter_layer_block_1_up.append(nn.Dropout(p=0.1))
        adapter_layer_block_1_up.append(nn.BatchNorm1d(512))
        adapter_layer_block_1_up.append(nn.ReLU(inplace=True))
        self.adapter_layer_block_1_up = nn.Sequential(*adapter_layer_block_1_up)
        
        # self.classifier = build_mlp(
        #     input_dim=args.adapter_refined_feat_dim, 
        #     hidden_dims=[args.adapter_num_classes//4, args.adapter_num_classes//2], 
        #     output_dim=args.adapter_num_classes)
        
        classifier_block_1 = []
        classifier_block_1.append(nn.Linear(512, 1024))
        classifier_block_1.append(nn.Dropout(p=0.1))
        classifier_block_1.append(nn.BatchNorm1d(1024))
        classifier_block_1.append(nn.ReLU(inplace=True))
        self.classifier_block_1 = nn.Sequential(*classifier_block_1)
        
        classifier_block_2 = []
        classifier_block_2.append(nn.Linear(1024, 2048))
        classifier_block_2.append(nn.Dropout(p=0.1))
        classifier_block_2.append(nn.BatchNorm1d(2048))
        classifier_block_2.append(nn.ReLU(inplace=True))
        self.classifier_block_2 = nn.Sequential(*classifier_block_2)
        
        classifier_block_3 = []
        classifier_block_3.append(nn.Linear(2048, 4096))
        classifier_block_3.append(nn.Dropout(p=0.1))
        classifier_block_3.append(nn.BatchNorm1d(4096))
        classifier_block_3.append(nn.ReLU(inplace=True))
        self.classifier_block_3 = nn.Sequential(*classifier_block_3)
        
        classifier_block_4 = []
        classifier_block_4.append(nn.Linear(4096, 8192))
        classifier_block_4.append(nn.Dropout(p=0.1))
        classifier_block_4.append(nn.BatchNorm1d(8192))
        classifier_block_4.append(nn.ReLU(inplace=True))
        self.classifier_block_4 = nn.Sequential(*classifier_block_4)
        
        classifier_output_layer = []
        classifier_output_layer.append(nn.Linear(8192, args.adapter_num_classes))
        classifier_output_layer.append(nn.Softmax(dim=1))
        self.classifier_output_layer = nn.Sequential(*classifier_output_layer)
        
        
       
        
        
        
    def forward(self, segment_feat, prediction=True):
        """
        - segment_feat: (B, 512)
        """
        
        output_adapter_block_1 = self.adapter_layer_block_1(segment_feat) # 384
        output_adapter_block_2 = self.adapter_layer_block_2(output_adapter_block_1) # 256
        output_adapter_block_3 = self.adapter_layer_block_3(output_adapter_block_2) # 128
        output_adapter_block_4 = self.adapter_layer_block_4(output_adapter_block_3) # 64
        
        output_adapter_block_4_up = self.adapter_layer_block_4_up(output_adapter_block_4) # 128
        output_adapter_block_4_up_res = self.args.skip_connection_refined_feat_ratio * output_adapter_block_4_up + (
            1 - self.args.skip_connection_refined_feat_ratio) * output_adapter_block_3
        
        output_adapter_block_3_up = self.adapter_layer_block_3_up(output_adapter_block_4_up_res) # 256
        output_adapter_block_3_up_res = self.args.skip_connection_refined_feat_ratio * output_adapter_block_3_up + (
            1 - self.args.skip_connection_refined_feat_ratio) * output_adapter_block_2
        
        output_adapter_block_2_up = self.adapter_layer_block_2_up(output_adapter_block_3_up_res) # 384
        output_adapter_block_2_up_res = self.args.skip_connection_refined_feat_ratio * output_adapter_block_2_up + (
            1 - self.args.skip_connection_refined_feat_ratio) * output_adapter_block_1
        
        output_adapter_block_1_up = self.adapter_layer_block_1_up(output_adapter_block_2_up_res) # 512
        output_adapter_block_1_up_res = self.args.skip_connection_refined_feat_ratio * output_adapter_block_1_up + (
            1 - self.args.skip_connection_refined_feat_ratio) * segment_feat
        
        output_classifier_blocks = self.classifier_block_4(
                    self.classifier_block_3(
                        self.classifier_block_2(
                            self.classifier_block_1(output_adapter_block_1_up_res))))
        output_pred_logits = self.classifier_output_layer(output_classifier_blocks)
        
        
        if prediction:
            return output_pred_logits
        else:
            return output_classifier_blocks
        
    
        