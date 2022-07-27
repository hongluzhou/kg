import numpy as np
import math
import copy
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.misc import build_mlp
from models.transformer import TransformerEncoderLayer


class Task_Head(nn.Module):
    def __init__(self, args, logger):
        super(Task_Head, self).__init__()
        
        self.args = args
        self.logger = logger
        
        #### embedding layers
        self.cls_embed_layer = nn.Embedding(1, args.model_step_forecasting_tx_hidden_dim)
        
        if args.model_step_forecasting_tx_time_pos_embed_type == 'absolute_learned_1D':
            
            from models.position_encoding import PositionEmbeddingAbsoluteLearned_1D
            self.time_embed_layer = PositionEmbeddingAbsoluteLearned_1D(
                args.model_step_forecasting_tx_max_time_ids_embed, args.model_step_forecasting_tx_hidden_dim)
        
        elif args.model_step_forecasting_tx_time_pos_embed_type == 'fixed_sinusoidal_1D':
            from models.position_encoding import PositionEmbeddingFixedSine_1D
            self.time_embed_layer = PositionEmbeddingFixedSine_1D(
                args.model_step_forecasting_tx_hidden_dim)
        
        else:
            raise ValueError(f"not supported {self.args.model_step_forecasting_tx_time_pos_embed_type}")
            
        
        self.long_term_model = TransformerEncoderLayer(
            args.model_step_forecasting_tx_hidden_dim, 
            args.model_step_forecasting_tx_nhead, 
            args.model_step_forecasting_tx_dim_feedforward,
            args.model_step_forecasting_tx_dropout, 
            args.model_step_forecasting_tx_activation)
        
        self.classifier = build_mlp(
            input_dim=args.model_step_forecasting_tx_hidden_dim, 
            hidden_dims=[args.model_step_forecasting_classifier_hidden_dim], 
            output_dim=args.model_step_forecasting_num_classes)
        
        
    def forward(self, video_feats, video_mask):
        """
        - video_feats: (B, num_segments, 512)
        - video_mask: (B, num_segments)
        """
        B = video_feats.shape[0]
        T = video_feats.shape[1]
        device = self.args.device
        
        # self.long_term_model
        
        # CLS initial embedding
        CLS_id = torch.arange(1, device=device).repeat(B, 1)
        CLS = self.cls_embed_layer(CLS_id)
        
        # time positional encoding
        if self.args.model_step_forecasting_tx_time_pos_embed_type == 'absolute_learned_1D':
            time_ids = torch.arange(1, T+1, device=device).repeat(B, 1)
            time_seq = self.time_embed_layer(time_ids) 
        elif self.args.model_step_forecasting_tx_time_pos_embed_type == 'fixed_sinusoidal_1D':
            time_seq = self.time_embed_layer(T, device=device).unsqueeze(0).unsqueeze(0).repeat(B, N, J, 1, 1)
        else:
            raise ValueError(f"not supported {self.args.model_step_forecasting_tx_time_pos_embed_type}")
      
        tx_updated_sequence = self.long_term_model(
            torch.cat([CLS, video_feats + time_seq], dim=1).transpose(0, 1),
            src_key_padding_mask = torch.cat([torch.zeros((B, 1)).bool().to(device), video_mask], dim=1)
        )
        
        fine_cls = tx_updated_sequence[0]
        pred_logist = self.classifier(fine_cls)
        return pred_logist