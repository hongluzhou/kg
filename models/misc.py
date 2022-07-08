import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


def build_mlp(input_dim, hidden_dims, output_dim=None, use_batchnorm=False, dropout=0):
    layers = []
    D = input_dim
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(input_dim))
    if hidden_dims:
        for dim in hidden_dims:
            layers.append(nn.Linear(D, dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dim))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.ReLU(inplace=True))
            D = dim
    if output_dim:
        layers.append(nn.Linear(D, output_dim))
    return nn.Sequential(*layers)