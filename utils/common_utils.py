import os
import numpy as np
import gzip
import json
import datetime
import pdb
from datetime import datetime as dt
import matplotlib.pyplot as plt
import pickle
import sys
import logging
from logging import *
import shutil
import math
from pathlib import Path
from typing import List, Dict, Tuple
from collections import OrderedDict
import random
from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import LambdaLR


_FMT = "[%(asctime)s] %(levelname)s: %(message)s"
_DATEFMT = "%m/%d/%Y %H:%M:%S"

logging.basicConfig(
    level=logging.INFO, format=_FMT, datefmt=_DATEFMT, stream=sys.stdout
)


def fileHandler(path, format, datefmt, mode="w"):
    handler = logging.FileHandler(path, mode=mode)
    formatter = logging.Formatter(format, datefmt=datefmt)
    handler.setFormatter(formatter)
    return handler

  
def getLogger(
    name=None,
    path=None,
    level=logging.INFO,
    format=_FMT,
    datefmt=_DATEFMT,
):

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if path is not None:
        from pathlib import Path
        path = str(Path(path).resolve())
        if not any(map(lambda hdr: hasattr(hdr, 'baseName') and hdr.baseFilename == path, logger.handlers)):
            handler = fileHandler(path, format, datefmt)
            logger.addHandler(handler)

    return logger


def save_checkpoint(state, is_best, dir='checkpoints/', name='checkpoint'):
    os.makedirs(dir, exist_ok=True)
    filename = dir + name + f"_e{state['epoch']}" + '.pth'
    torch.save(state, filename)
    if is_best:
        best_filename = dir + name + '_model_best.pth'
        shutil.copyfile(filename, best_filename)
        
        
def save_checkpoint_best_only(state, dir='checkpoints/', name='checkpoint'):
    os.makedirs(dir, exist_ok=True)
    best_filename = os.path.join(dir, name + '_model_best.pth')
    torch.save(state, best_filename)


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
        
class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    
@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    if maxk == 1:
        return res[0]
    else:
        return res


def trim(stat, prefix='module'):
    r"""Remove prefix of state_dict keys.
    """

    stat_new = OrderedDict()
    for k, v in stat.items():
        if k.startswith(prefix):
            stat_new[k[len(prefix)+1:]] = v

    return stat_new if stat_new else stat


def adjust_lr(optimizer, new_lr):
    print('change learning rate:',new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def plot_line(x, y, plot_name):
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(x, y)
    plt.show()
    plt.ylabel('LR')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(plot_name, bbox_inches='tight', transparent=True, pad_inches=0)
    plt.close()
    return 



def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
