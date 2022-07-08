import torch
import torch.nn as nn

import os
import pdb


def create_model(args, logger, model_name):
    
    if model_name == 'mlp_with_skip':
        from models.adapter_mlp_with_skip import Adapter
        model = Adapter(args, logger)
    elif model_name == 'transformer_one_layer':
        from models.task_head_task_cls import Task_Head
        model = Task_Head(args, logger)
    else:
        raise ValueError("Model {} not recognized.".format(args.adapter_name))


    
    # logger.info(model)
    logger.info("--> model {} was created".format(model_name))

    return model