import torch
import torch.nn as nn

import os
import pdb


def create_model(args, logger, model_name):
    
    if model_name == 'mlp_with_skip':
        from models.adapter_mlp_with_skip_connect import Adapter
        model = Adapter(args, logger)
    elif model_name == 'mlp_without_skip':
        from models.adapter_mlp import Adapter
        model = Adapter(args, logger)
    elif model_name == 'transformer_one_layer':
        from models.task_head_task_cls import Task_Head
        model = Task_Head(args, logger)
    else:
        raise ValueError("Model {} not recognized.".format(args.adapter_name))


    
    # logger.info(model)
    logger.info("--> model {} was created".format(model_name))

    return model


def check_adapter_objective(args, logger):
    if args.adapter_objective == 'step_cls_with_bg':
        assert args.adapter_num_classes == 10589
    elif args.adapter_objective == 'step_cls_without_bg':
        assert args.adapter_num_classes == 10588
    else:
        logger.info('The adapter_objective is not implemented yet!')
        os._exit(0)
    return