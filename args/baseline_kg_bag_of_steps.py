import os
import pdb
import yaml
import argparse
from pathlib import Path
import wandb


def get_args_parser():
    
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    ###############################################
    # Create parser for the train_adapter command #
    ###############################################
    train_adapter_parser = subparsers.add_parser('train_adapter')
    train_adapter_parser.add_argument('--mode', type=str, default="train_adapter")
    train_adapter_parser.add_argument('--cfg', type=str, default="config/config.yml",
                                      help="config file path")
    train_adapter_parser.add_argument('--use_wandb', type=int, default=0,
                                      help="1 means use wandb to log experiments, 0 otherwise")
    train_adapter_parser.add_argument("--checkpoint", type=str, required=False,
                                      help="a path to model checkpoint file to load pretrained weights")
    train_adapter_parser.add_argument('--hp_tune', action='store_true', 
                                      help="tune hyper-parameter or not by using wandb")
    train_adapter_parser.add_argument('--not_save_best_model', action='store_true', 
                                      help="not save best model in this run")
     
    args = parser.parse_args()
    
    if args.hp_tune:
        args.use_wandb = 1
    
    #################################
    # Read yaml file to update args #
    #################################
    with open(args.cfg, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    for k, v in cfg.items():
        parser.add_argument('--{}'.format(k), default=v, type=type(v))
    args = parser.parse_args()

    
    ###############
    # Setup wandb #
    ###############
    if args.use_wandb:
        if args.hp_tune:
            wandb.init(project=args.project, entity=args.entity, 
                       name=args.exp_name, notes='[Tune]')
        else:
            wandb.init(project=args.project, entity=args.entity, 
                       name=args.exp_name, notes=args.notes)
        wandb.config.update(args)
    
    
        #################
        # Tuning params #
        #################
        if args.hp_tune:
            print('\n\n----------------- HYPER-PARAMETER TUNE -----------------')
            for i in wandb.config.keys():
                if i in args.__dict__.keys():  # overite args with wandb configurations
                    args.__dict__[i] = wandb.config[i]

            # check whether the hyper-parameters are set correctly
            for k in wandb.config.keys():  
                assert k in args.__dict__.keys()  # every key in wandb.config must present in args
                assert args.__dict__[k] == wandb.config[k]  # their values must equal

            for k in args.__dict__.keys():
                assert k in wandb.config.keys()  # every key in args must present in wandb.config
                assert args.__dict__[k] == wandb.config[k]  # their values must equal

            
    ###############################################
    # Dynamically modify some args here if needed #
    ###############################################
    if args.num_workers == -1:
        args.num_workers = torch.get_num_threads() - 1
        
        
    ################
    # Create paths #
    ################
    if args.checkpoint_dir:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)   
        
    return args

