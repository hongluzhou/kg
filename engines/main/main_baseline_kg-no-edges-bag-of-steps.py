import os
import pdb
import sys
import time
from datetime import datetime
import pprint
import random
import numpy as np
import platform
import wandb

import sys
sys.path.insert(0, os.path.abspath('./'))

from args.baseline_kg_bag_of_steps import get_args_parser
from datasets import return_dataset
from models import create_model, losses
from utils.common_utils import (
    getLogger, set_seed, get_cosine_schedule_with_warmup, 
    save_checkpoint_best_only, save_checkpoint, 
    adjust_lr, trim,
    AverageMeter, accuracy)

import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
torch.autograd.set_detect_anomaly(True)



def main_train_adapter(args):
    start_time = time.time()
    
    curr_time = datetime.now().strftime("%Y-%m-%dT%H-%M-%SZ")
    
    if args.hp_tune:
        logfile_path = os.path.abspath(
            os.path.join(args.log_dir, args.exp_name + '-' + curr_time + 
                         '-tune.log'))
    else:
        logfile_path = os.path.abspath(
            os.path.join(args.log_dir, args.exp_name + '-' + curr_time + 
                         '-train_adapter.log'))
    
    logger = getLogger(name=__name__, path=logfile_path)
    
    if args.seed > 0:
        set_seed(args.seed)
    else:
        args.seed = random.randint(0, 1000000)
        set_seed(args.seed)
    
    logger.info("Working config: {}\n".format(args))
    logger.info("Host: {}".format(platform.node()))
    logger.info("Logfile path: {}".format(logfile_path))
    
    if args.use_wandb:
        wandb.config.logfile_path = logfile_path
    
    logger.info("\n" + '-'*20)
    
    # Define datasets
    adapter_train_dataset = return_dataset(args, logger, args.adapter_train_dataset_name)
    logger.info('total number of samples is {} for < adapter > training data'.format(
        adapter_train_dataset.__len__()))
    adapter_train_loader = DataLoader(adapter_train_dataset,
                                      batch_size=args.adapter_batch_size,
                                      shuffle=True,
                                      num_workers=args.num_workers, 
                                      pin_memory=True)
    
    
    if args.adapter_evaluate_first_epoch <= args.adapter_num_epochs:
        downstream_train_dataset = return_dataset(args, logger, args.downstream_dataset_name, 'train')
        logger.info('total number of samples is {} for downstream < task head > training data'.format(
            downstream_train_dataset.__len__()))
        downstream_train_loader = DataLoader(downstream_train_dataset,
                                             batch_size=args.task_head_batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers, 
                                             collate_fn=downstream_train_dataset.custom_collate,
                                             pin_memory=True)

        downstream_test_dataset = return_dataset(args, logger, args.downstream_dataset_name, 'test')
        logger.info('total number of samples is {} for downstream < task head > testing data'.format(
            downstream_test_dataset.__len__()))
        downstream_test_loader = DataLoader(downstream_test_dataset,
                                            batch_size=args.task_head_batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers, 
                                            collate_fn=downstream_test_dataset.custom_collate,
                                            pin_memory=True)
    
    # Define adapter model
    adapter_model = create_model(args, logger, args.adapter_name)
    if args.load_pretrained: # Load checkpoint
        adapter_checkpoint = torch.load(args.checkpoint)
        adapter_params = adapter_checkpoint['state_dict']
        adapter_params = trim(adapter_params)
        adapter_model.module.load_state_dict(
            adapter_params, strict=False) if hasattr(
            adapter_model, 'module') else adapter_model.load_state_dict(
            adapter_params, strict=False)
        logger.info("Loaded adapter checkpoint from {}".format(args.checkpoint))

    adapter_model = nn.DataParallel(adapter_model).to(args.device)
    adapter_model_n_parameters = sum(p.numel() for p in adapter_model.parameters() if p.requires_grad)
    logger.info('number of params is {} for < adapter > training model'.format(adapter_model_n_parameters))
    
    
    # Define adapter criterion
    if args.adapter_objective in {'step_cls_with_bg', 'step_cls_without_bg'}:
        adapter_criterion = torch.nn.CrossEntropyLoss().to(args.device)
    elif args.adapter_objective in {'step_kl_distribution_matching'}:
        adapter_criterion = torch.nn.KLDivLoss(reduction=args.adapter_kl_reduction).to(args.device)
    elif args.adapter_objective in {'step_regression'}:
        if args.step_regression_func == 'mse':
            adapter_criterion = torch.nn.MSELoss(
                reduction=args.step_regression_func_mse_reductin).to(args.device)  # MSE loss
        else:
            adapter_criterion = losses.StepRegressionNCELoss(args, logger).to(args.device)  # NCE loss
    else:
        logger.info('The adapter_objective is not implemented!\nFunc: {}\nFile:{}'.format(
            __name__, __file__))
        os._exit(0)
            

    # Define adapter optimizer
    if args.adapter_optimizer == 'adam':
        adapter_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, adapter_model.parameters()), 
            lr=args.adapter_learning_rate, weight_decay=args.adapter_weight_decay)
    else:
        logger.info('adapter_optimizer != adam, not implemented!')
        os._exit(0)
        
    # Define adapter scheduler
    if not args.adapter_lr_warm_up:
        adapter_scheduler = None
        adapter_lr_plan = {}
        # adapter_lr_plan = {20: 5.e-5, 30: 1.e-5}
    else:
        adapter_scheduler = get_cosine_schedule_with_warmup(
            adapter_optimizer, args.adapter_warmup_steps, len(adapter_train_loader) * args.adapter_num_epochs)
    
    if args.cudnn_benchmark:
        cudnn.benchmark = True
        
    
    logger.info('Starting training loop for the < adapter > ...')
    training_adapter_start_time = time.time()
    best_acc = -1
    best_adapter_epoch = -1
    best_task_head_epoch = -1
    adapter_early_stop_counter= 0
    
            
    ##################################################################################################################
    for adapter_epoch in range(1, args.adapter_num_epochs + 1):
        
        if adapter_scheduler is None:
            if adapter_epoch in adapter_lr_plan:
                adjust_lr(adapter_optimizer, adapter_lr_plan[adapter_epoch])
        
        torch.cuda.empty_cache()
        
        #################################
        # --- train adapter for one epoch
        #################################
        logger.info('='*90)
        train_adapter_for_one_epoch_start_time = time.time()
        adapter_acc, adapter_loss = train_adapter_for_one_epoch(
            args, logger, 
            adapter_train_loader, adapter_model, 
            adapter_criterion, adapter_optimizer, adapter_scheduler, 
            adapter_epoch)
        logger.info("Finished training < adapter > adapter_epoch-{}, took {} seconds".format(
            adapter_epoch, round(time.time() - train_adapter_for_one_epoch_start_time, 2)))
        logger.info('='*90)
        
        if args.always_save_adapter_each_epoch:
            save_checkpoint(
                {'cfg': args, 
                 'epoch': adapter_epoch,
                 'state_dict': adapter_model.module.state_dict() if hasattr(
                     adapter_model, 'module') else adapter_model.state_dict(),
                 'optimizer': adapter_optimizer.state_dict()
                },  
                False,
                dir=args.checkpoint_dir, 
                name='Adapter-' + curr_time)
                
        
        ################################
        # --- evaluate adatper at this epoch
        ################################
        if adapter_epoch >= args.adapter_evaluate_first_epoch and adapter_epoch % args.adapter_evaluate_freq == 0:

            adapter_model.eval()
            adapter_criterion.eval()
            
            # Define task head model
            task_head_model = create_model(args, logger, args.task_cls_head_name)
            task_head_model = nn.DataParallel(task_head_model).to(args.device)
            task_head_model_n_parameters = sum(p.numel() for p in task_head_model.parameters() if p.requires_grad)
            logger.info('number of params is {} for < task head > training model'.format(task_head_model_n_parameters))
    
            # Define task head criterion
            task_head_criterion = torch.nn.CrossEntropyLoss().to(args.device)

            # Define task head optimizer
            if args.task_head_optimizer == 'adam':
                task_head_optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, task_head_model.parameters()), 
                    lr=args.task_head_learning_rate, weight_decay=args.task_head_weight_decay)
            else:
                logger.info('task_head_optimizer != adam, not implemented!')
                os._exit(0)
                
            # Define task head scheduler
            if not args.task_head_lr_warm_up:
                task_head_scheduler = None
                task_head_lr_plan = {}
                # task_head_lr_plan = {20: 5.e-5, 30: 1.e-5}
            else:
                task_head_scheduler = get_cosine_schedule_with_warmup(
                    task_head_optimizer, args.task_head_warmup_steps, len(downstream_train_loader) * args.task_head_num_epochs)   

            logger.info('Starting training loop for the < task head > ...')
            logger.info('-'*60)
            training_task_head_start_time = time.time()
            best_acc_this_adapter_epoch = -1
            best_task_epoch_this_adapter_epoch = -1
            task_head_early_stop_counter = 0
            ##################################################################################################################
            for task_head_epoch in range(1, args.task_head_num_epochs + 1):
        
                if task_head_scheduler is None:
                    if task_head_epoch in task_head_lr_plan:
                        adjust_lr(task_head_optimizer, task_head_lr_plan[task_head_epoch])
        
                torch.cuda.empty_cache()
            
                ###################################
                # --- train task head for one epoch
                ###################################
                train_task_head_for_one_epoch_start_time = time.time()
                train_task_head_for_one_epoch(
                    args, logger, 
                    adapter_model, adapter_epoch,
                    downstream_train_loader, task_head_model,
                    task_head_criterion, task_head_optimizer, task_head_scheduler, 
                    task_head_epoch)
                logger.info("Finished training < task head > task_head_epoch-{} (adapter_epoch-{}), took {} seconds".format(
                    task_head_epoch, adapter_epoch, round(time.time() - train_task_head_for_one_epoch_start_time, 2)))
                logger.info('-'*60)
                
                #####################
                # --- test task head
                #####################
                test_task_head_start_time = time.time()
                test_task_head_acc = test_task_head( 
                    args, logger, 
                    adapter_model, adapter_epoch,
                    downstream_test_loader, task_head_model, task_head_criterion, 
                    task_head_epoch)
                logger.info("Finished testing < task head > task_head_epoch-{} (adapter_epoch-{}), took {} seconds".format(
                    task_head_epoch, adapter_epoch, round(time.time() - test_task_head_start_time, 2)))
                logger.info('-'*60)
                
                task_head_has_improved = test_task_head_acc > best_acc_this_adapter_epoch
                if task_head_has_improved:
                    best_acc_this_adapter_epoch = test_task_head_acc
                    best_task_epoch_this_adapter_epoch = task_head_epoch
                    
                # logger.info('+'*70)
                logger.info("Current Downstream Result (task_head_epoch-{} adapter_epoch-{}):".format(task_head_epoch, adapter_epoch))
                logger.info("Acc of this task head epoch: {}".format(round(test_task_head_acc, 2)))
                logger.info("Best acc so far of this adapter epoch: {} (@task_head_epoch-{})".format(
                    round(best_acc_this_adapter_epoch, 2), best_task_epoch_this_adapter_epoch))
                # logger.info("Best acc over all adapter epochs: {} (@adapter_epoch-{})".format(
                #     round(best_acc, 2), best_adapter_epoch))
                # logger.info('+'*70)
                logger.info('-'*60)
                
                if task_head_has_improved:
                    task_head_early_stop_counter = 0
                else:
                    task_head_early_stop_counter += 1
                    if task_head_early_stop_counter == args.task_head_early_stop_patience:
                        logger.info("Early stopping the < task head > model")
                        break
                    
            logger.info('+'*90)
            logger.info("!!! Finished training and testing < task head > for all epochs, took {} seconds".format(
                round(time.time() - training_task_head_start_time, 2))) 
            
            adapter_has_improved = best_acc_this_adapter_epoch > best_acc
            if adapter_has_improved:
                best_acc = best_acc_this_adapter_epoch
                best_task_head_epoch = best_task_epoch_this_adapter_epoch
                best_adapter_epoch = adapter_epoch
                
                # save the adapter if it is the best so far
                save_checkpoint_best_only(
                                {'cfg': args,
                                 'epoch': adapter_epoch,
                                 'state_dict': adapter_model.module.state_dict() if hasattr(
                                     adapter_model, 'module') else adapter_model.state_dict(),
                                 'optimizer': adapter_optimizer.state_dict()
                                },  
                                dir=args.checkpoint_dir, 
                                name='Adapter-' + curr_time)
            logger.info(
                "!!! Best downstream accuracy so far: {} best_task_head_epoch-{} of best_adapter_epoch-{}".format(
                    round(best_acc, 2), best_task_head_epoch, best_adapter_epoch)
            )
            logger.info('+'*90)
            logger.info("\n")
            
            if adapter_has_improved:
                adapter_early_stop_counter = 0
            else:
                adapter_early_stop_counter += 1
                if adapter_early_stop_counter == args.adapter_early_stop_patience:
                    logger.info("Early stopping the < adapter > model")
                    break
            
        # finished evaluating adapter at this epoch    
        if args.adapter_name == 'mlp_with_skip' and args.skip_connection_refined_feat_ratio == 'learnable':
            learned_ratio = round(adapter_model.module.skip_connection_refined_feat_ratio.data.item(), 4)
            logger.info("The learnable adapter refine ratio is {}".format(learned_ratio))
                
        
        if args.use_wandb:
            wandb_logdict = {
                "adapter_epoch": adapter_epoch,
                "adapter_train_loss": adapter_loss,
                "adapter_train_acc": adapter_acc
            }
            
            wandb_logdict.update({'learned_skip_connection_refined_feat_ratio': learned_ratio})
        
            if adapter_epoch >= args.adapter_evaluate_first_epoch:
                wandb_logdict.update(
                    {
                        "task_head_test_acc": best_acc_this_adapter_epoch,
                        "task_head_test_acc_epoch": best_task_epoch_this_adapter_epoch,
                        "best_acc": best_acc,
                        "best_adapter_epoch": best_adapter_epoch,
                        "best_task_head_epoch": best_task_head_epoch
                    }
                )
                
            wandb.log(wandb_logdict, step=adapter_epoch)
            
            

    logger.info('\n\n\n' + '#'*90)       
    logger.info("Finished training and testing < adapter > for all epochs, took {} seconds".format(
        round(time.time() - training_adapter_start_time, 2)))                    
    logger.info(
        "Best downstream accuracy so far: {} best_task_head_epoch-{} of best_adapter_epoch-{}".format(
            round(best_acc, 2), best_task_head_epoch, best_adapter_epoch)
    )
    return



def train_adapter_for_one_epoch(
    args, logger, 
    train_loader, model, criterion, optimizer, scheduler, epoch):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss = AverageMeter()  
    acc = AverageMeter()
    
    
    batch_start_time = time.time()

    model.train()
    criterion.train() 
    
    for i, batch_data in enumerate(train_loader):
        if args.adapter_objective == 'step_regression' and args.step_regression_func == 'mse':
            segment_video_feat, targets_clsids, pseudo_targets = batch_data
            targets_clsids = targets_clsids.to(args.device)
            targets_thisbatch = pseudo_targets.to(args.device)
        else:
            segment_video_feat, pseudo_targets = batch_data
            targets_thisbatch = pseudo_targets.to(args.device)
            
        data_time.update(time.time() - batch_start_time)
        
        optimizer.zero_grad()
        if (args.adapter_name == 'mlp_with_skip' and 
            args.skip_connection_refined_feat_ratio == 'learnable' and 
            epoch > 10):
            preds_thisbatch = model(segment_video_feat, update_ratio=True)
        else:
            preds_thisbatch = model(segment_video_feat)
        
        # measure accuracy and record loss 
        if args.adapter_objective in {'step_cls_with_bg', 'step_cls_without_bg'}:
            
            loss_thisbatch = criterion(preds_thisbatch, targets_thisbatch) 
            acc_thisbatch = accuracy(preds_thisbatch, targets_thisbatch, topk=(1,))
        
        elif args.adapter_objective in {'step_kl_distribution_matching'}:
            loss_thisbatch = criterion(preds_thisbatch, targets_thisbatch) 
            acc_thisbatch = accuracy(preds_thisbatch, torch.argmax(targets_thisbatch, dim=1), topk=(1,))
        
        elif args.adapter_objective == 'step_regression':
            if args.step_regression_func == 'mse':
                loss_thisbatch = criterion(preds_thisbatch, targets_thisbatch) 
                segment_step_sim_scores = torch.matmul(preds_thisbatch, targets_thisbatch.t())  
                acc_thisbatch = accuracy(segment_step_sim_scores, targets_clsids, topk=(1,))
            else:
                loss_thisbatch, segment_step_sim_scores = criterion(preds_thisbatch, targets_thisbatch) 
                acc_thisbatch = accuracy(segment_step_sim_scores, targets_thisbatch, topk=(1,))
        
        else:
            logger.info('The adapter_objective is not implemented!\nFunc: {}\nFile:{}'.format(
                __name__, __file__))
            os._exit(0)
        
        loss.update(loss_thisbatch.item(), len(targets_thisbatch))
        acc.update(acc_thisbatch.item(), len(targets_thisbatch))
            
            
        loss_thisbatch.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        
         # finish
        batch_time.update(time.time() - batch_start_time)
        batch_start_time = time.time()
        
        # log loss of this batch
        if (i+1) % args.adapter_batch_train_log_freq == 0:
            logger.info('Train < Adapter > [e{0:02d}][{1}/{2}] '
                        # 'Batch Processing Time {batch_time.val:.3f}({batch_time.avg:.3f}) '
                        # 'Batch Data Loading Time {data_time.val:.3f}({data_time.avg:.3f}) '
                        'Loss {loss.val:.4f}({loss.avg:.4f}) '
                        'Acc {acc.val:.4f}({acc.avg:.4f})'.format(
                            epoch, i+1, len(train_loader), 
                            # batch_time=batch_time, data_time=data_time, 
                            loss=loss, acc=acc))
        
    return acc.avg, loss.avg
    
    
    
def train_task_head_for_one_epoch(
    args, logger, 
    adapter_model, adapter_epoch,
    train_loader, model, criterion, optimizer, scheduler, epoch):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss = AverageMeter()  
    acc = AverageMeter()
    
    
    batch_start_time = time.time()

    model.train()
    criterion.train() 
    
    for i, batch_data in enumerate(train_loader):
        longterm_video_feat, target_classid, longterm_video_mask = batch_data
        data_time.update(time.time() - batch_start_time)
        
        with torch.no_grad():
            longterm_video_feat = adapter_model(
                longterm_video_feat.flatten(0,1), prediction=False).reshape(
                longterm_video_feat.shape[0], 
                longterm_video_feat.shape[1],
                -1)
           
        optimizer.zero_grad()
        preds_thisbatch = model(longterm_video_feat, longterm_video_mask)
        
        # measure accuracy and record loss 
        targets_thisbatch = target_classid.to(args.device)
        loss_thisbatch = criterion(preds_thisbatch, targets_thisbatch) 
        acc_thisbatch = accuracy(preds_thisbatch, targets_thisbatch, topk=(1,))
        
        loss.update(loss_thisbatch.item(), len(targets_thisbatch))
        acc.update(acc_thisbatch.item(), len(targets_thisbatch))
            
            
        loss_thisbatch.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        
         # finish
        batch_time.update(time.time() - batch_start_time)
        batch_start_time = time.time()
        
        # log loss of this batch
        if (i+1) % args.task_head_batch_train_log_freq == 0:
            logger.info('Train < Task Head > [adapter_e{0:02d}][e{0:02d}][{2}/{3}] '
                        # 'Batch Processing Time {batch_time.val:.3f}({batch_time.avg:.3f}) '
                        # 'Batch Data Loading Time {data_time.val:.3f}({data_time.avg:.3f}) '
                        'Loss {loss.val:.4f}({loss.avg:.4f}) '
                        'Acc {acc.val:.4f}({acc.avg:.4f})'.format(
                            adapter_epoch, epoch, i+1, len(train_loader), 
                            # batch_time=batch_time, data_time=data_time, 
                            loss=loss, acc=acc))
        
    return
    

@torch.no_grad()
def test_task_head(
    args, logger, 
    adapter_model, adapter_epoch,
    test_loader, model, criterion, epoch):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss = AverageMeter()  
    acc = AverageMeter()
    
    
    batch_start_time = time.time()

    model.eval()
    criterion.eval() 
    
    for i, batch_data in enumerate(test_loader):
        longterm_video_feat, target_classid, longterm_video_mask = batch_data
        data_time.update(time.time() - batch_start_time)
        
        with torch.no_grad():
            longterm_video_feat = adapter_model(
                longterm_video_feat.flatten(0,1), prediction=False).reshape(
                longterm_video_feat.shape[0], 
                longterm_video_feat.shape[1],
                -1)
           
        preds_thisbatch = model(longterm_video_feat, longterm_video_mask)
        
        # measure accuracy and record loss 
        targets_thisbatch = target_classid.to(args.device)
        loss_thisbatch = criterion(preds_thisbatch, targets_thisbatch) 
        acc_thisbatch = accuracy(preds_thisbatch, targets_thisbatch, topk=(1,))
        
        loss.update(loss_thisbatch.item(), len(targets_thisbatch))
        acc.update(acc_thisbatch.item(), len(targets_thisbatch))
        
        
         # finish
        batch_time.update(time.time() - batch_start_time)
        batch_start_time = time.time()
        
        # log loss of this batch
        if (i+1) % args.task_head_batch_test_log_freq == 0:
            logger.info('Test < Task Head > [adapter_e{0:02d}][e{0:02d}][{2}/{3}] '
                        # 'Batch Processing Time {batch_time.val:.3f}({batch_time.avg:.3f}) '
                        # 'Batch Data Loading Time {data_time.val:.3f}({data_time.avg:.3f}) '
                        'Loss {loss.val:.4f}({loss.avg:.4f}) '
                        'Acc {acc.val:.4f}({acc.avg:.4f})'.format(
                            adapter_epoch, epoch, i+1, len(test_loader), 
                            # batch_time=batch_time, data_time=data_time, 
                            loss=loss, acc=acc))

            
    return acc.avg
    


if __name__ == '__main__':
    
    args = get_args_parser()
    
    if args.mode == "train_adapter":
        main_train_adapter(args)
        
         
    else:
        print("Wrong command mode!")
        os._exit(0)