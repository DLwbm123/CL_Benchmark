# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from utils.status import progress_bar
from utils.seg_metrics import SegmentationMetrics
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from utils.loggers import CsvLogger
from utils import create_if_not_exists
from utils.metrics import *
from datasets import get_dataset
import copy
import numpy as np
import sys
import os
import h5py

def visualize(model: ContinualModel, dataset: ContinualDataset):
    metric_calculator = SegmentationMetrics(average=False, ignore_background=False)
    if dataset.SETTING == 'class-il':
        metric_calculator = SegmentationMetrics(average=False, ignore_background=False, activation='none')

    status = model.net.training
    model.net.eval()

    for k, test_loader in enumerate(dataset.test_loaders):
        dices = []
        pred_masks = []
        save_predcitions_path = os.path.join(dataset.args.pretrain, "T{}_preds.h5".format(k))
        hf = h5py.File(save_predcitions_path, 'w')
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            #if 'class-il' not in model.COMPATIBILITY:
            #    outputs = model(inputs, k)
            #else:
            #    outputs = model(inputs)
                #print("outputs", outputs.size())
            for batch_num in range(inputs.size(0)):
                outputs = model(inputs[batch_num].unsqueeze(0), k)
                outputs = outputs.detach()
            
                pred_mask = outputs.clone() #torch.argmax(outputs, dim=1) # 1 x H x W
                
                if dataset.SETTING == 'class-il':
                    #shift = sum(dataset.N_CLASSES_PER_TASK[:k]) if k>0 and k < dataset.N_TASKS else 0
                    
                    shift = test_loader.dataset.shift
                    up_shift = sum(dataset.N_CLASSES_PER_TASK[:k+1]) + 1
                    if up_shift < sum(dataset.N_CLASSES_PER_TASK):
                        outputs[:, up_shift:,:,:] = -float('inf')
                        
                    outputs_argmax = torch.argmax(outputs, dim=1) - shift
                    outputs_argmax = (outputs_argmax > 0)*outputs_argmax
                    #class_num = outputs.size(1) - shift
                    class_num = up_shift - shift
                    outputs = metric_calculator._one_hot(outputs_argmax, outputs, class_num)

                _, dice, precision, recall, asd = metric_calculator(labels[batch_num].unsqueeze(0), outputs)
                dices.append(dice)
                pred_masks.append(pred_mask.squeeze(0).permute(1,2,0))                
            
        pred_masks = torch.stack(pred_masks, dim = 0).cpu().numpy()
        print(pred_masks.shape)
        hf.create_dataset('pred_masks', data=pred_masks)
        hf.create_dataset('dices', data=np.array(dices))
        hf.close()
        
    model.net.train(status)
    
    return pred_masks, dices


def evaluate(model: ContinualModel, dataset: ContinualDataset):
    """
    Evaluates the dice of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    if dataset.SETTING == 'class-il':
        metric_calculator = SegmentationMetrics(average=False, ignore_background=False, activation='none')
    else:
        metric_calculator = SegmentationMetrics(average=False, ignore_background=False)
    status = model.net.training
    model.net.eval()

    dices = []
    asds = []
    precisions = []
    recalls = []
    #################################################
    for k, test_loader in enumerate(dataset.test_loaders):
        dics_per_testloader = []
        asds_per_testloader = []
        precisions_per_testloader = []
        recalls_per_testloader = []
        test_loader_iter = iter(test_loader)
        
        patient_info = [-1]+ test_loader.dataset.patient_info
        for n_patient in range(len(patient_info)-1):
            slice_labels = []
            slice_outputs = []
            for slice_index in range(patient_info[n_patient], patient_info[n_patient+1]):
                inputs, labels = next(test_loader_iter)
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                outputs = model(inputs, k)
                outputs = outputs.detach()
                slice_labels.append(labels)
                slice_outputs.append(outputs)
            labels = torch.cat(slice_labels, dim = 0)
            outputs = torch.cat(slice_outputs, dim = 0)
            
            
            if dataset.SETTING == 'class-il':
                #shift = sum(dataset.N_CLASSES_PER_TASK[:k]) if k>0 and k < dataset.N_TASKS else 0

                shift = test_loader.dataset.shift
                if k >= dataset.N_TASKS:
                    shift = 0
                    up_shift = sum(dataset.N_CLASSES_PER_TASK) + 1
                else:
                    up_shift = sum(dataset.N_CLASSES_PER_TASK[:k+1]) + 1
                if up_shift < sum(dataset.N_CLASSES_PER_TASK):
                    outputs[:, up_shift:,:,:] = -float('inf')
                        
                        
                outputs_argmax = torch.argmax(outputs, dim=1) - shift
                outputs_argmax = (outputs_argmax > 0)*outputs_argmax
                class_num = up_shift - shift

                outputs = metric_calculator._one_hot(outputs_argmax, outputs, class_num)

            _, dice, precision, recall, asd = metric_calculator(labels, outputs)
            dics_per_testloader.append(dice)
            asds_per_testloader.append(asd)
            precisions_per_testloader.append(precision)
            recalls_per_testloader.append(recall)
            
        '''for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            #if 'class-il' not in model.COMPATIBILITY:
            #    outputs = model(inputs, k)
            #else:
            #    outputs = model(inputs)
                #print("outputs", outputs.size())
            outputs = model(inputs, k)
            outputs = outputs.detach() '''
        ##########################################################


        dices.append(np.mean(dics_per_testloader))
        asds.append(np.mean(asds_per_testloader))
        precisions.append(np.mean(precisions_per_testloader))
        recalls.append(np.mean(recalls_per_testloader))
        

    model.net.train(status)
    return dices, asds, precisions, recalls

def evaluate_current_task(model: ContinualModel, dataset: ContinualDataset):
    """
    Evaluates the dice of the model for each current task only. For forward transfer calculation
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    if dataset.SETTING == 'class-il':
        metric_calculator = SegmentationMetrics(average=False, ignore_background=False, activation='none')
    else:
        metric_calculator = SegmentationMetrics(average=False, ignore_background=False)
    status = model.net.training
    model.net.eval()
    dics_per_testloader = []
    asds_per_testloader = []
    precisions_per_testloader = []
    recalls_per_testloader = []
    t = model.current_task
    for data in dataset.test_loader:
        inputs, labels = data
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        outputs = model(inputs)
        outputs = outputs.detach()
        # print("outputs", outputs.size())
        if dataset.SETTING == 'class-il':
            shift = dataset.test_loader.dataset.shift
            #shift = sum(dataset.N_CLASSES_PER_TASK[:t]) if t else 0
            if t >= dataset.N_TASKS: #N_TASKS + 1: whole heart
                shift = 0
                up_shift = sum(dataset.N_CLASSES_PER_TASK) + 1
            else:
                up_shift = sum(dataset.N_CLASSES_PER_TASK[:t+1]) + 1
            if up_shift < sum(dataset.N_CLASSES_PER_TASK):
                outputs[:, up_shift:,:,:] = -float('inf')
            outputs_argmax = torch.argmax(outputs, dim=1) - shift
            outputs_argmax = (outputs_argmax > 0)*outputs_argmax
            #class_num = outputs.size(1) - shift
            class_num = up_shift - shift
            outputs = metric_calculator._one_hot(outputs_argmax, outputs, class_num)
            
        _, dice, precision, recall, asd = metric_calculator(labels, outputs)
        dics_per_testloader.append(dice)
        asds_per_testloader.append(asd)
        precisions_per_testloader.append(precision)
        recalls_per_testloader.append(recall)

    model.net.train(status)
    return [np.mean(dics_per_testloader)], [np.mean(asds_per_testloader)], [np.mean(precisions_per_testloader)], [np.mean(recalls_per_testloader)]

#以后可能会加入的功能，在未来的任务上面测试
def evaluate_next_task(model: ContinualModel, dataset: ContinualDataset) -> Tuple[list, list]:
    """
    Evaluates the dice of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    if dataset.SETTING == 'class-il':
        metric_calculator = SegmentationMetrics(average=False, ignore_background=False)
    else:
        metric_calculator = SegmentationMetrics(average=False, ignore_background=False, activation='none')
    status = model.net.training
    model.net.eval()
    if dataset.next_test_loaders is not None:
        dics_per_testloader = []
        asds_per_testloader = []
        precisions_per_testloader = []
        recalls_per_testloader = []
        for data in dataset.next_test_loaders:
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = model(inputs)
            # print("outputs", outputs.size())
            _, dice, precision, recall, asd = metric_calculator(labels, outputs)
            dics_per_testloader.append(dice)
            asds_per_testloader.append(asd)
            precisions_per_testloader.append(precision)
            recalls_per_testloader.append(recall)
    else:
        dics_per_testloader = 0
        asds_per_testloader = 0
        precisions_per_testloader = 0
        recalls_per_testloader = 0
    model.net.train(status)
    return np.mean(dics_per_testloader), np.mean(asds_per_testloader), np.mean(precisions_per_testloader), np.mean(recalls_per_testloader)

def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace, logger) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    :param logger: plain logger
    """
    model.net.to(model.device)
    results = []
    results_asd = []
    results_precision = []
    results_recall = []
    csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
    ##############
    model_save_dir = os.path.join(csv_logger.savepath, args.name)
    create_if_not_exists(model_save_dir)
    ##############
    #for forward transfer
    dataset_copy = get_dataset(args)
    for t in range(dataset.N_TASKS):
        model.net.train()
        _, _ = dataset_copy.get_data_loaders(t)
    if model.NAME != 'icarl':# and model.NAME != 'pnn':
        model.net.eval()
        dices_random, asds_random, precisions_random, recalls_random = evaluate(model, dataset_copy)
        print("fwt", len(dices_random))

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        logger.info('Begin task {}'.format(t+1))
        if args.pseudo_label is not None and t:
            model_copy=copy.deepcopy(model)
            model_copy.net.eval()
            if args.pseudo_label == "entropy":
                model.thresholds, model.max_entropy = model.find_median(train_loader, logger)
                
        ##########################################
        model.current_task = t
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders(t) #train loader和test loader与任务无关
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        logger.info('current lr: {}'.format(model.opt.param_groups[0]['lr']))
        for param_group in model.opt.param_groups:
            param_group['lr'] = args.lr #重置学习率
        model.scheduler = torch.optim.lr_scheduler.StepLR(model.opt, step_size=args.lr_decay_epoch, gamma=args.lr_decay_rate, last_epoch=-1)
        ##########################################
        #just for forward transfer
        '''if t:
            dice_current, asd_current, precision_current, recall_current = evaluate_current_task(model, dataset)
            results[t-1] = results[t-1] + dice_current
            results_asd[t-1] = results_asd[t-1] + asd_current
            results_precision[t - 1] = results_precision[t - 1] + precision_current
            results_recall[t - 1] = results_recall[t - 1] + recall_current'''

        for epoch in range(args.n_epochs):
            for i, data in enumerate(train_loader):
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    #not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    if args.pseudo_label is not None and t:
                        outputs_old = model_copy.forward(inputs, t-1)
                        outputs_old = outputs_old.detach()
                        labels = model.pseudo_wrap(outputs_old, labels)
                    loss = model.observe(inputs, labels, inputs, logits)
                else:
                    inputs, labels = data
                    #print("inputs", inputs.size(), labels.size())
                    inputs, labels = inputs.to(model.device), labels.to(
                        model.device)
                    if args.pseudo_label is not None and t:
                        outputs_old = model_copy.forward(inputs, t-1)
                        outputs_old = outputs_old.detach()
                        labels = model.pseudo_wrap(outputs_old, labels)
                    #not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.observe(inputs, labels, inputs)

                
                progress_bar(logger, i, len(train_loader), epoch, t, loss)
                #print('break!')
                #break
            ################
            model.scheduler.step()
            ################
            if args.save_freq > 0 and epoch % args.save_freq == 0:
                logger.info('save models at epochs {}'.format(epoch))
                model.save_network(model.net, model_save_dir, t, epoch)
        if hasattr(model, 'end_task'):
            model.end_task(dataset)
        if args.save_freq >= 0:
            logger.info('save final models')
            model.save_network(model.net, model_save_dir, t, 'latest')
        dices, asds, precisions, recalls = evaluate(model, dataset_copy)

        results.append(dices)
        results_asd.append(asds)
        results_precision.append(precisions)
        results_recall.append(recalls)
        logger.info("\n Dice for current task: {}".format(dices))
        logger.info("\n Mean Dice: {}".format(np.mean(dices)))
        logger.info("\n Results: {}".format(results))
        logger.info('End task {}'.format(t+1))
        
    if dataset.SETTING == 'class-il':
        test_loader = dataset.get_whole_testloader()
        dice_whole, asd_whole, precision_whole, recall_whole = evaluate_current_task(model, dataset)
        logger.info("\n Dice for whole-heart seg: {}".format(dice_whole))
        logger.info("\n ASD for whole-heart seg: {}".format(asd_whole))
        logger.info("\n precision for whole-heart seg: {}".format(precision_whole))
        logger.info("\n recall for whole-heart seg: {}".format(recall_whole))
        logger.info('End final evaluation')
        
        
    csv_logger.add_mean_dice(results)
    csv_logger.add_bwt(results)
    csv_logger.add_fwt(results, dices_random)

    csv_logger.add_mean_asd(results_asd)
    csv_logger.add_bwt_asd(results_asd)
    csv_logger.add_fwt_asd(results_asd, asds_random)

    csv_logger.add_mean_precision(results_precision)
    csv_logger.add_bwt_precision(results_precision)
    csv_logger.add_fwt_precision(results_precision, precisions_random)

    csv_logger.add_mean_recall(results_recall)
    csv_logger.add_bwt_recall(results_recall)
    csv_logger.add_fwt_recall(results_recall, recalls_random)
    csv_logger.write(vars(args))
    
    
def test(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace, logger) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    :param logger: plain logger
    """
    model_list = ['T{}_latest.pth'.format(t) for t in range(dataset.N_TASKS)]

    results = []
    results_asd = []
    results_precision = []
    results_recall = []
    results_whole = []
    results_parameters = []
    
    model.net.to(model.device)
    model.net.eval()
    
    #for forward transfer
    ###########################
    dataset_copy = get_dataset(args)
    for t in range(dataset.N_TASKS):
        _, _ = dataset_copy.get_data_loaders(t)
    if model.NAME != 'icarl': # and model.NAME != 'pnn':
        dices_random, asds_random, precisions_random, recalls_random = evaluate(model, dataset_copy)
        print(dices_random)
        
    if dataset.SETTING == 'class-il':
        test_loader = dataset_copy.get_whole_testloader()
        model.current_task = dataset.N_TASKS
        dice_whole_random, asd_whole_random, precision_whole_random, recall_whole_random = evaluate_current_task(model, dataset_copy)
        model.current_task = 0
    ###########################
        
    #for backward transfer & mean dices
    ###########################
    for t in range(dataset.N_TASKS):
        
        logger.info('Begin task {}'.format(t+1))
        _, _ = dataset.get_data_loaders(t)
        model.current_task = t
        '''old_model = copy.deepcopy(model)
        old_model.net.eval()'''
        if model.NAME in ['pnn','dan']:
            model.begin_task(dataset)
            old_model.begin_task(dataset)
        model.net.eval()
        '''if t:
            dice_current, asd_current, precision_current, recall_current = evaluate_current_task(model, dataset)
            results[t-1] = results[t-1] + dice_current
            results_asd[t-1] = results_asd[t-1] + asd_current
            results_precision[t - 1] = results_precision[t - 1] + precision_current
            results_recall[t - 1] = results_recall[t - 1] + recall_current'''
            
        pretrain_path = os.path.join(args.pretrain, model_list[t])
        model.load_network(pretrain_path, model.net)
        
        '''#############################test##############
        if t >= 1:
            print('start comparison')
            for module1,module2 in zip(model.net.named_parameters(),old_model.net.named_parameters()):
                name1, p1 = module1
                name2, p2 = module2
                print(name1)
                print(p1.equal(p2))
            test_loader_iter = iter(dataset.test_loaders[0])
            inputs, labels = next(test_loader_iter)
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = model(inputs, 0)
            outputs_old = old_model(inputs, 0)
            print(outputs.equal(outputs_old))
        continue
        ###############################################'''
        
        n_parameters = sum(p.numel() for p in model.parameters()) #if p.requires_grad)
        results_parameters.append(n_parameters)
        logger.info('Model parameter: {:.2f}M'.format(n_parameters/1e6))
        
        dices, asds, precisions, recalls = evaluate(model, dataset_copy)
        results.append(dices)
        results_asd.append(asds)
        results_precision.append(precisions)
        results_recall.append(recalls)
        logger.info("\n Dice for current task: {}".format(dices))
        logger.info("\n Mean Dice: {}".format(np.mean(dices[:t+1])))
        logger.info('End task {}'.format(t+1))
        logger.info('-' * 50)
        
        if dataset.SETTING == 'class-il':
            test_loader = dataset.get_whole_testloader()
            model.current_task = dataset.N_TASKS
            dice_whole, asd_whole, precision_whole, recall_whole = evaluate_current_task(model, dataset)
            model.current_task = t
            
            logger.info("\n Dice for whole-heart seg: {}".format(dice_whole))
            logger.info("\n ASD for whole-heart seg: {}".format(asd_whole))
            logger.info("\n precision for whole-heart seg: {}".format(precision_whole))
            logger.info("\n recall for whole-heart seg: {}".format(recall_whole))
            logger.info('End final evaluation')
            results_whole.append(dice_whole)
    ###########################
    #visualization after last model loaded
    if args.save_image:
        _, _ = visualize(model, dataset_copy)
    
    logger.info('Total test Results is: ')
    fwt = forward_transfer(results, dices_random)
    bwt = backward_transfer(results)
    rma = restricted_modelability(results)
    logger.info("\n Mean Dice: {:.4f}".format(np.mean(dices)))
    logger.info("\n Forward Transfer for Dice: {:.6f}".format(fwt))
    logger.info("\n Backward Transfer for Dice: {:.6f}".format(bwt))
    logger.info("\n Restricted Model Ability (RMA) for Dice: {:.6f}".format(rma))
    
    if dataset.SETTING == 'class-il':
        logger.info("\n Dice for WHS: {:.6f}".format(dice_whole[0]))
        fwt_c = forward_transfer_class(results_whole, dice_whole_random)
        logger.info("\n WHS Forward Transfer for Dice: {:.6f}".format(fwt_c))
    logger.info('-' * 50)
    fwt_asd = forward_transfer(results_asd, asds_random)
    bwt_asd = backward_transfer(results_asd)
    logger.info("\n Mean asd: {:.6f}".format(np.mean(asds)))
    logger.info("\n Forward Transfer for asd: {:.6f}".format(fwt_asd))
    logger.info("\n Backward Transfer for asd: {:.6f}".format(bwt_asd))
    logger.info('-' * 50)
    fwt_precision = forward_transfer(results_precision, precisions_random)
    bwt_precision = backward_transfer(results_precision)
    logger.info("\n Mean precision: {:.4f}".format(np.mean(precisions)))
    logger.info("\n Forward Transfer for precision: {:.6f}".format(fwt_precision))
    logger.info("\n Backward Transfer for precision: {:.6f}".format(bwt_precision))
    logger.info('-' * 50)
    fwt_recall = forward_transfer(results_recall, recalls_random)
    bwt_recall = backward_transfer(results_recall)
    logger.info("\n Mean recall: {:.4f}".format(np.mean(recalls)))
    logger.info("\n Forward Transfer for recall: {:.6f}".format(fwt_recall))
    logger.info("\n Backward Transfer for recall: {:.6f}".format(bwt_recall))
    logger.info("\n Results: {}".format(results))

