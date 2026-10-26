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
from models.utils.loss import UnbiasedCrossEntropy #mib
from datasets import get_dataset
import copy
import numpy as np
import sys
import os
import h5py
import matplotlib.pyplot as plt
from datasets.utils.ram_cache import prepare_h5_dataset


def plot_tsne(model: ContinualModel, dataset: ContinualDataset):
    embeddings = []
    labels = []
    label_index = 0


    #############get embeddings###############
    for k, test_loader in enumerate(dataset.test_loaders):
        label_index = k
        for data in test_loader:
            inputs, labels = data
            embeds = inputs.detach().cpu() #TODO: use other embeds
            embeddings.append(embeds)
            labels.append(label_index)


    #####################generate tsne##################
    from sklearn.utils import shuffle
    from sklearn.neighbors import KernelDensity
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import normalize
    from sklearn.manifold import TSNE

    tsne = TSNE(
            n_components=2,
            verbose=1,
            n_iter=1000,
            learning_rate=100,
            perplexity=28,
            early_exaggeration=12,
            angle=0.3,
            #init="pca",
    )
    embeds, labels = shuffle(embeds, labels)
    N = embeds.shape[0]
    tsne_results = tsne.fit_transform(embeds.view(N,-1))
    colors = [ [226/255., 0, 0, 1],  [217/255., 153/255., 0,1], [217/255., 217/255., 0/255.,1], [17/255., 226/255., 56/255.,1], [0., 217/255., 203/255.,1], [0., 57/255., 217/255.,1]]
    legends = [ "ISBI","ISBI_1.5","I2CVB", 'UCL','BIDMC', 'HK']
    map_marker = ["o",'*','d',"+",'P','^']
    (_, ax) = plt.subplots(1)
    for label in torch.unique(labels):
        res = tsne_results[torch.where(labels==label)]
        ax.scatter(*res.T, marker=map_marker[label], alpha = 0.6, label=legends[label], color=colors[label],edgecolors="white")
        ax.legend(loc="best")

    save_tsne_path = os.path.join(dataset.args.output_folder, "tsne.jpg")
    plt.savefig(save_tsne_path)
    plt.close()


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


@torch.no_grad()
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
    ############ For New metrics ##########
    ##MS
    n_parameters = (model.evaluation_parameter_count()
                    if hasattr(model, 'evaluation_parameter_count')
                    else sum(p.numel() for p in model.parameters()))
    ##SSS
    if hasattr(model, 'buffer') and hasattr(model.buffer, 'examples'):
        n_memory = min(model.buffer.num_seen_examples, model.buffer.examples.shape[0])
    else:
        n_memory = 0
    #################################################
    for k, test_loader in enumerate(dataset.test_loaders):
        prepare_h5_dataset(test_loader.dataset)
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
    return dices, asds, precisions, recalls, n_parameters, n_memory

@torch.no_grad()
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
    prepare_h5_dataset(dataset.test_loader.dataset)
    for data in dataset.test_loader:
        inputs, labels = data
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        outputs = model(inputs, t)
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
@torch.no_grad()
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
        prepare_h5_dataset(dataset.next_test_loaders.dataset)
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
    results_whole = []

    ##########
    results_parameters = []
    results_memory = []
    ALL_DATA_SIZE = 0
    results_onepass = []
    ##########

    csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME, args)
    ##############
    model_save_dir = os.path.join(csv_logger.savepath, args.name) if args.output_folder is None else args.output_folder
    create_if_not_exists(model_save_dir)
    if hasattr(model, 'evaluation_parameter_count'):
        # Method-specific CLI options differ: do not append incompatible CSV
        # schemas from new methods to the legacy shared results file.
        csv_logger.savepath = model_save_dir + os.sep
    ##############
    #for forward transfer
    dataset_copy = get_dataset(args)
    for t in range(dataset.N_TASKS):
        model.net.train()
        _, _ = dataset_copy.get_data_loaders(t)
    if model.NAME != 'icarl':# and model.NAME != 'pnn':
        model.net.eval()
        dices_random, asds_random, precisions_random, recalls_random, _, _ = evaluate(model, dataset_copy)
        print("fwt", len(dices_random))

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        logger.info('Begin task {}'.format(t+1))
        model.current_task = t
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders(t) #train loader和test loader与任务无关
        prepare_h5_dataset(train_loader.dataset)
        ALL_DATA_SIZE += len(dataset.train_dataset) + len(dataset.test_dataset)
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        ########################segmentation focused##############
        if args.pseudo_label and t:
            model_copy=copy.deepcopy(model)
            model_copy.net.eval()

        if args.mib and t:
            old_classes = sum(dataset.N_CLASSES_PER_TASK[:t]) + 1
            model.loss = UnbiasedCrossEntropy(
                old_cl=old_classes, reduction='mean'
            )
        ##########################################
        logger.info('current lr: {}'.format(model.opt.param_groups[0]['lr']))
        for param_group in model.opt.param_groups:
            param_group['lr'] = getattr(model, 'task_learning_rate', args.lr)*param_group.get('lr_scale', 1.)
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
            if hasattr(model, 'begin_epoch'):
                model.begin_epoch(epoch, args.n_epochs)
            if model.NAME == 'joint':
                logger.info('For joint model, skip the normal training process')
                break
            for i, data in enumerate(train_loader):
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    #not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    if args.pseudo_label and t:
                        outputs_old = model_copy.forward(inputs, t-1)
                        outputs_old = outputs_old.detach()
                        labels = model.pseudo_wrap(outputs_old, labels)
                    loss = model.observe(inputs, labels, inputs, logits)
                else:
                    inputs, labels = data
                    #print("inputs", inputs.size(), labels.size())
                    inputs, labels = inputs.to(model.device), labels.to(
                        model.device)
                    if args.pseudo_label and t:
                        outputs_old = model_copy.forward(inputs, t-1)
                        outputs_old = outputs_old.detach()
                        labels = model.pseudo_wrap(outputs_old, labels)
                    #not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.observe(inputs, labels, inputs)


                progress_bar(logger, i, len(train_loader), epoch, t, loss)
                #print('break!')
                #break
            ################
            if hasattr(model, 'end_epoch'):
                model.end_epoch()
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
        dices, asds, precisions, recalls, n_parameters, n_memory = evaluate(model, dataset_copy)

        results.append(dices)
        results_asd.append(asds)
        results_precision.append(precisions)
        results_recall.append(recalls)
        results_parameters.append(n_parameters)
        results_memory.append(n_memory)
        logger.info('Model parameter at train step {}: {:.2f}M'.format(t, n_parameters/1e6))
        logger.info('n_memory at train step {}: {}'.format(t, n_memory))
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
    csv_logger.add_rma(results)
    if hasattr(model, 'rma_reference_results'):
        references = model.rma_reference_results(dataset.N_TASKS)
        if references is not None:
            csv_logger.add_rmac(results, references)

    csv_logger.add_ms(results_parameters)
    csv_logger.add_sss(results_memory, ALL_DATA_SIZE)

    csv_logger.add_mean_asd(results_asd)
    csv_logger.add_bwt_asd(results_asd)
    csv_logger.add_fwt_asd(results_asd, asds_random)

    csv_logger.add_mean_precision(results_precision)
    csv_logger.add_bwt_precision(results_precision)
    csv_logger.add_fwt_precision(results_precision, precisions_random)

    csv_logger.add_mean_recall(results_recall)
    csv_logger.add_bwt_recall(results_recall)
    csv_logger.add_fwt_recall(results_recall, recalls_random)
    if dataset.SETTING == 'class-il':
        csv_logger.add_whs(dice_whole)
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

    ##########
    results_parameters = []
    results_memory = []
    ALL_DATA_SIZE = 0
    results_onepass = []
    ##########

    model.net.to(model.device)
    model.net.eval()

    TSNE_ONLY = False

    #for forward transfer
    ###########################
    dataset_copy = get_dataset(args)
    for t in range(dataset.N_TASKS):
        _, _ = dataset_copy.get_data_loaders(t)

    #JUST PLOT TSNE
    if TSNE_ONLY:
        logger.info('Plot tsne for dataset {}'.format(args.dataset))
        plot_tsne(model, dataset_copy)
        return

    if model.NAME != 'icarl': # and model.NAME != 'pnn':
        dices_random, asds_random, precisions_random, recalls_random, _, _ = evaluate(model, dataset_copy)
        print(dices_random)

    if dataset.SETTING == 'class-il':
        test_loader = dataset_copy.get_whole_testloader()
        model.current_task = dataset.N_TASKS
        model.WHS = True
        dice_whole_random, asd_whole_random, precision_whole_random, recall_whole_random = evaluate_current_task(model, dataset_copy)
        model.WHS = False
        model.current_task = 0
    ###########################

    #for backward transfer & mean dices
    ###########################
    for t in range(dataset.N_TASKS):

        logger.info('Begin task {}'.format(t+1))
        _, _ = dataset.get_data_loaders(t)
        ALL_DATA_SIZE += len(dataset.train_dataset) + len(dataset.test_dataset)
        model.current_task = t
        '''old_model = copy.deepcopy(model)
        old_model.net.eval()'''
        if model.NAME in ['pnn','dan']:
            model.begin_task(dataset)
            #old_model.begin_task(dataset)
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

        ############ For New metrics ##########
        ##CE
        '''from calflops import calculate_flops
        input_shape = (args.batch_size, 1, 256, 256)
        flops, _, _ = calculate_flops(model=model,
                                      input_shape=input_shape,
                                      args=[t],
                                      include_backPropagation=True,
                                      output_as_string=True,
                                      output_precision=4)
        results_onepass.append(flops)'''
        ##############################

        dices, asds, precisions, recalls, n_parameters, n_memory = evaluate(model, dataset_copy)
        results.append(dices)
        results_asd.append(asds)
        results_precision.append(precisions)
        results_recall.append(recalls)
        results_parameters.append(n_parameters)
        results_memory.append(n_memory)
        logger.info('Model parameter at infer step {}: {:.2f}M'.format(t, n_parameters/1e6))
        logger.info('n_memory at infer step {}: {}'.format(t, n_memory))
        logger.info("\n Dice for current task: {}".format(dices))
        logger.info("\n Mean Dice: {}".format(np.mean(dices[:t+1])))
        logger.info('End task {}'.format(t+1))
        logger.info('-' * 50)

        if dataset.SETTING == 'class-il':
            test_loader = dataset.get_whole_testloader()
            model.current_task = dataset.N_TASKS
            model.WHS = True
            dice_whole, asd_whole, precision_whole, recall_whole = evaluate_current_task(model, dataset)
            model.WHS = False
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

    sgd_results = {}
    sgd_results['seq-prostate'] = {}
    sgd_results['seq-prostate']['1'] = [[0.8607566689085493, 0.7732968456157211, 0.549047003763808, 0.5029471611562699, 0.5101948564788817, 0.5445123203762846], [0.6352591639514548, 0.8784143696563267, 0.641025908764967, 0.5094007711362822, 0.5079104855400076, 0.528644066819144], [0.577414147339892, 0.6753745779545718, 0.8100085218106002, 0.5858587882035332, 0.6091430936323142, 0.5418427823220229], [0.48923745559025966, 0.49959943509444776, 0.503546618249898, 0.6316902643774003, 0.5164979626520356, 0.49560489082702636], [0.5431794329355325, 0.48971186372672065, 0.6210246968928114, 0.6655029134748335, 0.7584524617328141, 0.487640508441639], [0.5672700726819152, 0.6540342908633214, 0.5869617845920878, 0.5185054019271323, 0.4908353208185514, 0.8459981926725065]]
    sgd_results['seq-prostate']['2'] = [[0.8566224583801326, 0.7442797257198714, 0.5268710076840007, 0.5096005876410084, 0.5076837398322059, 0.5282668956156125], [0.6230227692385435, 0.8306002352833951, 0.5977192815490127, 0.5133325964425037, 0.5022491674431935, 0.5327739847678885], [0.5386860849430193, 0.6208349718879451, 0.7673033787402295, 0.5435671720041618, 0.5971699282489382, 0.5383088799199378], [0.4939164307107539, 0.4965929314306501, 0.5178372537357634, 0.6304799433083456, 0.525404092148706, 0.4949727255863728], [0.5604951751365527, 0.5075439072930781, 0.6357242163370782, 0.6912133689916804, 0.7222972734396353, 0.4880688475672141], [0.5601288873740988, 0.6408496319760442, 0.5876791046839074, 0.5102623228898235, 0.4791061943474805, 0.8275776963430541]]
    sgd_results['seq-prostate']['3'] = [[0.8655225492909148, 0.7645991020549016, 0.5396455292571987, 0.5119634228681738, 0.5175903225417559, 0.5549205803898117], [0.605230612811129, 0.8382624233187607, 0.6397174316721841, 0.5213582240670829, 0.5065754864113887, 0.5257597298152559], [0.5556698408703785, 0.6733666019757557, 0.8398953516376102, 0.6084193657435346, 0.6507577019050228, 0.5478208475213728], [0.492481567805934, 0.49640760239695864, 0.5035952658577639, 0.6530867070832931, 0.5140239870940508, 0.4946062258100903], [0.5533243999910245, 0.5264272635789132, 0.6515733080988693, 0.7161246657723408, 0.7256299260541337, 0.4957622971709843], [0.5831748709813057, 0.6581554084669906, 0.5712834245112279, 0.5119609708144426, 0.48354408007527294, 0.8471901989929667]]
    sgd_results['seq-prostate']['4'] = [[0.851022492943481, 0.7374887438291765, 0.5262750316687728, 0.5140733776292076, 0.5097011573679207, 0.5309826508477803], [0.6018337648610683, 0.8418945530598448, 0.6058770435529958, 0.5029032632539061, 0.5030880127843195, 0.5282367683699396], [0.5643501793400968, 0.7478496780937012, 0.7908610765382424, 0.5608023304837186, 0.580302613523013, 0.5756422921595755], [0.4925277862920545, 0.49575988649488406, 0.5185405750815972, 0.6448961074958126, 0.5229685441617815, 0.49657927448752187], [0.5676329480091535, 0.5093587095702422, 0.6411446374061133, 0.674576324441799, 0.7358385122102267, 0.487211740379771], [0.587940506444135, 0.659507703387564, 0.5717421168191185, 0.5166118121459637, 0.49437978453140163, 0.8318952252352132]]
    sgd_results['seq-prostate']['5'] = [[0.868394819403111, 0.8039902267524326, 0.5498239902510742, 0.5109667605266959, 0.5177842424410511, 0.5536145538992827], [0.627723864535092, 0.8439695272055046, 0.5833843377514878, 0.5134292075609286, 0.4997951924792227, 0.5350134660648713], [0.5716554750563303, 0.7390730247834276, 0.833716127735738, 0.5545976913439365, 0.6402727059419578, 0.5701896833043725], [0.49235867493229174, 0.4967535688646742, 0.5039726240579174, 0.6438046827084181, 0.5122189346815265, 0.4949670551684388], [0.5690125448485578, 0.5022942963921949, 0.6255353562757133, 0.6820913366871311, 0.7070059057492908, 0.4893924239409625], [0.5761055250735045, 0.6330210706667826, 0.5845150974051867, 0.5158236872992721, 0.4829829241753497, 0.8441900377853013]]

    sgd_results['seq-mmwhs'] = {}
    '''sgd_results['seq-mmwhs']['1'] = [[0.7914511612912153, 0.330286029660472, 0.3327776943923935], [0.24326167465037624, 0.7783279756548122, 0.3327776943923935], [0.2444336299729285, 0.3298531644311845, 0.8086819726843394]]
    sgd_results['seq-mmwhs']['2'] = [[0.7939775240727844, 0.330286029660472, 0.3327776943923935], [0.2434454016612594, 0.7416742560068305, 0.3327776943923935], [0.24441571508028093, 0.3300821252206179, 0.8032772974531985]]
    sgd_results['seq-mmwhs']['3'] = [[0.7889707127100029, 0.330286029660472, 0.3327776943923935], [0.24346739559799202, 0.7758087159549701, 0.3327776943923935], [0.24426830051350218, 0.3301930237226529, 0.7965671454572684]]
    sgd_results['seq-mmwhs']['4'] = [[0.7976454973333083, 0.330286029660472, 0.3327776943923935], [0.2432394894062032, 0.746380319900234, 0.3327776943923935], [0.24438793930923178, 0.33032572007965016, 0.7690857553810435]]
    sgd_results['seq-mmwhs']['5'] = [[0.8057921645524304, 0.330286029660472, 0.3327776943923935], [0.24285006699567002, 0.7429447422689228, 0.3327776943923935], [0.2445202293873155, 0.3298820404913695, 0.805321295203747]]'''
    sgd_results['seq-mmwhs']['1'] = [[0.8064179817637024, 0.330286029660472, 0.3327776943923935], [0.24412134722087864, 0.7191993257841455, 0.3327776943923935], [0.24456317089342125, 0.3302086360641338, 0.7760767657883948]]
    sgd_results['seq-mmwhs']['2'] = [[0.7729687183459275, 0.330286029660472, 0.3327776943923935], [0.24460505694559348, 0.7671530240082897, 0.3327776943923935], [0.24468497146729679, 0.3302242717504526, 0.7605477192570942]]
    sgd_results['seq-mmwhs']['3'] = [[0.8033155526751545, 0.330286029660472, 0.3327776943923935], [0.2436949995365097, 0.7082811786156603, 0.3327776943923935], [0.24438338911267776, 0.3302794859049099, 0.7669207605626394]]
    sgd_results['seq-mmwhs']['4'] = [[0.8010452209488959, 0.330286029660472, 0.3327776943923935], [0.24400048709655087, 0.7556823116510039, 0.3327776943923935], [0.2445145991034057, 0.330286029660472, 0.8092383889414766]]
    sgd_results['seq-mmwhs']['5'] = [[0.8038565806373548, 0.330286029660472, 0.3327776943923935], [0.24371834815988835, 0.7195413474010698, 0.3327776943923935], [0.24462729420649987, 0.33028405180966175, 0.797573997265031]]

    sgd_results['seq-taskincre'] = {}
    sgd_results['seq-taskincre']['1'] = [[0.8285384890213633, 0.07607171449803402, 0.0679168269248686, 0.49509392446481454], [0.02628027344669766, 0.6350154197391282, 0.4822608408169593, 0.39379464934867786], [0.0151897706646817, 0.49624558067053914, 0.960176164459756, 0.0384844405748447], [0.4762063685880113, 0.49360288402589514, 0.17171004696129002, 0.9201768203311004]]
    sgd_results['seq-taskincre']['2'] =  [[0.8410355073868795, 0.496686035650274, 0.44088025191748337, 0.5449831269805312], [0.4507259945395778, 0.6136893718886469, 0.48210324678793454, 0.520888506147734], [0.04070760030440086, 0.4941609097193271, 0.9615133463788929, 0.5424895055804412], [0.4983244186522665, 0.39168993238736766, 0.4821173774503699, 0.9196753083291476]]
    sgd_results['seq-taskincre']['3'] =  [[0.8456958116878737, 0.49614012866293294, 0.4850940910939747, 0.5025251005990641], [0.4533878005352573, 0.6263075570997487, 0.48194970445188534, 0.5027297773330581], [0.36372614635456435, 0.14971176357200083, 0.9623362986031597, 0.04277812516994772], [0.31296527112612393, 0.029100675761901763, 0.06491877807593656, 0.9214251228831596]]
    sgd_results['seq-taskincre']['4'] =  [[0.8188159902445052, 0.014381543121186577, 0.48124041109020166, 0.04476265590540528], [0.06129088622505528, 0.6340822007657, 0.2342032181936227, 0.4871191104148084], [0.4984832894882666, 0.02244916173253246, 0.9591879406109732, 0.5164322767698736], [0.4438710486961619, 0.4186419866512799, 0.47976694650802765, 0.9219488670590592]]
    sgd_results['seq-taskincre']['5'] = [[0.8564878586509556, 0.04437304172373172, 0.2773303086532, 0.49588409629174], [0.1763468593726582, 0.6374238230701827, 0.48170460860903175, 0.04824264764492274], [0.49035204644769087, 0.49646770150331815, 0.9612708925388093, 0.03692647823175568], [0.49772392573419566, 0.05719952831723282, 0.1663267781370168, 0.9184467345551407]]

    logger.info('Total test Results is: ')
    fwt = forward_transfer(results, dices_random)
    bwt = backward_transfer(results)
    rma = restricted_modelability(results)
    if hasattr(model, 'rma_reference_results'):
        references = model.rma_reference_results(dataset.N_TASKS)
        rma_c = (restricted_modelability_corrected(results, references)
                 if references is not None else None)
    else:
        references = sgd_results[args.dataset][str(args.seed)]
        rma_c = restricted_modelability_corrected(results, references)
    ###############################
    ms = model_size_efficiency(results_parameters)
    sss = samples_storage_size_efficiency(results_memory, ALL_DATA_SIZE)
    ###############################
    logger.info("\n Mean Dice: {:.4f}".format(np.mean(dices)))
    logger.info("\n Forward Transfer for Dice: {:.6f}".format(fwt))
    logger.info("\n Backward Transfer for Dice: {:.6f}".format(bwt))
    logger.info("\n Restricted Model Ability (RMA) for Dice: {:.6f}".format(rma))
    if rma_c is not None:
        logger.info("\n Corrected Restricted Model Ability (RMA) for Dice: {:.6f}".format(rma_c))
    else:
        logger.info('Normalized RMA unavailable: provide matched --rma_reference; '
                    'legacy hard-coded SGD results were not used for this method.')
    ##############################
    logger.info("\n Model size efficiency: {:.6f}".format(ms))
    logger.info("\n Samples storage size efficiency: {:.6f}".format(sss))
    logger.info("\n Number of operations (per pass): {}".format(results_onepass))
    ##############################

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

    ###################CSV LOGGER###################
    csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME, args, savename = "/test_results_taskincre.csv")
    if hasattr(model, 'evaluation_parameter_count'):
        csv_logger.savepath = (args.output_folder or args.pretrain) + os.sep
    csv_logger.add_mean_dice(results)
    csv_logger.add_bwt(results)
    csv_logger.add_fwt(results, dices_random)
    csv_logger.add_rma(results)
    if references is not None:
        csv_logger.add_rmac(results, references)

    csv_logger.add_ms(results_parameters)
    csv_logger.add_sss(results_memory, ALL_DATA_SIZE)

    csv_logger.add_mean_asd(results_asd)
    csv_logger.add_bwt_asd(results_asd)
    csv_logger.add_fwt_asd(results_asd, asds_random)

    csv_logger.add_mean_precision(results_precision)
    csv_logger.add_bwt_precision(results_precision)
    csv_logger.add_fwt_precision(results_precision, precisions_random)

    csv_logger.add_mean_recall(results_recall)
    csv_logger.add_bwt_recall(results_recall)
    csv_logger.add_fwt_recall(results_recall, recalls_random)
    if dataset.SETTING == 'class-il':
        csv_logger.add_whs(dice_whole)
    csv_logger.add_fwt_recall(results_recall, recalls_random)
    csv_logger.write(vars(args))

