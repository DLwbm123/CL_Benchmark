# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch.optim import SGD
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device, base_path
from utils import create_if_not_exists
import os
from torch.nn.parallel import DistributedDataParallel
from collections import OrderedDict


def entropy(probabilities):
    """Computes the entropy per pixel.
    # References:
        * ESL: Entropy-guided Self-supervised Learning for Domain Adaptation in Semantic Segmentation
          Saporta et al.
          CVPR Workshop 2020
    :param probabilities: Tensor of shape (b, c, w, h).
    :return: One entropy per pixel, shape (b, w, h)
    """
    factor = 1 / math.log(probabilities.shape[1] + 1e-8)
    return -factor * torch.mean(probabilities * torch.log(probabilities + 1e-8), dim=1)



class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                args: Namespace, transform: torchvision.transforms) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=args.lr_decay_epoch, gamma=args.lr_decay_rate, last_epoch=-1)
        self.current_task = 0

        self.device = get_device(args.GPU_ids)

    def forward(self, x: torch.Tensor, task_index = None) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        if task_index is None:
            task_index = self.current_task
        return self.net(x, task_index)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass

    def save_network(self, network, savepath, task_label, iter_label):
        save_filename = 'T{}_{}.pth'.format(task_label, iter_label)
        save_path = os.path.join(savepath, save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)
        
    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)
        
    def pseudo_wrap(self, outputs_old, targets):
        '''
        ref: https://github.com/arthurdouillard/CVPR2021_PLOP
        '''
        mask_background = (targets==0)
        probs = torch.softmax(outputs_old, dim=1)
        if self.args.pseudo_label == 'naive':
            targets[mask_background] = outputs_old.argmax(dim=1)[mask_background]
        elif self.args.pseudo_label == "entropy":
            probs = torch.softmax(outputs_old, dim=1)
            max_probs, pseudo_labels = probs.max(dim=1)
            mask_valid_pseudo = (entropy(probs)/self.max_entropy) < self.thresholds[pseudo_labels]

        return targets
    
    
    def find_median(self, train_loader, logger):
        """Find the median prediction score per class with the old model.
        Computing the median naively uses a lot of memory, to allievate it, instead
        we put the prediction scores into a histogram bins and approximate the median.
        https://math.stackexchange.com/questions/2591946/how-to-find-median-from-a-histogram
        """
        nb_current_classes = sum(train_loader.dataset.N_CLASSES_PER_TASK[:self.current_task-1]) + 1
        
        mode = self.args.pseudo_label
        if mode == "entropy":
            max_value = torch.log(torch.tensor(self.nb_current_classes).float().to(device))
            nb_bins = 100
        else:
            max_value = 1.0
            nb_bins = 20  # Bins of 0.05 on a range [0, 1]

        histograms = torch.zeros(nb_current_classes, nb_bins).long().to(self.device)

        for cur_step, data in enumerate(train_loader):
            if hasattr(train_loader.dataset, 'logits'):
                images, labels, logits = data
            else:
                images, labels = data
            images = images.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)

            outputs_old= self.forward(images, self.current_task-1)

            mask_bg = labels == 0
            probas = torch.softmax(outputs_old, dim=1)
            max_probas, pseudo_labels = probas.max(dim=1)

            if mode == "entropy":
                values_to_bins = entropy(probas)[mask_bg].view(-1) / max_value
            else:
                values_to_bins = max_probas[mask_bg].view(-1)

            x_coords = pseudo_labels[mask_bg].view(-1)
            y_coords = torch.clamp((values_to_bins * nb_bins).long(), max=nb_bins - 1)

            histograms.index_put_(
                (x_coords, y_coords),
                torch.LongTensor([1]).expand_as(x_coords).to(histograms.device),
                accumulate=True
            )

            if cur_step % 10 == 0:
                logger.info(f"Median computing {cur_step}/{len(train_loader)}.")

        thresholds = torch.zeros(nb_current_classes, dtype=torch.float32).to(
            self.device
        )  # zeros or ones? If old_model never predict a class it may be important

        logger.info("Approximating median")
        for c in range(nb_current_classes):
            total = histograms[c].sum()
            if total <= 0.:
                continue

            half = total / 2
            running_sum = 0.
            for lower_border in range(nb_bins):
                lower_border = lower_border / nb_bins
                bin_index = int(lower_border * nb_bins)
                if half >= running_sum and half <= (running_sum + histograms[c, bin_index]):
                    break
                running_sum += lower_border * nb_bins

            median = lower_border + ((half - running_sum) /
                                     histograms[c, bin_index].sum()) * (1 / nb_bins)

            thresholds[c] = median

        base_threshold = 0.001
        if "_" in mode:
            mode, base_threshold = mode.split("_")
            base_threshold = float(base_threshold)

        if mode == "entropy":
            for c in range(len(thresholds)):
                thresholds[c] = max(thresholds[c], base_threshold)
        else:
            for c in range(len(thresholds)):
                thresholds[c] = min(thresholds[c], base_threshold)
        logger.info(f"Finished computing median {thresholds}")
        return thresholds.to(device), max_value
    
    
