import numpy as np
import torch
import logging
from types import MethodType
import torch.nn.functional as F
from models.utils.continual_model import ContinualModel
import time
from utils.args import *
import copy
import json
import math
import torch.nn as nn
#modified from https://github.com/arthurdouillard/CVPR2021_PLOP

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Gradient Episodic Memory.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument(
        "--pod",default="local",type=str,choices=["spatial", "local", "global"]
    )
    parser.add_argument("--pod_factor", default=0.01, type=float)
    parser.add_argument("--pod_options", default= {"switch": {"after": {"extra_channels": "sum", "factor": 0.0005, "type": "local"}}}, type=json.loads)
    parser.add_argument("--pod_prepro", default="pow", type=str)
    parser.add_argument("--threshold", default=0.001)
    return parser



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


def _local_pod(x, spp_scales=[1, 2, 4], normalize=False, normalize_per_scale=False):
    b = x.shape[0]
    w = x.shape[-1]
    emb = []

    for scale_index, scale in enumerate(spp_scales):
        k = w // scale

        nb_regions = scale**2

        for i in range(scale):
            for j in range(scale):
                tensor = x[..., i * k:(i + 1) * k, j * k:(j + 1) * k]

                horizontal_pool = tensor.mean(dim=3).view(b, -1)
                vertical_pool = tensor.mean(dim=2).view(b, -1)

                if normalize_per_scale is True:
                    horizontal_pool = horizontal_pool / nb_regions
                    vertical_pool = vertical_pool / nb_regions
                elif normalize_per_scale == "spm":
                    if scale_index == 0:
                        factor = 2 ** (len(spp_scales) - 1)
                    else:
                        factor = 2 ** (len(spp_scales) - scale_index)
                    horizontal_pool = horizontal_pool / factor
                    vertical_pool = vertical_pool / factor

                if normalize:
                    horizontal_pool = F.normalize(horizontal_pool, dim=1, p=2)
                    vertical_pool = F.normalize(vertical_pool, dim=1, p=2)

                emb.append(horizontal_pool)
                emb.append(vertical_pool)

    return torch.cat(emb, dim=1)


def features_distillation(
    list_attentions_a,
    list_attentions_b,
    collapse_channels="spatial",
    normalize=True,
    labels=None,
    index_new_class=None,
    pod_apply="all",
    pod_deeplab_mask=False,
    pod_deeplab_mask_factor=None,
    interpolate_last=False,
    pod_factor=1.,
    prepro="pow",
    deeplabmask_upscale=True,
    spp_scales=[1, 2, 4],
    pod_options=None,
    outputs_old=None,
    use_pod_schedule=True,
    nb_current_classes=-1,
    nb_new_classes=-1
):
    """A mega-function comprising several features-based distillation.

    :param list_attentions_a: A list of attention maps, each of shape (b, n, w, h).
    :param list_attentions_b: A list of attention maps, each of shape (b, n, w, h).
    :param collapse_channels: How to pool the channels.
    :param memory_flags: Integer flags denoting exemplars.
    :param only_old: Only apply loss to exemplars.
    :return: A float scalar loss.
    """
    device = list_attentions_a[0].device

    assert len(list_attentions_a) == len(list_attentions_b)
    #print(len(list_attentions_a))

    if pod_deeplab_mask_factor is None:
        pod_deeplab_mask_factor = pod_factor

    #if collapse_channels in ("spatial_tuple", "spp", "spp_noNorm", "spatial_noNorm"):
    normalize = True

    apply_mask = "background"
    upscale_mask_topk = 1
    mask_position = "last"  # Others choices "all" "backbone"
    use_adaptative_factor = False
    mix_new_old = None

    loss = torch.tensor(0.).to(list_attentions_a[0].device)
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        adaptative_pod_factor = 1.0
        difference_function = "frobenius"
        pool = True
        use_adaptative_factor = False
        handle_extra_channels = "sum"
        normalize_per_scale = False

        if pod_options and pod_options.get("switch"):
            if i < len(list_attentions_a) - 1:
                if "before" in pod_options["switch"]:
                    collapse_channels = pod_options["switch"]["before"].get(
                        "type", collapse_channels
                    )
                    pod_factor = pod_options["switch"]["before"].get("factor", pod_factor)
                    normalize = pod_options["switch"]["before"].get("norm", False)
                    prepro = pod_options["switch"]["before"].get("prepro", prepro)
                    use_adaptative_factor = pod_options["switch"]["before"].get(
                        "use_adaptative_factor", use_adaptative_factor
                    )
            else:
                if "after" in pod_options["switch"]:
                    collapse_channels = pod_options["switch"]["after"].get(
                        "type", collapse_channels
                    )
                    pod_factor = pod_options["switch"]["after"].get("factor", pod_factor)
                    normalize = pod_options["switch"]["after"].get("norm", False)
                    prepro = pod_options["switch"]["after"].get("prepro", prepro)

                    apply_mask = pod_options["switch"]["after"].get("apply_mask", apply_mask)
                    upscale_mask_topk = pod_options["switch"]["after"].get(
                        "upscale_mask_topk", upscale_mask_topk
                    )
                    use_adaptative_factor = pod_options["switch"]["after"].get(
                        "use_adaptative_factor", use_adaptative_factor
                    )
                    mix_new_old = pod_options["switch"]["after"].get("mix_new_old", mix_new_old)

                    handle_extra_channels = pod_options["switch"]["after"].get(
                        "extra_channels", handle_extra_channels
                    )
                    spp_scales = pod_options["switch"]["after"].get(
                        "spp_scales", spp_scales
                    )
                    use_pod_schedule = pod_options["switch"]["after"].get(
                        "use_pod_schedule", use_pod_schedule
                    )

            mask_position = pod_options["switch"].get("mask_position", mask_position)
            normalize_per_scale = pod_options["switch"].get(
                "normalize_per_scale", normalize_per_scale
            )
            pool = pod_options.get("pool", pool)

        if a.shape[1] != b.shape[1]:
            assert i == len(list_attentions_a) - 1
            assert a.shape[0] == b.shape[0]
            assert a.shape[2] == b.shape[2]
            assert a.shape[3] == b.shape[3]

            assert handle_extra_channels in ("trim", "sum"), handle_extra_channels

            if handle_extra_channels == "sum":
                _b = torch.zeros_like(a).to(a.dtype).to(a.device)
                _b[:, 0] = b[:, 0] + b[:, index_new_class:].sum(dim=1)
                _b[:, 1:] = b[:, 1:index_new_class]
                b = _b
            elif handle_extra_channels == "trim":
                b = b[:, :index_new_class]

        # shape of (b, n, w, h)
        assert a.shape == b.shape, (a.shape, b.shape)

        if not pod_deeplab_mask and use_adaptative_factor:
            adaptative_pod_factor = (labels == 0).float().mean()

        if prepro == "pow":
            a = torch.pow(a, 2)
            b = torch.pow(b, 2)
        elif prepro == "none":
            pass
        elif prepro == "abs":
            a = torch.abs(a, 2)
            b = torch.abs(b, 2)
        elif prepro == "relu":
            a = torch.clamp(a, min=0.)
            b = torch.clamp(b, min=0.)

        a = _local_pod(a, spp_scales, normalize=False, normalize_per_scale=normalize_per_scale)
        b = _local_pod(b, spp_scales, normalize=False, normalize_per_scale=normalize_per_scale)

        if i == len(list_attentions_a) - 1 and pod_options is not None:
            if "difference_function" in pod_options:
                difference_function = pod_options["difference_function"]
        elif pod_options is not None:
            if "difference_function_all" in pod_options:
                difference_function = pod_options["difference_function_all"]

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        if difference_function == "frobenius":
            if isinstance(a, list):
                layer_loss = torch.tensor(
                    [torch.frobenius_norm(aa - bb, dim=-1) for aa, bb in zip(a, b)]
                ).to(device)
            else:
                layer_loss = torch.frobenius_norm(a - b, dim=-1)
            #print(layer_loss)
        elif difference_function == "frobenius_mix":
            layer_loss_old = torch.frobenius_norm(a[0] - b[0], dim=-1)
            layer_loss_new = torch.frobenius_norm(a[1] - b[1], dim=-1)

            layer_loss = mix_new_old * layer_loss_old + (1 - mix_new_old) * layer_loss_new
        elif difference_function == "l1":
            if isinstance(a, list):
                layer_loss = torch.tensor(
                    [torch.norm(aa - bb, p=1, dim=-1) for aa, bb in zip(a, b)]
                ).to(device)
            else:
                layer_loss = torch.norm(a - b, p=1, dim=-1)
        elif difference_function == "kl":
            d1, d2, d3 = a.shape
            a = (a.view(d1 * d2, d3) + 1e-8).log()
            b = b.view(d1 * d2, d3) + 1e-8

            layer_loss = F.kl_div(a, b, reduction="none").view(d1, d2, d3).sum(dim=(1, 2))
        elif difference_function == "bce":
            d1, d2, d3 = a.shape
            layer_loss = bce(a.view(d1 * d2, d3), b.view(d1 * d2, d3)).view(d1, d2,
                                                                            d3).mean(dim=(1, 2))
        else:
            raise NotImplementedError(f"Unknown difference_function={difference_function}")

        assert torch.isfinite(layer_loss).all(), layer_loss
        assert (layer_loss >= 0.).all(), layer_loss

        layer_loss = torch.mean(adaptative_pod_factor * layer_loss)
        if pod_factor <= 0.:
            continue

        layer_loss = pod_factor * layer_loss
        if use_pod_schedule:
            layer_loss = layer_loss * math.sqrt(nb_current_classes / nb_new_classes)
        loss += layer_loss
        

    return loss / len(list_attentions_a)





class PLOP(ContinualModel):
    NAME = 'plop'
    COMPATIBILITY = ['class-il']

    def __init__(self, backbone, loss, args, transform):
        super(PLOP, self).__init__(backbone, loss, args, transform)
        self.current_task = 0
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.pod = args.pod
        self.pod_options = args.pod_options
        self.pod_factor = args.pod_factor
        self.pod_prepro = args.pod_prepro
        
        self.threshold = args.threshold
        self.classif_adaptive_factor = True
        self.classif_adaptive_min_factor = 0.0
        self.nb_current_classes = 1
        self.old_classes = 1
        self.model_old = None
        self.ret_intermediate = True
        
        def new_features(self, x, ret_intermediate = False):
        # Encode
        
            x1 = self.conv1(x)
            x1 = self.bn1(x1)
            x1 = F.relu(x1)
            x1 = self.conv2(x1)
            #x1 = self.input_layer(x)
            x2 = self.residual_conv_1(x1)
            x3 = self.residual_conv_2(x2)
            # Bridge
            x4 = self.bridge(x3)
            # Decode
            x4 = self.upsample_1(x4)
            x5 = torch.cat([x4, x3], dim=1)

            x6 = self.up_residual_conv1(x5)

            x6 = self.upsample_2(x6)
            x7 = torch.cat([x6, x2], dim=1)

            x8 = self.up_residual_conv2(x7)

            x8 = self.upsample_3(x8)
            x9 = torch.cat([x8, x1], dim=1)

            x10 = self.up_residual_conv3(x9)

            #output = self.output_layer(x10)
            if ret_intermediate:
                midfeatures = [x1,x2,x3,x4,x5,
                             x6,x7,x8,x9,x10]
                return x10, midfeatures
            return x10
        self.net.features = MethodType(new_features, self.net)
        
        
    def begin_task(self, dataset):
        ########3确定当前class数#####
        #0:background
        self.old_classes = self.nb_current_classes
        self.nb_new_classes = dataset.N_CLASSES_PER_TASK[self.current_task]
        self.nb_current_classes += self.nb_new_classes
        logger = logging.getLogger('base')
        if self.current_task > 0:
            train_loader, test_loader = dataset.get_data_loaders(self.current_task)
            self.thresholds, self.max_entropy = self.find_median(train_loader, self.device, logger, mode="entropy")

    def end_task(self, dataset):
        self.model_old = copy.deepcopy(self.net)
        self.current_task += 1
        
    def find_median(self, train_loader, device, logger, mode="probability"):
        """Find the median prediction score per class with the old model.

        Computing the median naively uses a lot of memory, to allievate it, instead
        we put the prediction scores into a histogram bins and approximate the median.

        https://math.stackexchange.com/questions/2591946/how-to-find-median-from-a-histogram
        """
        if mode == "entropy":
            max_value = torch.log(torch.tensor(self.nb_current_classes).float().to(device))
            nb_bins = 100
        else:
            max_value = 1.0
            nb_bins = 20  # Bins of 0.05 on a range [0, 1]
        #if self.pseudo_nb_bins is not None:
        #    nb_bins = self.pseudo_nb_bins

        histograms = torch.zeros(self.nb_current_classes, nb_bins).long().to(self.device)

        for cur_step, (images, labels) in enumerate(train_loader):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs_old = self.model_old(images, self.current_task)

            mask_bg = labels == 0
            probas = torch.softmax(outputs_old[:,:self.old_classes], dim=1)
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

        thresholds = torch.zeros(self.old_classes, dtype=torch.float32).to(
            self.device
        )  # zeros or ones? If old_model never predict a class it may be important

        logger.info("Approximating median")
        for c in range(self.old_classes):
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

        base_threshold = self.threshold
        if "_" in mode:
            mode, base_threshold = mode.split("_")
            base_threshold = float(base_threshold)
        #if self.step_threshold is not None:
        #    self.threshold += self.step * self.step_threshold

        if mode == "entropy":
            for c in range(len(thresholds)):
                thresholds[c] = max(thresholds[c], base_threshold)
        else:
            for c in range(len(thresholds)):
                thresholds[c] = min(thresholds[c], base_threshold)
        logger.info(f"Finished computing median {thresholds}")
        return thresholds.to(device), max_value
        
    def observe(self, inputs, labels, not_aug_inputs):


        # now compute the grad on the current data
        self.opt.zero_grad()
        classif_adaptive_factor = 1.0
        
        if self.model_old is not None and self.current_task > 0: #伪标签策略
            x10_old, features_old = self.model_old.features(inputs, ret_intermediate=self.ret_intermediate)
            outputs_old = self.model_old.logits(x10_old,self.current_task-1)
            mask_background = labels < self.old_classes
            probs = torch.softmax(outputs_old[:,:self.old_classes], dim=1)
            max_probs, pseudo_labels = probs.max(dim=1)
            
            mask_valid_pseudo = (entropy(probs)/self.max_entropy) < self.thresholds[pseudo_labels]
            labels[~mask_valid_pseudo & mask_background] = 0
            labels[mask_valid_pseudo & mask_background] = pseudo_labels[mask_valid_pseudo & mask_background]
            
            if self.classif_adaptive_factor:
                # Number of old/bg pixels that are certain
                num = (mask_valid_pseudo & mask_background).float().sum(dim=(1,2))
                # Number of old/bg pixels
                den =  mask_background.float().sum(dim=(1,2))
                # If all old/bg pixels are certain the factor is 1 (loss not changed)
                # Else the factor is < 1, i.e. the loss is reduced to avoid
                # giving too much importance to new pixels
                classif_adaptive_factor = num / (den + 1e-6)
                classif_adaptive_factor = classif_adaptive_factor[:, None, None]

                if self.classif_adaptive_min_factor:
                    classif_adaptive_factor = classif_adaptive_factor.clamp(min=self.classif_adaptive_min_factor)

        x10, features = self.net.features(inputs, ret_intermediate=self.ret_intermediate)
        outputs = self.net.logits(x10,self.current_task)
        
        
        loss = self.loss(outputs, labels)
        loss = classif_adaptive_factor*loss
        loss = loss.mean()
        
        pod_loss = 0
        if self.pod is not None and self.current_task > 0:
            attentions_old = features_old
            attentions_new = features
            attentions_old.append(outputs_old[:,1:self.old_classes]) #
            attentions_new.append(outputs[:,1:self.old_classes]) #only choose old_classes
            
            pod_loss = features_distillation(
                    attentions_old,
                    attentions_new,
                    collapse_channels=self.pod,
                    labels=labels,
                    index_new_class=self.old_classes,
                    interpolate_last=False,
                    pod_factor=self.pod_factor,
                    prepro=self.pod_prepro,
                    spp_scales=[1, 2, 4],
                    pod_options=self.pod_options,
                    outputs_old=outputs_old,
                    nb_current_classes=self.nb_current_classes,
                    nb_new_classes=self.nb_new_classes
                )

            #pod_loss = pod_loss
        
        
        loss_tot = loss + pod_loss
        loss_tot.backward()
        self.opt.step()

        return loss_tot.item()
    

