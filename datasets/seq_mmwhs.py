import h5py
from types import MethodType
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import random
from backbone.ResUnet import resunet32
from backbone.ResUnet_wD import resunet32_withdict
import cv2
import torch.nn.functional as F
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders

class MMWHSCL_Dataset(Dataset):
    def __init__(self, task_name, mode, patch_size):
        self.shift = 0
        self.task = task_name
        root_heart = "/home/zhouhangqi/zhq/data/MMWHS/"
        self.mode = mode
        if task_name == "1":
            task_h5 = os.path.join(root_heart, "myo_lv_la" + ".h5")  # BIDMC
            self.shift = 0
        elif task_name == "2":
            task_h5 = os.path.join(root_heart, "ra_rv" + ".h5")  # UCL
            self.shift = 3
        elif task_name == "3":
            task_h5 = os.path.join(root_heart, "ao_pa" + ".h5")  # ISBI
            self.shift = 5
        elif task_name == "whole_heart":
            task_h5 = os.path.join(root_heart, "whole_heart_test" + ".h5")  # ISBI
            self.shift = 0
        self.task_name = task_name
        data_file = h5py.File(task_h5, mode='r')
        if mode == "train":
            self.images_all = data_file["train_images"]
            self.labels_all = data_file["train_labels"]
            self.patient_info = list(data_file["patient_info_val"])
        elif mode == "val":
            self.images_all = data_file["val_images"]
            self.labels_all = data_file["val_labels"]
            self.patient_info = list(data_file["patient_info_val"])
        else:
            self.images_all = data_file["test_images"]
            self.labels_all = data_file["test_labels"]
            self.patient_info = list(data_file["patient_info_test"])
        self.patch_size = patch_size
        
    def label_shift(self, labels):
        return (labels>0)*self.shift + labels

    def __getitem__(self, index):
        image = self.images_all[:, :, index]
        target = self.labels_all[:, :, index]
        image_patch = cv2.resize(image, (self.patch_size, self.patch_size))
        target_patch = cv2.resize(target, (self.patch_size, self.patch_size))
        # image_patch, target_patch = get_patch(image_patch, target_patch, self.patch_size)
        image_patch = torch.from_numpy(image_patch).float()
        image_patch = torch.unsqueeze(image_patch, dim=0)
        target_patch = torch.from_numpy(target_patch).long()
        if self.mode == 'train':
            target_patch = self.label_shift(target_patch)
        if hasattr(self, 'logits'):
            return image_patch, target_patch, self.logits[index]
        return image_patch, target_patch

    def __len__(self):
        return self.images_all.shape[2]

def get_patch(img_in, img_tar, patch_size):
    ih, iw = img_in.shape[:2]
    tp = patch_size
    ip = tp
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = ix, iy
    img_in = img_in[iy:iy + ip, ix:ix + ip]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp]

    return img_in, img_tar

def MMWHS_Gen(task_names, dataset, patch_size=256, test = False):
    train_dataset_splits = {}
    val_dataset_splits = {}
    for i, task_name in enumerate(task_names):
        train_dataset_splits[task_name] = dataset(task_name, "train", patch_size)
        val_dataset_splits[task_name] = dataset(task_name, "test", patch_size) if test else dataset(task_name, "val", patch_size)
    return train_dataset_splits, val_dataset_splits


class SequentialMMWHS(ContinualDataset):

    NAME = 'seq-mmwhs'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = [3,2,2]
    N_TASKS = 3

    def get_data_loaders(self, t):
        train_splits, test_splits = MMWHS_Gen(["1", "2", "3"], dataset=MMWHSCL_Dataset)
        train_dataset = train_splits[str(t +1)]
        test_dataset = test_splits[str(t +1)]
        if t < 2:
            next_test_dataset = test_splits[str( t +2)]
            self.next_test_loaders = DataLoader(next_test_dataset,
                                                batch_size=1, shuffle=False)
        else:
            next_test_dataset = None
            self.next_test_loaders = None
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset,
                                 batch_size=1, shuffle=False)
        self.train_dataset = train_dataset
        self.test_loaders.append(test_loader)
        self.train_loader = train_loader
        self.test_loader = test_loader

        return train_loader, test_loader


    def not_aug_dataloader(self, batch_size):

        # return self.train_loader
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
    
    def get_whole_testloader(self):
        test_dataset = MMWHSCL_Dataset(task_name = "whole_heart", mode = 'test', patch_size = 256)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        #self.test_loaders.append(test_loader)
        self.test_loader = test_loader
        return test_loader


    @staticmethod
    def get_backbone(args):
        if 'gpm' in args.model:
            model = resunet32_withdict(args.inputsize, size = args.baseline_size)
        else:
            model = resunet32(size = args.baseline_size)

        lastn = model.last.in_channels
        if args.multihead:
            N_TASKS = 3
            N_CLASSES_PER_TASK = [3,2,2]
            model.last = nn.ModuleDict()
            total_class = 1 #background
            for task in range(N_TASKS):
                #total_class = total_class + N_CLASSES_PER_TASK[task]
                model.last[str(task)] = nn.Conv2d(lastn, 8, 1, 1, bias=False)
            def new_logits(self, x, k):
                return self.last[str(k)](x)
            model.logits = MethodType(new_logits, model)

            
        model.last = nn.Conv2d(lastn, 8, 1, 1, bias=False)
        
        return model

    @staticmethod
    def get_loss():
        return F.cross_entropy