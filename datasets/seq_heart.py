import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import random
from backbone.ResUnet import resunet32
import cv2
import torch.nn.functional as F
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders


class ProstateCL_Dataset(Dataset):
    def __init__(self, task_name, mode, patch_size):
        self.task = task_name
        root = "/home/wangbomin/MnMs/"
        root_pro = "/home/wbm/Research/Continual Learning/Prostate_CL"
        root_other = "/home/wbm/Research/Continual Learning/Others/"
        if task_name == "1":
            task_h5 = os.path.join(root, "lv" + ".h5") #BIDMC
        elif task_name == "2":
            task_h5 = os.path.join(root, "myo" + ".h5") #UCL
        elif task_name == "3":
            task_h5 = os.path.join(root, "rv" + ".h5") #ISBI
        '''elif task_name == "4":
            task_h5 = os.path.join(root_pro, "BIDMC" + ".h5")  # UCL
        elif task_name == "5":
            task_h5 = os.path.join(root_pro, "I2CVB" + ".h5")  # ISBI'''
        self.task_name = task_name
        data_file = h5py.File(task_h5, mode='r')
        if mode == "train":
            self.images_all = data_file["train_images"]
            self.labels_all = data_file["train_labels"]
        elif mode == "val":
            self.images_all = data_file["val_images"]
            self.labels_all = data_file["val_labels"]
        self.patch_size = patch_size

    def __getitem__(self, index):
        image = self.images_all[:, :, index]
        target = self.labels_all[:, :, index]
        image = cv2.resize(image, (self.patch_size, self.patch_size))
        target = cv2.resize(target, (self.patch_size, self.patch_size))
        image_patch, target_patch = get_patch(image, target, self.patch_size)
        image_patch = torch.from_numpy(image_patch).float()
        image_patch = torch.unsqueeze(image_patch, dim=0)
        target_patch = torch.from_numpy(target_patch).long()
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

def Prostate_Gen(task_names, dataset, patch_size=120):
    train_dataset_splits = {}
    val_dataset_splits = {}
    for i, task_name in enumerate(task_names):
        train_dataset_splits[task_name] = dataset(task_name, "train", patch_size)
        val_dataset_splits[task_name] = dataset(task_name, "val", patch_size)
    return train_dataset_splits, val_dataset_splits


class SequentialHeart(ContinualDataset):

    NAME = 'seq-heart'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 3

    def get_data_loaders(self, t):
        train_splits, test_splits = Prostate_Gen(["1", "2", "3"], dataset=ProstateCL_Dataset)
        train_dataset = train_splits[str(t+1)]
        test_dataset = test_splits[str(t+1)]
        train_loader = DataLoader(train_dataset,
                                  batch_size=8, shuffle=True)
        test_loader = DataLoader(test_dataset,
                                 batch_size=8, shuffle=False)
        self.test_loaders.append(test_loader)
        self.train_loader = train_loader
        return train_loader, test_loader

    def not_aug_dataloader(self, batch_size):

        return self.train_loader


    @staticmethod
    def get_backbone():
        return resunet32()

    @staticmethod
    def get_loss():
        return F.cross_entropy

