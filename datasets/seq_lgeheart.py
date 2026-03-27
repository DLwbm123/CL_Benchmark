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


class LGEHeartCL_Dataset(Dataset):
    def __init__(self, task_name, mode, patch_size):
        self.task = task_name
        root_pro = "/data4/CL_Seg_data/LGEHeart/"
        if task_name == "3":
            task_h5 = os.path.join(root_pro, "Utah" + ".h5")  # BIDMC
        elif task_name == "2":
            task_h5 = os.path.join(root_pro, "KCL" + ".h5")  # UCL
        elif task_name == "1":
            task_h5 = os.path.join(root_pro, "Yale" + ".h5")  # ISBI
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
        image_patch = cv2.resize(image, (self.patch_size, self.patch_size))
        target_patch = cv2.resize(target, (self.patch_size, self.patch_size))
        # image_patch, target_patch = get_patch(image_patch, target_patch, self.patch_size)
        image_patch = torch.from_numpy(image_patch).float()
        image_patch = torch.unsqueeze(image_patch, dim=0)
        target_patch = torch.from_numpy(target_patch).long()
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


def LGEHeart_Gen(task_names, dataset, patch_size=256):
    train_dataset_splits = {}
    val_dataset_splits = {}
    for i, task_name in enumerate(task_names):
        train_dataset_splits[task_name] = dataset(task_name, "train", patch_size)
        val_dataset_splits[task_name] = dataset(task_name, "val", patch_size)
    return train_dataset_splits, val_dataset_splits


class SequentialLGEHeart(ContinualDataset):
    NAME = 'seq-lgeheart'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 3

    def get_data_loaders(self, t):
        train_splits, test_splits = LGEHeart_Gen(["1", "2", "3"], dataset=LGEHeartCL_Dataset)
        train_dataset = train_splits[str(t + 1)]
        test_dataset = test_splits[str(t + 1)]
        if t < 2:
            next_test_dataset = test_splits[str(t + 2)]
            self.next_test_loaders = DataLoader(next_test_dataset,
                                                batch_size=2, shuffle=False)
        else:
            next_test_dataset = None
            self.next_test_loaders = None
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset,
                                 batch_size=2, shuffle=False)
        self.train_dataset = train_dataset
        self.test_loaders.append(test_loader)
        self.train_loader = train_loader
        self.test_loader = test_loader

        return train_loader, test_loader

    def not_aug_dataloader(self, batch_size):

        # return self.train_loader
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

    @staticmethod
    def get_backbone(args):
        if 'gpm' in args.model:
            model = resunet32_withdict(args.inputsize)
        else:
            model = resunet32()

        if args.multihead:
            N_TASKS = 3
            model.last = nn.ModuleDict()
            for task in range(N_TASKS):
                model.last[str(task)] = nn.Conv2d(32, 2, 1, 1)

            def new_logits(self, x, k):
                return self.last[str(k)](x)

            model.logits = MethodType(new_logits, model)
        return model

    @staticmethod
    def get_loss():
        return F.cross_entropy

if __name__ == '__main__':
    dataset = LGEHeartCL_Dataset("1", "train", 256)
    loader  = DataLoader(dataset, 2)
    a, b = next(iter(loader))
    a = a.cuda()
    print(a.size(), b.size(), torch.unique(b))