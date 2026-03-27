import h5py
import os
import numpy as np
import nibabel as nib
import random
import cv2

data_p = "E:\Medical Datasets\Medical decathon\Task02_Heart\imagesTr/"
label_p = "E:\Medical Datasets\Medical decathon\Task02_Heart\labelsTr/"
savep = "E:\Medical Datasets\other/heart.h5"

hf = h5py.File(savep, 'w')

all_cases = []
for i in os.listdir(data_p):
    all_cases.append(i)


ims = np.zeros((256, 256, 1))
labels = np.zeros((256, 256, 1))

for file in all_cases:
    if "Seg" not in file:
        filep = os.path.join(data_p, file)
        gtp = os.path.join(label_p, file)
        im = nib.load(filep).get_fdata()
        gt = nib.load(gtp).get_fdata()

        binary_mask = np.ones(gt.shape)
        mean = np.sum(im * binary_mask) / np.sum(binary_mask)
        std = np.sqrt(np.sum(np.square(im - mean) * binary_mask) / np.sum(binary_mask))
        im = (im - mean) / std  # normalize per image, using statistics within the brain, but apply to whole image
        gt[gt == 2] = 1
        for s in range(im.shape[2]):
            im_s = im[:, :, s]
            gt_s = gt[:, :, s]
            if np.max(gt_s) == 1.0:
                if s % 4 == 0:
                    im_s = cv2.resize(im_s, (256, 256))
                    gt_s = cv2.resize(gt_s, (256, 256))
                    im_s = im_s[:, :, np.newaxis]
                    gt_s = gt_s[:, :, np.newaxis]
                    ims = np.concatenate([ims, im_s], axis=2)
                    labels = np.concatenate([labels, gt_s], axis=2)
        print("all", file, "successfully!")
ims = ims[:, :, 1:]
labels = labels[:, :, 1:]
print("all images", ims.shape)

num_slice = ims.shape[2]
num_list = list(range(num_slice))
random.shuffle(num_list)
train_list = num_list[:int(0.75*len(num_list))]
val_list = num_list[int(0.75*len(num_list)):]
print("val list", val_list)

hf.create_dataset('train_images', data=ims[:, :, train_list])
hf.create_dataset('train_labels', data=labels[:, :, train_list])
hf.create_dataset('val_images', data=ims[:, :, val_list])
hf.create_dataset('val_labels', data=labels[:, :, val_list])
hf.close()