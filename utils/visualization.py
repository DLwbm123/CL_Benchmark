import matplotlib.pyplot as plt
import os
import h5py
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
import cv2


indexs = [ 32, 125, 112, 106,  33,  38, 119,  31,  29,  65, 108,   0,  27, 123,
         28,   2, 122,  51,  17, 100,  69,  19,  18,  50,  48,  47,  83, 124,
         53,  24,  37,  90,  39,  84, 101,  64,  13,  71,  93, 103]

def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header

def save_nii(img_path, data, affine, header):
    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)
    
def generate_colors(n_color = 1):
    '''images: tensor, B, 1, h, w'''
    color_list = np.zeros((3,n_color))
    for i in range(n_color):
        x = i/n_color
        color_list[0,i] = 0.7*x*(x < 0.5) + (0.7+(x - 0.5)*0.6)*(x > 0.5)
        color_list[1,i] = (0.3 + 1.4*x)*(x < 0.5) + (1-2*(x - 0.5))*(x > 0.5)
        color_list[2,i] = 1 - 0.5*x
    return color_list


def infer(model: ContinualModel, dataset: ContinualDataset, args: Namespace, logger):
    model.net.to(model.device)
    model.net.eval()
    pretrain_path = os.path.join(args.pretrain, 'T{}_latest.pth'.format(dataset.N_TASKS))
    #load pretrain model
    model.load_network(pretrain_path, model.net)
    
    test_folder = args.input_folder
    label_folder = args.label_folder
    output_folder = "./data/inference/" + "/" + dataset.NAME + "/" + model.NAME
    create_if_not_exists(output_folder)
    
    test_files = sorted(os.listdir(test_folder))
    label_files = sorted(os.listdir(label_folder))
    assert len(test_files) == len(label_files)
    
    target_resolution = (256,256)
    
    for file_index in range(len(test_files)):
        test_file = test_files[file_index] 

        label_file = label_files[file_index]
        file_mask = os.path.join(label_folder, label_file)
        mask_dat = load_nii(file_mask)
        mask = mask_dat[0]

        img_path = os.path.join(test_folder, test_file)
        img_dat = load_nii(img_path)
        img = img_dat[0].copy()
        
        pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
        img_size = (img.shape[0], img.shape[1])

        
        img = img.astype(np.float32)
        img = np.divide((img - np.mean(img)), np.std(img))
        
        slice_rescaleds = []
        for slice_index in range(img.shape[2]):
            img_slice = cv2.resize(img[:,:,slice_index], target_resolution)
            slice_rescaleds.append(img_slice)
        img = np.stack(slice_rescaleds, axis=2)
        
        predictions = []
        
        for slice_index in range(img.shape[2]):
            img_slice = img[:,:,slice_index]
            img_slice = np.divide((img_slice - np.mean(img_slice)), np.std(img_slice))
            img_slice = np.reshape(img_slice, (1,1,nx,ny))

            img_slice = torch.from_numpy(img_slice)
            img_slice = img_slice.to(model.device)
            img_slice = img_slice.float()
            
            outputs = model(img_slice)
            
            softmax_out = outputs["pred_masks"]
            softmax_out = softmax_out.detach().cpu().numpy()
            prediction_cropped = np.squeeze(softmax_out[0,...])

            slice_predictions = np.zeros((prediction_cropped.shape[0],x,y))
            prediction = cv2.resize(slice_predictions, img_size)
            prediction = np.uint8(np.argmax(prediction, axis=0))
            #prediction = keep_largest_connected_components(prediction)
            predictions.append(prediction)
            
        prediction_arr = np.transpose(np.asarray(predictions, dtype=np.uint8), (1,2,0))
        dir_pred = os.path.join(output_folder, "predictions")
        makefolder(dir_pred)
        out_file_name = os.path.join(dir_pred, label_file)
        out_affine = mask_dat[1]
        out_header = mask_dat[2]

        save_nii(out_file_name, prediction_arr, out_affine, out_header)

        dir_gt = os.path.join(output_folder, "masks")
        makefolder(dir_gt)
        mask_file_name = os.path.join(dir_gt, label_file)
        save_nii(mask_file_name, mask_dat[0], out_affine, out_header)
        logger.info('finish inference of {}'.format(label_file))
    
        
def save_im_and_pred(model: ContinualModel, dataset: ContinualDataset, args: Namespace, logger):
    task_h5 = args.input_folder
    pred_h5 = args.label_folder
    output_folder = "./data/inference/" + "/" + args.dataset + "/" + args.model + "/" + args.name
    create_if_not_exists(output_folder)
    image_label_p = os.path.join(output_folder, "image_label")
    image_pred_p = os.path.join(output_folder, "image_pred")
    create_if_not_exists(image_label_p)
    create_if_not_exists(image_pred_p)
    
    #set color
    n_color = 1+sum(dataset.N_CLASSES_PER_TASK) if dataset.SETTING == 'class-il' else dataset.N_CLASSES_PER_TASK
    color_list = generate_colors(n_color)
    
    #read images
    data_file = h5py.File(task_h5, mode='r')
    pred_file = h5py.File(pred_h5, mode='r')
    
    images_all = data_file["val_images"]
    labels_all = data_file["val_labels"]
    preds_all = pred_file['pred_masks']
    
    for i in list(indexs):
        image = images_all[:, :, i] #H x W
        label = labels_all[:, :, i]
        pred = preds_all[i]
        image_label_path = os.path.join(image_label_p, "image_label_{}.png".format(i))
        image_pred_path = os.path.join(image_pred_p, "image_pred_{}.png".format(i))
        
        pred = cv2.resize(pred, image.shape) # C x H x W
        pred = np.argmax(pred,axis=2)
        
        #normalize images
        image = image - np.min(image)
        image = (image / np.max(image)) * 255
        image = image.astype(np.uint8)
        #to color images
        image_3 = np.expand_dims(image, axis=2)
        image_3 = np.repeat(image_3, 3, axis=2)
        image_3_pre = np.expand_dims(image, axis=2)
        image_3_pre = np.repeat(image_3_pre, 3, axis=2)
        
        for color_index in range(1,n_color):
            image_3[:, :, 0] = np.where(label == color_index, color_list[0,color_index]*255, image_3[:, :, 0])
            image_3[:, :, 1] = np.where(label == color_index, color_list[1,color_index]*255, image_3[:, :, 1])
            image_3[:, :, 2] = np.where(label == color_index, color_list[2,color_index]*255, image_3[:, :, 2])
            image_3_pre[:, :, 0] = np.where(pred == color_index, color_list[0,color_index]*255, image_3_pre[:, :, 0])
            image_3_pre[:, :, 1] = np.where(pred == color_index, color_list[1,color_index]*255, image_3_pre[:, :, 1])
            image_3_pre[:, :, 2] = np.where(pred == color_index, color_list[2,color_index]*255, image_3_pre[:, :, 2])
            
        cv2.imwrite(image_label_path, image_3)
        cv2.imwrite(image_pred_path, image_3_pre)
        logger.info("Saved for {}".format(i))
        
    
    
    
    
'''root = "/data4/CL_Seg_data/Prostate/"
task_h5 = os.path.join(root, "HK" + ".h5") #BIDMC
data_file = h5py.File(task_h5, mode='r')
ims = data_file["train_images"]
labels = data_file["train_labels"]
print("len dataset", ims.shape[2])

im = ims[:, :, 50]
label = labels[:, :, 50]

plt.subplot(1, 2, 1)
plt.imshow(im, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(label, cmap="gray")
plt.show()'''

#ISBI 421
#BIDMC 261
#I2CVB 468
#HK 158
#ISBI_1.5 384
#UCL 175    
    