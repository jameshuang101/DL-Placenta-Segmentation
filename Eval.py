import os
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="1"

#import libraries
import h5py
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import cv2
from PIL import Image

MAT_FILE_PATH = '/home/jhuang/Public/Advisory_Folder/Placenta_Project/Outputs/10_14_22_Mats_Processed/Axial/'
OUT_PATH = '/home/jhuang/Public/Advisory_Folder/Placenta_Project/Outputs/10_17_22_SegOverlays/'

def boundary_overlay(image1, pLabel=None, uLabel=None):
    #get boundaries of seg maps
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
    pLabel = cv2.merge([0*pLabel, pLabel//3, 0*pLabel])
    #seg2m_border = auto_canny(seg2m)
    #seg2m_b = cv2.merge([seg2m_border, 0*seg2m_border, seg2m_border])
    uLabel = cv2.merge([uLabel//3, 0*uLabel, uLabel//3])
    return cv2.add(image1_rgb, cv2.add(pLabel, uLabel))

def save_boundaries(vol, pvol, uvol, filename):
    shape = vol.shape
    pvol = pvol*255
    uvol = uvol*255
    uvol = uvol-pvol

    for i in range(shape[2]):
        bound = boundary_overlay(vol[:,:,i], pvol[:,:,i], uvol[:,:,i])
        cv2.imwrite(os.path.join(OUT_PATH, filename + '_Slice_' + str(i).zfill(2) + '.png'), bound)

def post_dice(true, pred, k = 1):
    true = np.where(true==k, 1, 0)
    pred = np.where(pred==k, 1, 0)
    intersection = np.sum(true * pred)
    if (np.sum(true)==0) and (np.sum(pred)==0):
        return 1
    dice = (2*intersection) / (np.sum(pred) + np.sum(true))
    return dice


def haus(true, pred, k=1):
    true = np.where(true==k, 1, 0)
    pred = np.where(pred==k, 1, 0)
    haus_max = 0.
    for i in range(true.shape[2]):
        haus_max = max(haus_max, directed_hausdorff(true[:,:,i], pred[:,:,i])[0])
    return haus_max


def vol_diff(true, pred, k=1):
    true = np.where(true==k, 1, 0)
    pred = np.where(pred==k, 1, 0)
    pdiff = (np.sum(pred) - np.sum(true)) / np.sum(true)
    diff = np.sum(pred) - np.sum(true)
    return diff, pdiff

f = scipy.io.loadmat(os.path.join(MAT_FILE_PATH, 'Image_0265.mat'))
mrImage = np.array(f['mrImage'],dtype='uint8')
pLabel = np.array(f['pLabel'],dtype='uint8')
uLabel = np.array(f['uLabel'],dtype='uint8')
nm = 'P0265_Seg'
save_boundaries(mrImage, pLabel, uLabel, nm)
