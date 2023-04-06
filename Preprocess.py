import os
import numpy as np
import matplotlib.pyplot as plt

os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="0"

import random
import h5py
import scipy.io
from tqdm import tqdm

ROOT_PATH = '/home/jhuang/Public/Advisory_Folder/Placenta_Project'
IMG_PATH = os.path.join(ROOT_PATH, 'Data/Sag_3D_pp/Images/Val/')
LBL_PATH = os.path.join(ROOT_PATH, 'Data/Sag_3D_pp/Labels/Val/')

# Save path
IMG_TARGET = os.path.join(ROOT_PATH, 'Data/Sag_3D_pp/Images/Val_Block5/')
LBL_TARGET = os.path.join(ROOT_PATH, 'Data/Sag_3D_pp/Labels/Val_Block5/')


def preprocess_img_3d(filenames, target):
    print(len(filenames))
    n = 1
    for filename in filenames:
        print('Preprocessing file', n, filename)
        file = scipy.io.loadmat(os.path.join(IMG_PATH, filename))
        file = np.array(file['mrImage'], dtype=np.float32)
        name = filename.replace('.mat','')
        shape = file.shape
        array = np.zeros(shape, dtype=np.float32)
        p5 = np.percentile(file, 5)
        p999 = np.percentile(file, 99.9)
        # Intensity normalization
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    if file[x,y,z] < p5:
                        array[x,y,z] = 0
                    elif file[x,y,z] > p999:
                        array[x,y,z] = 1
                    else:
                        array[x,y,z] = (file[x,y,z] - p5) / (p999 - p5)
        # Cropping out empty slices
        # Find first and last non-empty slices
        start, end = get_startend(name, shape[2])
        print('Cropping to {}, {}'.format(start, end))
        array_cropped = array[:,:,start:end]
        print('New shape: {}'.format(array_cropped.shape))
        np.save(os.path.join(target, name + '.npy'), array_cropped)
        n += 1


# Get the locations of the slices containing uterus and placenta
def get_startend(filename, nslices):
    start = 0
    end = nslices-1
    ut_file = scipy.io.loadmat(os.path.join(LBL_PATH, filename.replace('Image', 'Label') + '_Uterus.mat'))
    ut_file = np.array(ut_file['utLabel'], dtype=np.uint8)
    pl_file = scipy.io.loadmat(os.path.join(LBL_PATH, filename.replace('Image', 'Label') + '_Placenta.mat'))
    if 'mrLabel' in pl_file:
        pl_file = np.array(pl_file['mrLabel'], dtype=np.uint8)
    else:
        pl_file = np.array(pl_file['plLabel'], dtype=np.uint8)
    ut_file = np.add(ut_file, pl_file)
    while np.all((ut_file[:,:,start] == 0)):
        start += 1
    while np.all((ut_file[:,:,end - 1] == 0)):
        end -= 1
    return start, end

# Separate the ternary label into 3 binary labels of placenta, uterus, background
def preprocess_lbl_3d(filenames, target):
    print(len(filenames))
    for ut_filename in filenames:
        pl_filename = ut_filename.replace('Uterus', 'Placenta')
        ut_file = scipy.io.loadmat(os.path.join(LBL_PATH, ut_filename))
        ut_file = np.array(ut_file['utLabel'], dtype=np.uint8)
        pl_file = scipy.io.loadmat(os.path.join(LBL_PATH, pl_filename))
        if 'mrLabel' in pl_file:
            pl_file = np.array(pl_file['mrLabel'], dtype=np.uint8)
        else:
            pl_file = np.array(pl_file['plLabel'], dtype=np.uint8)

        print('Preprocessing file', str(ut_filename), ut_file.shape)
        ut_file = ut_file - pl_file
        bg_label = np.bitwise_or(pl_file, ut_file)
        bg_label = 1 - bg_label # Invert union to get background label

        name = ut_filename.replace('_Uterus.mat','')
        start, end = get_startend(name, ut_file.shape[2])
        ut_file = ut_file[:,:,start:end]
        pl_file = pl_file[:,:,start:end]
        bg_label = bg_label[:,:,start:end]
        labelstack = np.stack((bg_label, pl_file, ut_file), axis=-1)
        print('New shape: ', labelstack.shape)
        np.save(os.path.join(target, name + '.npy'), labelstack)


def get_shapes(image_dir):
    files = os.listdir(image_dir)
    min_slices = 100
    max_slices = 0
    for file in files:
        image_vol = np.load(os.path.join(image_dir, file))
        min_slices = min(min_slices, image_vol.shape[2])
        max_slices = max(max_slices, image_vol.shape[2])
    print(min_slices, max_slices)


# Break images and labels into blocks of n slices then save to target folder
def preprocess_img_lbl_block(img_path, img_target_path, lbl_path, lbl_target_path, block_size=5):
    files = os.listdir(img_path)
    for file in files:
        img = np.load(os.path.join(img_path, file))
        lbl = np.load(os.path.join(lbl_path, file.replace('Image', 'Label')))
        print('Processing ', file, 'of shape', img.shape)
        for s in tqdm(range(img.shape[2] - block_size + 1)):
            img_block = img[:, :, s:s+block_size]
            lbl_block = lbl[:, :, s:s+block_size, :]
            np.save(os.path.join(img_target_path, file.replace('.npy', '_Block_' + str(s).zfill(2) + '.npy')), img_block)
            np.save(os.path.join(lbl_target_path, file.replace('.npy', '_Block_' + str(s).zfill(2) + '.npy').replace('Image', 'Label')), lbl_block)


if __name__ == '__main__':
    preprocess_img_lbl_block(IMG_PATH, IMG_TARGET, LBL_PATH, LBL_TARGET)
