import os
import numpy as np
import h5py
import scipy.io as io

TRAIN_IMG_MAT_PATH = 'W:/04_Segmentation/Data/DL_data_sagittal/Mat_Files/Images/Train'
TRAIN_LBL_MAT_PATH = 'W:/04_Segmentation/Data/DL_data_sagittal/Mat_Files/Labels/Train'
TEST_IMG_MAT_PATH = 'W:/04_Segmentation/Data/DL_data_sagittal/Mat_Files/Images/Test'
TEST_LBL_MAT_PATH = 'W:/04_Segmentation/Data/DL_data_sagittal/Mat_Files/Labels/Test'
VAL_IMG_MAT_PATH = 'W:/04_Segmentation/Data/DL_data_sagittal/Mat_Files/Images/Val'
VAL_LBL_MAT_PATH = 'W:/04_Segmentation/Data/DL_data_sagittal/Mat_Files/Labels/Val'

TRAIN_IMG_NPY_PATH = 'W:/04_Segmentation/Data/DL_data_sagittal/Npy_Files/Images/Train'
TRAIN_LBL_NPY_PATH = 'W:/04_Segmentation/Data/DL_data_sagittal/Npy_Files/Labels/Train'
TEST_IMG_NPY_PATH = 'W:/04_Segmentation/Data/DL_data_sagittal/Npy_Files/Images/Test'
TEST_LBL_NPY_PATH = 'W:/04_Segmentation/Data/DL_data_sagittal/Npy_Files/Labels/Test'
VAL_IMG_NPY_PATH = 'W:/04_Segmentation/Data/DL_data_sagittal/Npy_Files/Images/Val'
VAL_LBL_NPY_PATH = 'W:/04_Segmentation/Data/DL_data_sagittal/Npy_Files/Labels/Val'

# utLabel, plLabel, mrImage

def mat2npy(mat_path, npy_path, type=''):
    filenames = os.listdir(mat_path)
    filenames = [x for x in filenames if x.endswith('.mat')]
    for file in filenames:
        mat_name = os.path.join(mat_path, file)
        matr = io.loadmat(mat_name)
        if type == 'Image':
            data = matr['mrImage']
        elif type == 'Placenta':
            if 'plLabel' in matr:
                data = matr['plLabel']
            else:
                continue
        elif type == 'Uterus':
            if 'utLabel' in matr:
                data = matr['utLabel']
            else:
                continue
        else:
            continue

        npy_data = np.array(data)
        npy_name = os.path.join(npy_path, file.replace('.mat',''))
        print(npy_name)
        np.save(npy_name, npy_data)

if __name__ == '__main__':
    mat2npy(VAL_IMG_MAT_PATH, VAL_IMG_NPY_PATH, 'Image')
    mat2npy(VAL_LBL_MAT_PATH, VAL_LBL_NPY_PATH, 'Placenta')
    mat2npy(VAL_LBL_MAT_PATH, VAL_LBL_NPY_PATH, 'Uterus')
    mat2npy(TRAIN_IMG_MAT_PATH, TRAIN_IMG_NPY_PATH, 'Image')
    mat2npy(TRAIN_LBL_MAT_PATH, TRAIN_LBL_NPY_PATH, 'Placenta')
    mat2npy(TRAIN_LBL_MAT_PATH, TRAIN_LBL_NPY_PATH, 'Uterus')
    mat2npy(TEST_IMG_MAT_PATH, TEST_IMG_NPY_PATH, 'Image')
    mat2npy(TEST_LBL_MAT_PATH, TEST_LBL_NPY_PATH, 'Placenta')
    mat2npy(TEST_LBL_MAT_PATH, TEST_LBL_NPY_PATH, 'Uterus')
