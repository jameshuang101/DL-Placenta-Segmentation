## For 5 output layer networks
import os
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="1,2,3"

#import libraries
import tensorflow as tf
import tensorflow.keras as k
#from keras_unet_collection import models, base, utils, losses
import scipy
import h5py
import numpy as np
import pickle
import random
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from scipy.io import savemat
import cv2
import csv
from PIL import Image

from Networks import conv_block, UNet_3Plus_3D
from Losses import dice_coef, dice_loss, custom_loss_function
from Eval import post_dice, haus, vol_diff


DATE = 'AUG_10_22'
MODEL_NAME = 'UNet_3Plus_3D_Sag' + DATE
ROOT_PATH = '/home/jhuang/Public/Advisory_Folder/Placenta_Project'
PLOTS_PATH = os.path.join(ROOT_PATH, 'Plots')
WEIGHTS_PATH = os.path.join(ROOT_PATH, 'Weights/' + MODEL_NAME + '.h5')
HISTORY_PATH = os.path.join(ROOT_PATH, 'History/' + MODEL_NAME)

TRAIN_IMG_PATH = os.path.join(ROOT_PATH, 'Data/Sag_3d_pp/Images/Train_Block5')
TRAIN_LBL_PATH = os.path.join(ROOT_PATH, 'Data/Sag_3d_pp/Labels/Train_Block5')
VAL_IMG_PATH = os.path.join(ROOT_PATH, 'Data/Sag_3d_pp/Images/Val_Block5')
VAL_LBL_PATH = os.path.join(ROOT_PATH, 'Data/Sag_3d_pp/Labels/Val_Block5')

TEST_IMG_PATH = os.path.join(ROOT_PATH, 'Data/Sag_3d_pp/Images/Val')
TEST_LBL_PATH = os.path.join(ROOT_PATH, 'Data/Sag_3d_pp/Labels/Val')

# Prediction output paths
CSV_WRITE_PATH = os.path.join(ROOT_PATH, 'Outputs/' + MODEL_NAME + '.csv')
MAT_OUT = os.path.join(ROOT_PATH, 'Outputs/10_14_22_Mats/Sagittal')
OUTPUT_PATH = os.path.join(ROOT_PATH, 'Outputs/081122_Sag/')

# ------------------------------------- STEP 0: Configure hyperparameters ---------------------------------------------------------

LEARNING_RATE = 1e-4
NUM_EPOCHS = 500
STEPS_PER_EPOCH = 40
VAL_STEPS_PER_EPOCH = 8
BATCH_SIZE = 10
D_SIZE = (256,256,5,1)
NUM_CLASSES = 3
PATIENCE = 30
NUM_GPUS = 4

USE_PRETRAINED_WEIGHTS = False
NUM_TEST = 20

# ------------------------------------- STEP 1: get training data ---------------------------------------------------------

# Yields batches of user-defined size of data blocks 256x256x5
def data_gen_blocks(img_dir=TRAIN_IMG_PATH, lbl_dir=TRAIN_LBL_PATH, batch_size=BATCH_SIZE):
    filenames = os.listdir(img_dir)
    n = len(filenames)
    random.shuffle(filenames)
    c = 0
    step = 1
    while True:
        images_batch = np.zeros((batch_size, *D_SIZE), dtype=np.dtype('float32'))
        labels_batch = np.zeros((batch_size, *D_SIZE[:3], NUM_CLASSES), dtype=np.dtype('uint8'))

        for i in range(batch_size):
            image = np.load(os.path.join(img_dir, filenames[c]))
            label = np.load(os.path.join(lbl_dir, filenames[c].replace('Image', 'Label')))

            # Flip horizontally at a 50% rate
            toFlip = random.uniform(0,1)
            if toFlip < 0.5:
                image = np.flip(image, axis=1)
                label = np.flip(label, axis=1)

            images_batch[i] = image[..., np.newaxis]
            labels_batch[i] = label

            if c + 1 >= n:
                c = 0
                random.shuffle(filenames)
            else: c += 1

        step += 1
        if step > STEPS_PER_EPOCH:
            random.shuffle(filenames)
            c = 0
            step = 1

        yield images_batch, labels_batch

# ------------------------------------ STEP 2: Compile model and train ----------------------------------------------------------------
def train():
    train_gen = data_gen_blocks(TRAIN_IMG_PATH, TRAIN_LBL_PATH)
    val_gen = data_gen_blocks(VAL_IMG_PATH, VAL_LBL_PATH)

    # strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1", "GPU:2", "GPU:3"])
    # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    # with strategy.scope():
    model = UNet_3Plus_3D()
    model.summary(line_length=150)
    if(USE_PRETRAINED_WEIGHTS):
        model = UNet_3Plus_3D(weights = WEIGHTS_PATH)
    optimizer = k.optimizers.Adam(learning_rate=LEARNING_RATE)
    lr_metric = get_lr_metric(optimizer)
    model.compile(loss=custom_loss_function,
                  optimizer=optimizer,
                  metrics=[dice_coef, 'accuracy', lr_metric])
    print(get_model_memory_usage(BATCH_SIZE, model))

    # Set metrics for stopping and saving
    checkpoint = ModelCheckpoint(WEIGHTS_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    earlystopping = EarlyStopping(monitor='val_loss', verbose=1, patience=PATIENCE, mode='min')

    callbacks_list = [checkpoint, earlystopping]

    #Fit model
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    history = model.fit(train_gen, epochs=NUM_EPOCHS,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    callbacks=callbacks_list,
                    validation_data=val_gen,
                    validation_steps=VAL_STEPS_PER_EPOCH,
                    verbose=1)

    with open(HISTORY_PATH, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32)
    return lr


def get_model_memory_usage(batch_size, model):
    try:
        from keras import backend as K
    except:
        from tensorflow.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


# ----------------------------------- STEP 3: Save training curves ---------------------------------------------------


def plot_hist():
    history = pickle.load(open(HISTORY_PATH, "rb"))
    print(history.keys())
    plt.figure(figsize=(16,8))
    # summarize history for accuracy
    plt.plot(history['loss'][:200])
    plt.plot(history['val_loss'][:200])
    plt.title('model weighted dice loss')
    plt.ylabel('Dice loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    # summarize history for loss
    # save plots
    plt.savefig(os.path.join(PLOTS_PATH, DATE + '_Dice_Loss.png'))

    plt.clf()
    plt.figure(figsize=(16,8))
    # summarize history for accuracy
    plt.plot(history['accuracy'][:200])
    plt.plot(history['val_accuracy'][:200])
    plt.title('model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    # summarize history for loss
    # save plots
    plt.savefig(os.path.join(PLOTS_PATH, DATE + '_Acc.png'))

    plt.clf()
    plt.figure(figsize=(16,8))
    # summarize history for accuracy
    plt.plot(history['dice_coef'][:200])
    plt.plot(history['val_dice_coef'][:200])
    plt.title('model dice coef')
    plt.ylabel('DSC')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    # summarize history for loss
    # save plots
    plt.savefig(os.path.join(PLOTS_PATH, DATE + '_DiceCoef.png'))


# ----------------------------------- STEP 4: Make predictions on test data ---------------------------------------------------


def postprocess_label(lbl):
    lbl_p = np.squeeze(lbl)
    lbl_p = tf.argmax(lbl, axis=-1)
    lbl_p = (np.array(lbl_p)).astype('uint8')
    return lbl_p


def get_labels(label):
    uLabel = np.where(label==2, 1, 0).astype('uint8')
    pLabel = np.where(label==1, 1, 0).astype('uint8')
    return pLabel, uLabel


def predict():
    model = UNet_3Plus_3D(weights=WEIGHTS_PATH)
    filenames = os.listdir(TEST_IMG_PATH)
    with open(CSV_WRITE_PATH, 'w') as file:
        writer = csv.writer(file)
        header = ['name', 'pl_dsc', 'ut_dsc', 'pl_haus', 'ut_haus', 'pl_vol_diff', 'ut_vol_diff', 'pl_percent_vol_diff', 'ut_percent_vol_diff']
        writer.writerow(header)

        for i in range(len(filenames)):
            print('Testing for ', filenames[i])
            image_vol = np.load(os.path.join(TEST_IMG_PATH, filenames[i])) #256x256xN
            label_vol = np.load(os.path.join(TEST_LBL_PATH, filenames[i].replace('Image', 'Label'))) #256x256xNx3
            if image_vol.shape[0] < 256 or image_vol.shape[1] < 256:
                print('padding from', image_vol.shape)
                image_vol = np.pad(image_vol, ((int(np.floor((256-image_vol.shape[0])/2)), int(np.ceil((256-image_vol.shape[0])/2))),
                                    (int(np.floor((256-image_vol.shape[1])/2)), int(np.ceil((256-image_vol.shape[1])/2))), (0,0)))
                label_vol = np.pad(label_vol, ((int(np.floor((256-label_vol.shape[0])/2)), int(np.ceil((256-label_vol.shape[0])/2))),
                                    (int(np.floor((256-label_vol.shape[1])/2)), int(np.ceil((256-label_vol.shape[1])/2))), (0,0), (0,0)))
            label_vol_pred = np.zeros(label_vol.shape, dtype=np.dtype('float32')) #256x256xNx3
            for s in range(image_vol.shape[2] - 4):
                image = image_vol[:,:,s:s+5] #256x256x5x1
                image = image[np.newaxis,:,:,:,np.newaxis] #1x256x256x5x1
                image_f = np.flip(image, axis=2)
                label_pred = model.predict(image) # 1x256x256x5x3
                label_pred_f = model.predict(image_f)
                label_pred_f = np.flip(label_pred_f, axis=2)
                label_pred_avg = np.add(label_pred, label_pred_f) / 2
                label_vol_pred[:,:,s:s+5,:] = np.add(label_vol_pred[:,:,s:s+5,:], label_pred_avg[0])

            label_vol_pred = postprocess_label(label_vol_pred)
            label_vol = postprocess_label(label_vol)

            image_vol = (image_vol*255).astype('uint8')
            pLabel_pred, uLabel_pred = get_labels(label_vol_pred)
            pLabel_true, uLabel_true = get_labels(label_vol)

            # Save predicted labels as mat files
            mdic = {'patientnum':filenames[i].replace('.npy',''), 'mrImage':image_vol, 'pLabel_pred':pLabel_pred, 'uLabel_pred':uLabel_pred,
                    'pLabel_true':pLabel_true, 'uLabel_true':uLabel_true}
            savemat(os.path.join(MAT_OUT, filenames[i].replace('.npy', '.mat')), mdic)

            # Save csv of evaluation metrics
            phaus = haus(label_vol, label_vol_pred, k=1)
            uhaus = haus(label_vol, label_vol_pred, k=2)
            pdice = post_dice(label_vol, label_vol_pred, k=1)
            udice = post_dice(label_vol, label_vol_pred, k=2)
            pvoldif, p_pvoldif = vol_diff(label_vol, label_vol_pred, k=1)
            uvoldif, p_uvoldif = vol_diff(label_vol, label_vol_pred, k=2)
            print('Placenta haus:', phaus, 'Uterus haus:', uhaus)
            print('Placenta dice:', pdice, 'Uterus dice:', udice)
            print('Placenta vol diff:', pvoldif, 'Uterus vol diff:', uvoldif)
            data = [filenames[i], pdice, udice, phaus, uhaus, pvoldif, uvoldif, p_pvoldif, p_uvoldif]
            writer.writerow(data)

            # Save gifs of image and labels
            # label_vol_pred = label_vol_pred * 127
            # label_vol = label_vol * 127
            # image_vol = (np.array(image_vol)*255).astype('uint8')
            # save_npy_gif(image_vol, filenames[i].replace('.npy', '.gif'))
            # save_npy_gif(label_vol, filenames[i].replace('.npy', '_True.gif').replace('Image', 'Label'))
            # save_npy_gif(label_vol_pred, filenames[i].replace('.npy', '_Pred.gif').replace('Image', 'Label'))
            file.close()


def save_npy_gif(array, filename, out_dir=OUTPUT_PATH):
    array = np.squeeze(array)
    array = np.transpose(array, (2, 0, 1))
    imgs = [Image.fromarray(img) for img in array]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save(os.path.join(out_dir, filename.replace('.npy','.gif')), save_all=True, append_images=imgs[1:], duration=200, loop=0)

if __name__ == "__main__":
    train()
    plot_hist()
    predict()
