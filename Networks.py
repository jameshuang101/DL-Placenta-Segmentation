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


def conv_block(x, kernels, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', is_bn=True, is_relu=True, n=2):
    for i in range(n):
        x = k.layers.Conv3D(filters=kernels, kernel_size=kernel_size, padding=padding, strides=strides,
                            kernel_regularizer=k.regularizers.l2(1e-4),
                            kernel_initializer=k.initializers.he_normal(seed=5))(x)
        if is_bn:
            x = k.layers.BatchNormalization()(x)
        if is_relu:
            x = k.activations.relu(x)

    return x



def UNet_3Plus_3D(INPUT_SHAPE=(256, 256, 5), OUTPUT_CHANNELS=3, weights=None):
    filters = [32, 64, 128, 256, 512]
    input_layer = k.layers.Input(shape=INPUT_SHAPE, name="input_layer") # 256x256x5

    """ Encoder """
    # block 1
    e1 = conv_block(input_layer, filters[0], kernel_size=(5, 5, 3)) # 256x256x5x64

    # block 2
    e2 = k.layers.MaxPool3D(pool_size=(2, 2, 1))(e1) # 128x128x5x64
    e2 = conv_block(e2, filters[1], kernel_size=(5, 5, 3)) # 128x128x5x128

    # block 3
    e3 = k.layers.MaxPool3D(pool_size=(2, 2, 1))(e2) # 64x64x5x128
    e3 = conv_block(e3, filters[2]) # 64x64x5x256

    # block 4
    e4 = k.layers.MaxPool3D(pool_size=(2, 2, 1))(e3) # 32x32x5x256
    e4 = conv_block(e4, filters[3]) # 32x32x5x512

    # block 5
    e5 = k.layers.MaxPool3D(pool_size=(2, 2, 1))(e4) # 16x16x5x512
    e5 = conv_block(e5, filters[4]) # 16x16x5x1024

    """ Decoder """
    cat_channels = filters[0]
    cat_blocks = len(filters)
    upsample_channels = cat_blocks * cat_channels

    """ d4 """
    e1_d4 = k.layers.MaxPool3D(pool_size=(8, 8, 1))(e1)
    e1_d4 = conv_block(e1_d4, cat_channels, n=1)                        # 256x256x5x64 --> 32x32x5x64

    e2_d4 = k.layers.MaxPool3D(pool_size=(4, 4, 1))(e2)
    e2_d4 = conv_block(e2_d4, cat_channels, n=1)                        # 128x128x5x128 --> 32x32x5x64

    e3_d4 = k.layers.MaxPool3D(pool_size=(2, 2, 1))(e3)
    e3_d4 = conv_block(e3_d4, cat_channels, n=1)                        # 64x64x5x256 --> 32x32x5x64

    e4_d4 = conv_block(e4, cat_channels, n=1)                           # 32x32x5x512 --> 32x32x5x64

    e5_d4 = k.layers.UpSampling3D(size=(2, 2, 1))(e5)  # 64x64x5x256  --> 32x32x5x64
    e5_d4 = conv_block(e5_d4, cat_channels, n=1)                        # 16x16x5x1024  --> 16x16x5x64

    d4 = k.layers.concatenate([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4])
    d4 = conv_block(d4, upsample_channels, n=1)                         # 32x32x5x160

    """ d3 """
    e1_d3 = k.layers.MaxPool3D(pool_size=(4, 4, 1))(e1)
    e1_d3 = conv_block(e1_d3, cat_channels, n=1)

    e2_d3 = k.layers.MaxPool3D(pool_size=(2, 2, 1))(e2)
    e2_d3 = conv_block(e2_d3, cat_channels, n=1)

    e3_d3 = conv_block(e3, cat_channels, n=1)

    d4_d3 = k.layers.UpSampling3D(size=(2, 2, 1))(d4)
    d4_d3 = conv_block(d4_d3, cat_channels, n=1)

    e5_d3 = k.layers.UpSampling3D(size=(4, 4, 1))(e5)
    e5_d3 = conv_block(e5_d3, cat_channels, n=1)

    d3 = k.layers.concatenate([e1_d3, e2_d3, e3_d3, d4_d3, e5_d3])
    d3 = conv_block(d3, upsample_channels, n=1)

    """ d2 """
    e1_d2 = k.layers.MaxPool3D(pool_size=(2, 2, 1))(e1)
    e1_d2 = conv_block(e1_d2, cat_channels, n=1)

    e2_d2 = conv_block(e2, cat_channels, n=1)

    d3_d2 = k.layers.UpSampling3D(size=(2, 2, 1))(d3)
    d3_d2 = conv_block(d3_d2, cat_channels, n=1)

    d4_d2 = k.layers.UpSampling3D(size=(4, 4, 1))(d4)
    d4_d2 = conv_block(d4_d2, cat_channels, n=1)

    e5_d2 = k.layers.UpSampling3D(size=(8, 8, 1))(e5)
    e5_d2 = conv_block(e5_d2, cat_channels, n=1)

    d2 = k.layers.concatenate([e1_d2, e2_d2, d3_d2, d4_d2, e5_d2])
    d2 = conv_block(d2, upsample_channels, n=1)

    """ d1 """
    e1_d1 = conv_block(e1, cat_channels, n=1)

    d2_d1 = k.layers.UpSampling3D(size=(2, 2, 1))(d2)
    d2_d1 = conv_block(d2_d1, cat_channels, n=1)

    d3_d1 = k.layers.UpSampling3D(size=(4, 4, 1))(d3)
    d3_d1 = conv_block(d3_d1, cat_channels, n=1)

    d4_d1 = k.layers.UpSampling3D(size=(8, 8, 1))(d4)
    d4_d1 = conv_block(d4_d1, cat_channels, n=1)

    e5_d1 = k.layers.UpSampling3D(size=(16, 16, 1))(e5)
    e5_d1 = conv_block(e5_d1, cat_channels, n=1)

    d1 = k.layers.concatenate([e1_d1, d2_d1, d3_d1, d4_d1, e5_d1, ])
    d1 = conv_block(d1, upsample_channels, n=1)

    d = conv_block(d1, OUTPUT_CHANNELS, n=1, is_bn=False, is_relu=False)

    output = k.activations.softmax(d)

    model = k.Model(inputs=input_layer, outputs=output, name='UNet_3Plus')

    if weights:
        model.load_weights(weights)

    return model
