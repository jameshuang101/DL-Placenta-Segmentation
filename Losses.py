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

def dice_coef(y_true, y_pred, const=K.epsilon()):
    # flatten 2-d tensors
    y_true_pos = tf.reshape(y_true, [-1])
    y_pred_pos = tf.reshape(y_pred, [-1])

    # get true pos (TP), false neg (FN), false pos (FP).
    true_pos  = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1-y_pred_pos))
    false_pos = tf.reduce_sum((1-y_true_pos) * y_pred_pos)

    # 2TP/(2TP+FP+FN) == 2TP/()
    coef_val = (2.0 * true_pos + const)/(2.0 * true_pos + false_pos + false_neg)

    return coef_val

def dice_loss(y_true, y_pred, const=K.epsilon()):
    # tf tensor casting
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # <--- squeeze-out length-1 dimensions.
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)

    loss_val = 1 - dice_coef(y_true, y_pred, const=const)

    return loss_val

def custom_loss_function(y_true, y_pred):

	# print('LOGITS: ', np.shape(y_true))
	# print('LABELS: ', np.shape(y_pred))

	y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1, NUM_CLASSES])
	y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1, NUM_CLASSES])

	pLabels = y_true[:, 1:2]
	uLabels = y_true[:, 2:3]
	pPreds = y_pred[:, 1:2]
	uPreds = y_pred[:, 2:3]

	# print('LOGITS - P (new): ', np.shape(pPreds))
	# print('LABELS - P (new): ', np.shape(pLabels))
	# print('LOGITS - U (new): ', np.shape(uPreds))
	# print('LABELS - U (new): ', np.shape(uLabels))

	uIntersection = K.sum(uLabels*uPreds)
	uSegmentation = K.sum(uPreds)
	uReference = K.sum(uLabels)

	pIntersection = K.sum(pLabels*pPreds)
	pSegmentation = K.sum(pPreds)
	pReference = K.sum(pLabels)

	uDiceH = (uIntersection + 1e-9) / (uSegmentation + uReference + 1e-9)
	pDiceH = (pIntersection + 1e-9) / (pSegmentation + pReference + 1e-9) # Weight placenta slightly higher

	loss = 1 - K.mean(uDiceH) - K.mean(pDiceH)

	return loss
