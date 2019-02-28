from __future__ import print_function

import time
import numpy as np
np.random.seed(0)

# ----------------------------------------------------------------------

def load(fname):
    print('Reading:', fname)
    return np.load(fname)
    
X_valid = load('X_valid.npy')
Y_valid = load('Y_valid.npy')

print('X_valid:', X_valid.shape, X_valid.dtype)
print('Y_valid:', Y_valid.shape, Y_valid.dtype)

# ----------------------------------------------------------------------

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# ----------------------------------------------------------------------

import tensorflow as tf

config = tf.ConfigProto()

config.gpu_options.allow_growth = True

from keras.backend.tensorflow_backend import set_session

set_session(tf.Session(config=config))

# ----------------------------------------------------------------------

from keras.models import load_model

fname = 'model.hdf'
print('Reading:', fname)
model = load_model(fname)    

# ----------------------------------------------------------------------

Y_pred = model.predict(X_valid, batch_size=500, verbose=1)

print('Y_pred:', Y_pred.shape)

# ----------------------------------------------------------------------

val_loss = np.mean(np.abs(Y_valid-Y_pred))

print('val_loss: %.6f' % val_loss)
