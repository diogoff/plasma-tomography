from __future__ import print_function

import time
import numpy as np
np.random.seed(0)

# ----------------------------------------------------------------------

def load(fname):
    array = np.load(fname)
    print('Reading:', fname, array.shape, array.dtype)
    return array

X_train = load('X_train.npy')
Y_train = load('Y_train.npy')

# ----------------------------------------------------------------------

r = np.arange(X_train.shape[0])

X_valid = X_train[r % 10 == 0]
Y_valid = Y_train[r % 10 == 0]

X_train = X_train[r % 10 != 0]
Y_train = Y_train[r % 10 != 0]

print('X_train:', X_train.shape)
print('Y_train:', Y_train.shape)

print('X_valid:', X_valid.shape)
print('Y_valid:', Y_valid.shape)

# ----------------------------------------------------------------------

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.backend.tensorflow_backend import set_session
set_session(session)

# ----------------------------------------------------------------------

from model import *
from keras.utils import *
from keras.optimizers import *

with tf.device('/cpu:0'):
    model = create_model()

parallel_model = multi_gpu_model(model, gpus=8)

opt = Adam(lr=1e-4)

parallel_model.compile(loss='mae', optimizer=opt)

# ----------------------------------------------------------------------

from keras.callbacks import *

f = open('train.log', 'w')
f.close()

def log_print(s):
    print(s)
    f = open('train.log', 'a')
    f.write(s+'\n')
    f.flush()
    f.close()

class MyCallback(Callback):

    def __init__(self):
        self.min_val_loss = None
        self.min_val_epoch = None

    def on_train_begin(self, logs=None):
        log_print('%-10s %5s %10s %10s' % ('time', 'epoch', 'loss', 'val_loss'))

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        t = time.strftime('%H:%M:%S')
        loss = logs['loss']
        val_loss = logs['val_loss']
        if (self.min_val_loss == None) or (val_loss < self.min_val_loss):
            self.min_val_loss = val_loss
            self.min_val_epoch = epoch
            model.save_weights('model_weights.hdf', overwrite=True)
            log_print('%-10s %5d %10.6f %10.6f *' % (t, epoch, loss, val_loss))
        else:
            log_print('%-10s %5d %10.6f %10.6f' % (t, epoch, loss, val_loss))
        if epoch >= 2*self.min_val_epoch:
            print('Stopping training.')
            parallel_model.stop_training = True

# ----------------------------------------------------------------------

mc = MyCallback()

try:
    parallel_model.fit(X_train, Y_train,
                       batch_size=11200,
                       epochs=1000000,
                       verbose=0,
                       callbacks=[mc],
                       validation_data=(X_valid, Y_valid))
except KeyboardInterrupt:
    print('Training interrupted.')
