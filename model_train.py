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

X_valid = X_train[(r+1) % 10 == 0]
Y_valid = Y_train[(r+1) % 10 == 0]

X_train = X_train[(r+1) % 10 != 0]
Y_train = Y_train[(r+1) % 10 != 0]

print('X_train:', X_train.shape)
print('Y_train:', Y_train.shape)

print('X_valid:', X_valid.shape)
print('Y_valid:', Y_valid.shape)

exit()

# ----------------------------------------------------------------------

import tensorflow as tf

config = tf.ConfigProto()

config.gpu_options.allow_growth = True

session = tf.Session(config=config)

# ----------------------------------------------------------------------

from model import *
from keras.utils import *
from keras.optimizers import *

with tf.device('/cpu:0'):
    cpu_model = create_model()

gpu_model = multi_gpu_model(cpu_model, gpus=8)

opt = Adam(lr=1e-4)

gpu_model.compile(loss='mae', optimizer=opt)

# ----------------------------------------------------------------------

from keras.callbacks import *

fname = 'train.log'
print('Writing:', fname)
f = open(fname, 'w')

class MyCallback(Callback):
    def __init__(self):
        self.min_val_loss = None
    def on_epoch_end(self, epoch, logs={}):
        loss = logs['loss']
        val_loss = logs['val_loss']
        t = time.strftime('%H:%M:%S')
        if (self.min_val_loss == None) or (val_loss < self.min_val_loss):
            if self.min_val_loss == None:
                s = '%-10s %5s %10s %10s' % ('time', 'epoch', 'loss', 'val_loss')
                print(s)
                f.write(s+'\n')
                f.flush()
            self.min_val_loss = val_loss
            s = '%-10s %5d %10.6f %10.6f *' % (t, epoch, loss, val_loss)
            print(s)
            f.write(s+'\n')
            f.flush()
            fname = 'model_weights.hdf'
            cpu_model.save_weights(fname, overwrite=True)
        else:
            s = '%-10s %5d %10.6f %10.6f' % (t, epoch, loss, val_loss)
            print(s)
            f.write(s+'\n')
            f.flush()

# ----------------------------------------------------------------------

mc = MyCallback()

try:
    gpu_model.fit(X_train, Y_train,
                  batch_size=2800,
                  epochs=100000,
                  verbose=0,
                  callbacks=[mc],
                  validation_data=(X_valid, Y_valid))
except KeyboardInterrupt:
    print('Training interrupted.')

f.close()
