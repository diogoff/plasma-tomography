from __future__ import print_function

import time
import numpy as np
np.random.seed(0)

# ----------------------------------------------------------------------

def load(fname):
    print('Reading:', fname)
    return np.load(fname)
    
X_train = load('X_train.npy')
Y_train = load('Y_train.npy')

X_valid = load('X_valid.npy')
Y_valid = load('Y_valid.npy')

print('X_train:', X_train.shape, X_train.dtype)
print('Y_train:', Y_train.shape, Y_train.dtype)

print('X_valid:', X_valid.shape, X_valid.dtype)
print('Y_valid:', Y_valid.shape, Y_valid.dtype)

# ----------------------------------------------------------------------

from model import *
from keras.optimizers import *

model = create_model()

opt = Adam(lr=1e-4)

model.compile(optimizer=opt, loss='mae')

# ----------------------------------------------------------------------

from keras.callbacks import *

class MyCallback(Callback):

    def __init__(self):
        self.min_val_loss = None
        self.min_val_epoch = None
        self.f = open('train.log', 'w')
        self.f.close()

    def log_print(self, s):
        print(s)
        self.f = open('train.log', 'a')
        self.f.write(s+'\n')
        self.f.flush()
        self.f.close()

    def on_train_begin(self, logs=None):
        self.log_print('%-10s %5s %10s %10s' % ('time', 'epoch', 'loss', 'val_loss'))

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        t = time.strftime('%H:%M:%S')
        loss = logs['loss']
        val_loss = logs['val_loss']
        if (self.min_val_loss == None) or (val_loss < self.min_val_loss):
            self.min_val_loss = val_loss
            self.min_val_epoch = epoch
            self.model.save_weights('model_weights.hdf', overwrite=True)
            self.log_print('%-10s %5d %10.6f %10.6f *' % (t, epoch, loss, val_loss))
        else:
            self.log_print('%-10s %5d %10.6f %10.6f' % (t, epoch, loss, val_loss))
        if epoch >= 2*self.min_val_epoch:
            print('Stop training.')
            self.model.stop_training = True

# ----------------------------------------------------------------------

mc = MyCallback()

batch_ratio = float(X_train.shape[0]) / 9.
batch_size = int(np.ceil(batch_ratio))
max_epochs = 10000

print('batch_size:', batch_size, '(%.2f)' % batch_ratio)

try:
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=max_epochs,
              verbose=0,
              callbacks=[mc],
              validation_data=(X_valid, Y_valid))
except KeyboardInterrupt:
    print('Training interrupted.')
