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

print('X_valid:', X_valid.shape)
print('Y_valid:', Y_valid.shape)

# ----------------------------------------------------------------------

from model import *

model = create_model()

fname = 'model_weights.hdf'
print('Reading:', fname)
model.load_weights(fname)

# ----------------------------------------------------------------------

Y_pred = model.predict(X_valid, batch_size=500, verbose=1)

print('Y_pred:', Y_pred.shape)

# ----------------------------------------------------------------------

val_loss = np.mean(np.abs(Y_valid-Y_pred))

print('val_loss:', val_loss)
