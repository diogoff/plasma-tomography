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

print('val_loss: %.6f' % val_loss)
