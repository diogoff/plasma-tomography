from __future__ import print_function

import time
import numpy as np
np.random.seed(0)

# ----------------------------------------------------------------------

def load(fname):
    print('Reading:', fname, end='', flush=True)
    array = np.load(fname)
    print(array.shape, array.dtype)
    return array
    
X_valid = load('X_valid.npy')
Y_valid = load('Y_valid.npy')

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
