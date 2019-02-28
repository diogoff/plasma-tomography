from __future__ import print_function

import h5py
import numpy as np
from ppf_data import *

# ----------------------------------------------------------------------

fname = 'train_data.hdf'
print('Reading:', fname)
f = h5py.File(fname, 'r')

bolo = dict()
bolo_t = dict()

for pulse in f:
    g = f[pulse]
    bolo[pulse] = g['bolo'][:]
    bolo_t[pulse] = g['bolo_t'][:]

print('pulses:', len(bolo))

f.close()

# ----------------------------------------------------------------------

from keras.models import load_model

fname = 'model.hdf'
print('Reading:', fname)
model = load_model(fname)    

# ----------------------------------------------------------------------

tomo = dict()
tomo_t = dict()

for pulse in bolo:

    X_test = bolo[pulse]
    print('X_test:', X_test.shape, X_test.dtype)

    Y_pred = model.predict(X_test, batch_size=500, verbose=1)
    print('Y_pred:', Y_pred.shape, Y_pred.dtype)

    tomo[pulse] = np.squeeze(Y_pred)
    tomo_t[pulse] = bolo_t[pulse]

    print('tomo:', tomo.shape, tomo.dtype)
    print('tomo_t:', tomo_t.shape, tomo_t.dtype)

# ----------------------------------------------------------------------

fname = 'test_data.hdf'
print('Writing:', fname)
f = h5py.File(fname, 'a')

for pulse in bolo:
    g = f[pulse]
    g.create_dataset('tomo', data=tomo)
    g.create_dataset('tomo_t', data=tomo_t)

f.close()
