from __future__ import print_function

import sys
import h5py
import numpy as np
from ppf_data import *

# ----------------------------------------------------------------------

if len(sys.argv) < 5:
    print('Usage: %s pulse t0 t1 dt' % sys.argv[0])
    print('Example: %s 92213 46.40 54.79 0.01' % sys.argv[0])
    exit()
    
# ----------------------------------------------------------------------

pulse = int(sys.argv[1])
print('pulse:', pulse)

t0 = float(sys.argv[2])
print('t0:', t0)

t1 = float(sys.argv[3])
print('t1:', t1)

dt = float(sys.argv[4])
print('dt:', dt)

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

bolo_t = np.arange(t0, t1+dt/2., dt)

bolo, bolo_t = get_bolo(pulse, bolo_t)

# ----------------------------------------------------------------------

X_test = np.clip(bolo, 0., None)/1e6

print('X_test:', X_test.shape, X_test.dtype)

# ----------------------------------------------------------------------

Y_pred = model.predict(X_test, batch_size=500, verbose=1)

print('Y_pred:', Y_pred.shape, Y_pred.dtype)

# ----------------------------------------------------------------------

tomo = np.squeeze(Y_pred)
tomo_t = bolo_t

print('tomo:', tomo.shape, tomo.dtype)
print('tomo_t:', tomo_t.shape, tomo_t.dtype)

# ----------------------------------------------------------------------

fname = 'test_data.hdf'
print('Writing:', fname)
f = h5py.File(fname, 'a')

k = str(pulse)

if k in f:
    print('Warning: deleting', k)
    del f[k]
    
g = f.create_group(k)
g.create_dataset('bolo', data=bolo)
g.create_dataset('bolo_t', data=bolo_t)
g.create_dataset('tomo', data=tomo)
g.create_dataset('tomo_t', data=tomo_t)

f.close()
