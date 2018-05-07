from __future__ import print_function

import h5py
import numpy as np
from data import *

# ----------------------------------------------------------------------

pulse = 92213
t0 = 46.40
t1 = 54.79
dt = 0.01

bolo_t = np.arange(t0, t1+dt/2., dt)

bolo, bolo_t = get_bolo(pulse, bolo_t)

# ----------------------------------------------------------------------

from model import *

model = create_model()

fname = 'model_weights.hdf'
print('Reading:', fname)
model.load_weights(fname)

# ----------------------------------------------------------------------

X_test = np.clip(bolo, 0., None)/1e6

Y_pred = model.predict(X_test, batch_size=500, verbose=1)

tomo = np.squeeze(Y_pred)
tomo_t = bolo_t

print('tomo:', tomo.shape, tomo.dtype)
print('tomo_t:', tomo_t.shape, tomo_t.dtype)

# ----------------------------------------------------------------------

fname = 'test_data.hdf'
print('Writing:', fname)
f = h5py.File(fname, 'w')

g = f.create_group(str(pulse))
g.create_dataset('bolo', data=bolo)
g.create_dataset('bolo_t', data=bolo_t)
g.create_dataset('tomo', data=tomo)
g.create_dataset('tomo_t', data=tomo_t)

f.close()
