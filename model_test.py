from __future__ import print_function

import sys
import h5py
import numpy as np
from data import *

# ----------------------------------------------------------------------

if len(sys.argv) < 5:
    print('Usage: %s pulse t0 t1 dt' % sys.argv[0])
    print('Example: %s 92213 46.40 54.79 0.01' % sys.argv[0])
    exit()
    
# ----------------------------------------------------------------------

try:
    pulse = int(sys.argv[1])
    print('pulse:', pulse)
except:
    print('Unable to parse: pulse')
    exit()

try:
    t0 = float(sys.argv[2])
    print('t0:', t0)
except:
    print('Unable to parse: t0')
    exit()

try:
    t1 = float(sys.argv[3])
    print('t1:', t1)
except:
    print('Unable to parse: t1')
    exit()

try:
    dt = float(sys.argv[4])
    print('dt:', dt)
except:
    print('Unable to parse: df')
    exit()

# ----------------------------------------------------------------------

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
f = h5py.File(fname, 'a')

if str(pulse) in f:
    del f[str(pulse)]
    
g = f.create_group(str(pulse))
g.create_dataset('bolo', data=bolo)
g.create_dataset('bolo_t', data=bolo_t)
g.create_dataset('tomo', data=tomo)
g.create_dataset('tomo_t', data=tomo_t)

f.close()
