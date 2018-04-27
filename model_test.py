from __future__ import print_function

import h5py
import numpy as np
from ppf import *

ppfgo()
ppfssr(i=[0,1,2,3,4])

# ----------------------------------------------------------------------

def get_kb5(pulse, t0, step):
    dt = 0.0025
    ppfuid('jetppf', 'r')
    ihdata, iwdata, kb5h, x, kb5h_t, ier = ppfget(pulse, 'bolo', 'kb5h', reshape=1)
    ihdata, iwdata, kb5v, x, kb5v_t, ier = ppfget(pulse, 'bolo', 'kb5v', reshape=1)
    print(pulse, 'kb5h:', kb5h.shape, kb5h.dtype)
    print(pulse, 'kb5v:', kb5v.shape, kb5v.dtype)
    assert np.all(kb5h_t == kb5v_t)
    kb5_t = kb5h_t
    kb5 = []
    tt = []
    t = max(t0, kb5_t[0])
    while t < kb5_t[-1]-2.*dt:
        i0 = np.argmin(np.fabs(kb5_t - t))
        i1 = np.argmin(np.fabs(kb5_t - (t + 2.*dt)))
        print('%10d %10.4f %10d %10d %10d %10.4f %10.4f' % (len(kb5)+1, t, i0, i1, i1-i0+1, kb5_t[i0], kb5_t[i1]))
        hstack = np.hstack((kb5h[i0:i1+1], kb5v[i0:i1+1]))
        mean = np.mean(hstack, axis=0)
        kb5.append(mean)
        tt.append(t)
        t += step
    kb5 = np.array(kb5)
    tt = np.array(tt)
    print(pulse, 'kb5:', kb5.shape, kb5.dtype)
    return kb5, tt

# ----------------------------------------------------------------------

pulse = 92213

t0 = 40.

step = 0.01

kb5, tt = get_kb5(pulse, t0, step)

X_test = np.clip(kb5, 0., None)/1e6

print('X_test:', X_test.shape, X_test.dtype)

# ----------------------------------------------------------------------

from model import *

model = create_model()

fname = 'model_weights.hdf'
print('Reading:', fname)
model.load_weights(fname)

# ----------------------------------------------------------------------

Y_pred = model.predict(X_test, batch_size=500, verbose=1)

print('Y_pred:', Y_pred.shape, Y_pred.shape)

# ----------------------------------------------------------------------

tomo = np.squeeze(Y_pred)
tomo_t = tt

print('tomo:', tomo.shape, tomo.dtype)
print('tomo_t:', tomo_t.shape, tomo_t.dtype)

# ----------------------------------------------------------------------

fname = 'tomo_test.hdf'
print('Writing:', fname)
f = h5py.File(fname, 'w')

g = f.create_group(str(pulse))
g.create_dataset('tomo', data=tomo)
g.create_dataset('tomo_t', data=tomo_t)
g.create_dataset('step', data=[step])

f.close()
