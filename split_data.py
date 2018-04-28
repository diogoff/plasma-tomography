from __future__ import print_function

import h5py
import numpy as np

# ----------------------------------------------------------------------

fname = 'train_data.hdf'
print('Reading:', fname)
f = h5py.File(fname, 'r')

X_all = []
Y_all = []

for pulse in f:
    g = f[pulse]
    bolo = g['bolo'][:]
    tomo = g['tomo'][:]
    for i in range(bolo.shape[0]):
        x = np.clip(bolo[i], 0., None)/1e6
        y = np.clip(tomo[i], 0., None)/1e6
        X_all.append(x)
        Y_all.append(y)

f.close()

# ----------------------------------------------------------------------

X_all = np.array(X_all, dtype=np.float32)
Y_all = np.array(Y_all, dtype=np.float32)

Y_all = np.lib.pad(Y_all, ((0,0),(2,2),(2,3)), 'constant')
Y_all = np.reshape(Y_all, Y_all.shape + (1,))

print('X_all:', X_all.shape)
print('Y_all:', Y_all.shape)

# ----------------------------------------------------------------------

r = np.arange(X_all.shape[0])

X_train = X_all[(r % 10) != 0]
Y_train = Y_all[(r % 10) != 0]

X_valid = X_all[(r % 10) == 0]
Y_valid = Y_all[(r % 10) == 0]

print('X_train:', X_train.shape, X_train.dtype)
print('Y_train:', Y_train.shape, Y_train.dtype)

print('X_valid:', X_valid.shape, X_valid.dtype)
print('Y_valid:', Y_valid.shape, Y_valid.dtype)

# ----------------------------------------------------------------------

def save(array, fname):
    print('Writing:', fname)
    np.save(fname, array)

save(X_train, 'X_train.npy')
save(Y_train, 'Y_train.npy')

save(X_valid, 'X_valid.npy')
save(Y_valid, 'Y_valid.npy')
