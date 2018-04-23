from __future__ import print_function

import h5py
import numpy as np

fname = 'tomo_data.hdf'
print('Reading:', fname)
f = h5py.File(fname, 'r')

X_train = []
Y_train = []

for pulse in f:
    g = f[pulse]
    tomo = g['tomo'][:]
    tomo_t = g['tomo_t'][:]
    kb5 = g['kb5'][:]
    for i in range(len(tomo_t)):
        x = np.clip(kb5[i], 0., None)/1e6
        y = np.clip(tomo[i], 0., None)/1e6
        X_train.append(x)
        Y_train.append(y)
        print('%10d %10s t=%.4fs' % (len(X_train), pulse, tomo_t[i]))

f.close()

X_train = np.array(X_train, dtype=np.float32)
Y_train = np.array(Y_train, dtype=np.float32)

Y_train = np.lib.pad(Y_train, ((0,0),(2,2),(2,3)), 'constant')
Y_train = np.reshape(Y_train, Y_train.shape + (1,))

print('X_train:', X_train.shape, X_train.dtype)
print('Y_train:', Y_train.shape, Y_train.dtype)

def save(fname, array):
    print('Writing:', fname, array.shape, array.dtype)
    np.save(fname, array)
 
save('X_train.npy', X_train)
save('Y_train.npy', Y_train)
 
print('Data has been clipped, rescaled and padded.')
