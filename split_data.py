from __future__ import print_function

import h5py
import numpy as np

# ----------------------------------------------------------------------

fname = 'tomo_data.hdf'
print('Reading:', fname)
f = h5py.File(fname, 'r')

X_all = []
Y_all = []

for pulse in f:
    g = f[pulse]
    bolo = g['bolo'][:]
    tomo = g['tomo'][:]
    print('pulse:', pulse, 'bolo:', '%8s' % str(bolo.shape), bolo.dtype, 'tomo:', '%14s' % str(tomo.shape), tomo.dtype)
    for i in range(bolo.shape[0]):
        X_all.append(bolo[i])
        Y_all.append(tomo[i])

X_all = np.array(X_all)
Y_all = np.array(Y_all)

print('X_all:', X_all.shape, X_all.dtype)
print('Y_all:', Y_all.shape, X_all.dtype)

f.close()

# ----------------------------------------------------------------------

r = np.arange(X_all.shape[0])

N = 10

i_train = ((r % N) <= N-3)
i_valid = ((r % N) == N-2)
i_test = ((r % N) == N-1)

# ----------------------------------------------------------------------

X_train = X_all[i_train]
Y_train = Y_all[i_train]

X_valid = X_all[i_valid]
Y_valid = Y_all[i_valid]

X_test = X_all[i_test]
Y_test = Y_all[i_test]

# ----------------------------------------------------------------------

print('X_train:', X_train.shape, X_train.dtype)
print('Y_train:', Y_train.shape, X_train.dtype)

print('X_valid:', X_valid.shape, X_valid.dtype)
print('Y_valid:', Y_valid.shape, X_valid.dtype)

print('X_test:', X_test.shape, X_test.dtype)
print('Y_test:', Y_test.shape, X_test.dtype)

# ----------------------------------------------------------------------

def save(fname, array):
    print('Writing:', fname)
    np.save(fname, array)

save('X_train.npy', X_train)
save('Y_train.npy', Y_train)

save('X_valid.npy', X_valid)
save('Y_valid.npy', Y_valid)

save('X_test.npy', X_test)
save('Y_test.npy', Y_test)
