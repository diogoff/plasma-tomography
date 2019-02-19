from __future__ import print_function

import h5py
import numpy as np

# ----------------------------------------------------------------------

fname = 'train_data.hdf'
print('Reading:', fname)
f = h5py.File(fname, 'r')

pulses = np.array(f.keys())
print('pulses:', len(pulses))

# ----------------------------------------------------------------------

r = np.arange(len(pulses))

N = 10

i_train = ((r % N) <= N-2)
i_valid = ((r % N) == N-1)

train_pulses = list(pulses[i_train])
valid_pulses = list(pulses[i_valid])

print('train_pulses:', len(train_pulses))
print('valid_pulses:', len(valid_pulses))

# ----------------------------------------------------------------------

X_train = []
Y_train = []

X_valid = []
Y_valid = []

# ----------------------------------------------------------------------

for pulse in train_pulses + valid_pulses:
    g = f[pulse]
    bolo = g['bolo'][:]
    tomo = g['tomo'][:]
    for i in range(bolo.shape[0]):
        x = np.clip(bolo[i], 0., None)/1e6
        y = np.clip(tomo[i], 0., None)/1e6
        if pulse in train_pulses:
            X_train.append(x)
            Y_train.append(y)
        if pulse in valid_pulses:
            X_valid.append(x)
            Y_valid.append(y)

f.close()

# ----------------------------------------------------------------------

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_valid = np.array(X_valid)
Y_valid = np.array(Y_valid)

# ----------------------------------------------------------------------

print('X_train:', X_train.shape, X_train.dtype)
print('Y_train:', Y_train.shape, X_train.dtype)

print('X_valid:', X_valid.shape, X_valid.dtype)
print('Y_valid:', Y_valid.shape, X_valid.dtype)

# ----------------------------------------------------------------------

def save(fname, array):
    print('Writing:', fname)
    np.save(fname, array)

save('X_train.npy', X_train)
save('Y_train.npy', Y_train)

save('X_valid.npy', X_valid)
save('Y_valid.npy', Y_valid)
