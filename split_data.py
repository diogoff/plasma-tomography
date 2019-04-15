from __future__ import print_function

import h5py
import numpy as np

# ----------------------------------------------------------------------

fname = 'tomo_data.hdf'
print('Reading:', fname)
f = h5py.File(fname, 'r')

pulses = np.array(sorted(f.keys()))
print('pulses:', len(pulses))    

# ----------------------------------------------------------------------

N = 10

r = np.arange(len(pulses))

i_train = r[(r % N) <= N-3]
i_valid = r[(r % N) == N-2]
i_test = r[(r % N) == N-1]

train_pulses = pulses[i_train]
valid_pulses = pulses[i_valid]
test_pulses = pulses[i_test]

print('train_pulses:', len(train_pulses))    
print('valid_pulses:', len(valid_pulses))    
print('test_pulses:', len(test_pulses))    
    
# ----------------------------------------------------------------------

def get_XY(pulses):
    X = []
    Y = []
    for pulse in pulses:
        g = f[pulse]
        bolo = g['bolo'][:]
        tomo = g['tomo'][:]
        for i in range(bolo.shape[0]):
            X.append(bolo[i])
            Y.append(tomo[i])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

# ----------------------------------------------------------------------

X_train, Y_train = get_XY(train_pulses)

print('X_train:', X_train.shape, X_train.dtype)
print('Y_train:', Y_train.shape, Y_train.dtype)

# ----------------------------------------------------------------------

X_valid, Y_valid = get_XY(valid_pulses)

print('X_valid:', X_valid.shape, X_valid.dtype)
print('Y_valid:', Y_valid.shape, Y_valid.dtype)

# ----------------------------------------------------------------------

X_test, Y_test = get_XY(test_pulses)

print('X_test:', X_test.shape, X_test.dtype)
print('Y_test:', Y_test.shape, Y_test.dtype)

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

# ----------------------------------------------------------------------

f.close()

# ----------------------------------------------------------------------

n = X_train.shape[0]

d = len(str(n))

for i in range(1, n+1):
    if n % i == 0:
        print('%*d batches : %*d updates/epoch' % (d, i, d, n/i))
