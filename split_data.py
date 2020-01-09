
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

i_train = r[(r % N) != N-1]
i_valid = r[(r % N) == N-1]

train_pulses = pulses[i_train]
valid_pulses = pulses[i_valid]

print('train_pulses:', len(train_pulses))    
print('valid_pulses:', len(valid_pulses))    
    
# ----------------------------------------------------------------------

def get_XY(pulses):
    X = []
    Y = []
    for pulse in pulses:
        g = f[pulse]
        bolo = g['bolo'][:]
        tomo = g['tomo'][:]
        X.append(bolo)
        Y.append(tomo)
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
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

def save(fname, array):
    print('Writing:', fname)
    np.save(fname, array)

save('X_train.npy', X_train)
save('Y_train.npy', Y_train)

save('X_valid.npy', X_valid)
save('Y_valid.npy', Y_valid)

# ----------------------------------------------------------------------

f.close()
