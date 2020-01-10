
import h5py
import tqdm
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

i_train = r[(r % N) != 0]
i_valid = r[(r % N) == 0]

train_pulses = pulses[i_train]
valid_pulses = pulses[i_valid]

print('train_pulses:', len(train_pulses))    
print('valid_pulses:', len(valid_pulses))    
    
# ----------------------------------------------------------------------

def get_XY(pulses):
    X = []
    Y = []
    for pulse in tqdm.tqdm(pulses):
        g = f[pulse]
        tomo = np.clip(g['tomo'][:], 0., None)/1e6
        bolo = np.clip(g['bolo'][:], 0., None)/1e6
        tomo_t = g['tomo_t'][:]
        bolo_t = g['bolo_t'][:]
        for i in range(tomo.shape[0]):
            t = tomo_t[i]
            dt = 0.005
            i0 = np.argmin(np.fabs(bolo_t - t))
            i1 = np.argmin(np.fabs(bolo_t - (t + dt)))
            x = np.mean(bolo[i0:i1+1], axis=0)
            y = tomo[i]
            X.append(x)
            Y.append(y)
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

def save(fname, array):
    print('Writing:', fname)
    np.save(fname, array)

save('X_train.npy', X_train)
save('Y_train.npy', Y_train)

save('X_valid.npy', X_valid)
save('Y_valid.npy', Y_valid)

# ----------------------------------------------------------------------

f.close()
