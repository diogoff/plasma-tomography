
import h5py
import numpy as np
from tensorflow.keras.models import load_model

# ----------------------------------------------------------------------

fname = 'bolo_data.hdf'
print('Reading:', fname)
f = h5py.File(fname, 'r')

bolo = dict()
bolo_t = dict()

for pulse in f:
    g = f[pulse]
    bolo[pulse] = g['bolo'][:]
    bolo_t[pulse] = g['bolo_t'][:]

print('pulses:', len(bolo))

f.close()

# ----------------------------------------------------------------------

fname = 'tomo_model.hdf'
print('Reading:', fname)
model = load_model(fname)    

model.summary()

# ----------------------------------------------------------------------

tomo = dict()
tomo_t = dict()

for pulse in bolo:

    X_pred = bolo[pulse]
    print('X_pred:', X_pred.shape, X_pred.dtype)

    Y_pred = model.predict(X_pred, batch_size=500, verbose=1)
    print('Y_pred:', Y_pred.shape, Y_pred.dtype)

    tomo[pulse] = np.squeeze(Y_pred)
    tomo_t[pulse] = bolo_t[pulse]

# ----------------------------------------------------------------------

fname = 'bolo_data.hdf'
print('Writing:', fname)
f = h5py.File(fname, 'a')

for pulse in bolo:
    g = f[pulse]
    if 'tomo' in g:
        del g['tomo']
        del g['tomo_t']
    g.create_dataset('tomo', data=tomo[pulse])
    g.create_dataset('tomo_t', data=tomo_t[pulse])

f.close()
