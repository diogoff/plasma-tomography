
import h5py
import numpy as np
import tensorflow as tf

# ----------------------------------------------------------------------

fname = 'model.h5'
print('Reading:', fname)
model = tf.keras.models.load_model(fname)

model.summary()

# ----------------------------------------------------------------------

fname = 'bolo_data.h5'
print('Reading:', fname)
f = h5py.File(fname, 'r+')

# ----------------------------------------------------------------------

for pulse in f:

    g = f[pulse]
    bolo = np.clip(g['bolo'][:], 0., None)/1e6
    bolo_t = g['bolo_t'][:]
    print('%-10s %-10s %-20s %-10s' % (pulse, 'bolo', bolo.shape, bolo.dtype))
    print('%-10s %-10s %-20s %-10s' % (pulse, 'bolo_t', bolo_t.shape, bolo_t.dtype))    
    
    tomo = model.predict(bolo, batch_size=1000, verbose=1)*1e6
    tomo_t = bolo_t
    print('%-10s %-10s %-20s %-10s' % (pulse, 'tomo', tomo.shape, tomo.dtype))
    print('%-10s %-10s %-20s %-10s' % (pulse, 'tomo_t', tomo_t.shape, tomo_t.dtype))    

    if 'tomo' in g:
        del g['tomo']
    if 'tomo_t' in g:
        del g['tomo_t']

    g.create_dataset('tomo', data=tomo)
    g.create_dataset('tomo_t', data=tomo_t)

# ----------------------------------------------------------------------

f.close()
