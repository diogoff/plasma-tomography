from __future__ import print_function

import h5py
import numpy as np
import matplotlib.pyplot as plt

fname = 'tomo_data.hdf'
print('Reading:', fname)
f = h5py.File(fname, 'r')

while True:
    pulse = np.random.choice(f.keys())
    g = f[pulse]
    tomo = g['tomo'][:]
    tomo_t = g['tomo_t'][:]
    kb5 = g['kb5'][:]
    i = np.random.randint(len(tomo_t))
    plt.imshow(tomo[i], vmin=0, vmax=np.max(tomo[i]))
    title = 'Pulse %s t=%.4fs' % (pulse, tomo_t[i])
    print(title)
    plt.title(title)
    plt.show()

f.close()
