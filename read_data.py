from __future__ import print_function

import h5py
import numpy as np

fname = 'tomo_data.hdf'
print('Reading:', fname)
f = h5py.File(fname, 'r')

count = 0

for pulse in f:
    g = f[pulse]
    tomo = g['tomo'][:]
    tomo_t = g['tomo_t'][:]
    kb5 = g['kb5'][:]
    for i in range(len(tomo_t)):
        count += 1
        print('%10d %10s %10.4f' % (count, pulse, tomo_t[i]))

f.close()
