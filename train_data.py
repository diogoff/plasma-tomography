from __future__ import print_function

import h5py
import numpy as np
from get_data import *

fname = 'train_data.hdf'
print('Writing:', fname)
f = h5py.File(fname, 'w')

pulse0 = 80128
pulse1 = 92504

for pulse in range(pulse0, pulse1+1):
    tomo, tomo_t = get_tomo(pulse)
    if len(tomo) > 0:
        bolo, bolo_t = get_bolo(pulse, tomo_t)
        g = f.create_group(str(pulse))
        g.create_dataset('bolo', data=bolo)
        g.create_dataset('bolo_t', data=bolo_t)
        g.create_dataset('tomo', data=tomo)
        g.create_dataset('tomo_t', data=tomo_t)
        print('-'*76)

f.close()
