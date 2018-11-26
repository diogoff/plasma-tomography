from __future__ import print_function

import h5py
import numpy as np
import pyexcel_ods
from data import *

reliable_only = True

if reliable_only:

    fname = 'tomography_completed.reliable.ods'
    print('Reading:', fname)
    ods_data = pyexcel_ods.get_data(fname)
    pulses = []
    for page in ods_data:
        pulses += [row[0] for row in ods_data[page][1:]]
    pulses = sorted(set(pulses))
    print('pulses:', pulses)

else:

    pulse0 = 80128
    pulse1 = 92504
    pulses = range(pulse0, pulse1+1)

fname = 'train_data.hdf'
print('Writing:', fname)
f = h5py.File(fname, 'w')

for pulse in pulses:
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
