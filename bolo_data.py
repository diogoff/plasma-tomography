from __future__ import print_function

import sys
import h5py
import numpy as np
from ppf_data import *

# ----------------------------------------------------------------------

if len(sys.argv) < 2:
    print('Usage: %s pulse pulse pulse ...' % sys.argv[0])
    exit()
    
pulses = [int(pulse) for pulse in sys.argv[1:]]
print('pulses:', pulses)

# ----------------------------------------------------------------------

fname = 'bolo_data.hdf'
print('Writing:', fname)
f = h5py.File(fname, 'a')

# ----------------------------------------------------------------------

for pulse in pulses:

    bolo, bolo_t = get_bolo(pulse)
    
    k = str(pulse)
    if k in f:
        del f[k]

    g = f.create_group(k)
    g.create_dataset('bolo', data=bolo)
    g.create_dataset('bolo_t', data=bolo_t)

# ----------------------------------------------------------------------

f.close()
