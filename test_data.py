from __future__ import print_function

import sys
import h5py
import numpy as np
from ppf_data import *

# ----------------------------------------------------------------------

if len(sys.argv) < 5:
    print('Usage: %s pulse t0 t1 dt' % sys.argv[0])
    print('Example: %s 92213 46.40 54.79 0.01' % sys.argv[0])
    exit()
    
# ----------------------------------------------------------------------

pulse = int(sys.argv[1])
print('pulse:', pulse)

t0 = float(sys.argv[2])
print('t0:', t0)

t1 = float(sys.argv[3])
print('t1:', t1)

dt = float(sys.argv[4])
print('dt:', dt)

# ----------------------------------------------------------------------

bolo_t = np.arange(t0, t1+dt/2., dt)

bolo, bolo_t = get_bolo(pulse, bolo_t)

# ----------------------------------------------------------------------

fname = 'test_data.hdf'
print('Writing:', fname)
f = h5py.File(fname, 'a')

k = str(pulse)
if k in f:
    del f[k]
    
g = f.create_group(k)
g.create_dataset('bolo', data=bolo)
g.create_dataset('bolo_t', data=bolo_t)

f.close()
