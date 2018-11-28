from __future__ import print_function

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from cmap import *

# ----------------------------------------------------------------------

if len(sys.argv) < 2:
    print('Usage: %s pulse vmax' % sys.argv[0])
    print('[vmax (in MW/m3) defines the dynamic range of the plots]')
    print('Example: %s 92213 1.0' % sys.argv[0])
    exit()
    
# ----------------------------------------------------------------------

try:
    pulse = int(sys.argv[1])
    print('pulse:', pulse)
except:
    print('Unable to parse: pulse')
    exit()

try:
    vmax = float(sys.argv[2])
    print('vmax:', vmax, 'MW/m3')
except:
    print('Unable to parse: vmax')
    exit()

# ----------------------------------------------------------------------

fname = 'test_data.hdf'
print('Reading:', fname)
f = h5py.File(fname, 'r')

g = f[str(pulse)]
tomo = g['tomo'][:]
tomo_t = g['tomo_t'][:]

print(pulse, 'tomo:', tomo.shape, tomo.dtype)
print(pulse, 'tomo_t:', tomo_t.shape, tomo_t.dtype)

f.close()

# ----------------------------------------------------------------------

step = np.mean(tomo_t[1:]-tomo_t[:-1])
digits = 0
while round(step*10.**digits) == 0.:
    digits += 1

# ----------------------------------------------------------------------

path = 'frames'
if not os.path.exists(path):
    os.makedirs(path)

# ----------------------------------------------------------------------

nrows = 4
ncols = 15
k = 0
while k < tomo.shape[0]:
    k0 = k
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    for i in range(nrows):
        for j in range(ncols):
            if k < tomo.shape[0]:
                im = ax[i,j].imshow(tomo[k], cmap=get_cmap(),
                                    vmin=0., vmax=vmax,
                                    interpolation='bilinear')
                title = 't=%.*fs' % (digits, tomo_t[k])
                ax[i,j].set_title(title, fontsize='small')
                ax[i,j].set_axis_off()
                k1 = k
                k += 1
            else:
                ax[i,j].set_axis_off()
    fig.set_size_inches(18, 8)
    plt.subplots_adjust(left=0.001, right=1.-0.001, bottom=0.001, top=1.-0.025, wspace=0.02, hspace=0.12)
    fname = '%s/%s_%.*f_%.*f_%.*f.png' % (path, pulse, digits, tomo_t[k0], digits, tomo_t[k1], digits, step)
    print('Writing:', fname, '(%d frames)' % (k-k0), '(total: %d)' % k)
    plt.savefig(fname)
    plt.cla()
    plt.clf()
    plt.close()
