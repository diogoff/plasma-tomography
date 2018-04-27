from __future__ import print_function

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------------------------------------------------

colors = [ # (0, 0, 255),
          (51, 0, 255),
          (76, 0, 255),
          (102, 0, 255),
          (127, 0, 255),
          (153, 0, 255),
          (179, 0, 255),
          (204, 0, 255),
          (229, 0, 255),
          (255, 0, 255),
          (255, 0, 0),
          (255, 51, 0),
          (255, 76, 0),
          (255, 102, 0),
          (255, 127, 0),
          (255, 153, 0),
          (255, 179, 0),
          (255, 204, 0),
          (255, 229, 0),
          (255, 255, 0),
          (255, 255, 51),
          (255, 255, 102),
          (255, 255, 153),
          (255, 255, 204),
          (255, 255, 255)]

colors = np.array(colors, dtype=np.float64)/255.
cmap = LinearSegmentedColormap.from_list('jet', colors, N=2048)

# ----------------------------------------------------------------------

fname = 'tomo_test.hdf'
print('Reading:', fname)
f = h5py.File(fname, 'r')

for pulse in f:
    g = f[pulse]
    tomo = g['tomo'][:]
    tomo_t = g['tomo_t'][:]
    step = g['step'][0]

    print('pulse:', pulse)
    print('tomo:', tomo.shape, tomo.dtype)
    print('tomo_t:', tomo_t.shape, tomo_t.dtype)
    print('step:', step)

f.close()

# ----------------------------------------------------------------------

vmax = 1.5
digits = len(str(step).split('.')[-1])

nrows = 4
ncols = 15
k = 0
while k < len(tomo):
    k0 = k
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    for i in range(nrows):
        for j in range(ncols):
            if k < len(tomo):
                im = ax[i,j].imshow(tomo[k], cmap=cmap,
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
    fname = '%s_%.*f_%.*f.png' % (pulse, digits, tomo_t[k0], digits, tomo_t[k1])
    print('Writing:', fname, '(%d frames)' % (k-k0), '(total: %d)' % k)
    plt.savefig(fname)
    plt.cla()
    plt.clf()
    plt.close()
