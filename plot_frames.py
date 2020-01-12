
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from cmap import *
from tensorflow.keras.models import load_model

# ----------------------------------------------------------------------

if len(sys.argv) < 6:
    print('Usage: %s pulse t0 t1 dt vmax' % sys.argv[0])
    print('Example: %s 92213 48.0 54.0 0.01 1.0' % sys.argv[0])
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

digits = len(str(dt).split('.')[-1])

vmax = float(sys.argv[5])
print('vmax:', vmax)

# ----------------------------------------------------------------------

fname = 'bolo_data.h5'
print('Reading:', fname)
f = h5py.File(fname, 'r')

g = f[str(pulse)]
tomo = np.clip(g['tomo'][:], 0., None)/1e6
tomo_t = g['tomo_t'][:]

print('%-10s %-10s %-20s %-10s' % (pulse, 'tomo', tomo.shape, tomo.dtype))
print('%-10s %-10s %-20s %-10s' % (pulse, 'tomo_t', tomo_t.shape, tomo_t.dtype))    

f.close()

# ----------------------------------------------------------------------

if t0 < tomo_t[0]:
    t0 = tomo_t[0]
    print('t0:', t0, '(overwrite)')

if t1 > tomo_t[-1]:
    t1 = tomo_t[-1]
    print('t1:', t1, '(overwrite)')

# ----------------------------------------------------------------------

frames = []
frames_t = []

for t in np.arange(t0, t1, dt):
    i = np.argmin(np.fabs(tomo_t - t))
    frames.append(tomo[i])
    frames_t.append(tomo_t[i])

frames = np.array(frames)
frames_t = np.array(frames_t)    

print('%-10s %-10s %-20s %-10s' % (pulse, 'frames', frames.shape, frames.dtype))
print('%-10s %-10s %-20s %-10s' % (pulse, 'frames_t', frames_t.shape, frames_t.dtype))    

# ----------------------------------------------------------------------

path = 'frames'
if not os.path.exists(path):
    os.makedirs(path)

# ----------------------------------------------------------------------

w = 17
h = 8

nrows = 4
ncols = 15

k = 0

while k < frames.shape[0]:
    k0 = k
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    for i in range(nrows):
        for j in range(ncols):
            if k < frames.shape[0]:
                im = ax[i,j].imshow(frames[k], cmap=get_cmap(),
                                    vmin=0., vmax=vmax,
                                    interpolation='bilinear')
                title = 't=%.*fs' % (digits, frames_t[k])
                ax[i,j].set_title(title, fontsize='small')
                ax[i,j].set_axis_off()
                k1 = k
                k += 1
            else:
                ax[i,j].set_axis_off()
    fig.set_size_inches(w, h)
    plt.subplots_adjust(left=0.001, right=1.-0.001, bottom=0.001, top=1.-0.028, wspace=0.01, hspace=0.14)
    fname = '%s/%s_%.*f_%.*f_%.*f.png' % (path, pulse, digits, frames_t[k0], digits, frames_t[k1], digits, dt)
    print('Writing:', fname, '(%d frames)' % (k-k0), '[total: %*d]' % (len(str(frames.shape[0])), k))
    plt.savefig(fname)
    plt.cla()
    plt.clf()
    plt.close()
