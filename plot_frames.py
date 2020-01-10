
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

fname = 'tomo_data.hdf'
print('Reading:', fname)
f = h5py.File(fname, 'r')

g = f[str(pulse)]
bolo = np.clip(g['bolo'][:], 0., None)/1e6
bolo_t = g['bolo_t'][:]

f.close()

# ----------------------------------------------------------------------

if t0 < bolo_t[0]:
    t0 = bolo_t[0]

if t1 > bolo_t[-1]:
    t1 = bolo_t[-1]

# ----------------------------------------------------------------------

X = []
X_t = []

for t in np.arange(t0, t1, dt):
    dt = 0.005
    i0 = np.argmin(np.fabs(bolo_t - t))
    i1 = np.argmin(np.fabs(bolo_t - (t + dt)))
    x = np.mean(bolo[i0:i1+1], axis=0)
    X.append(x)
    X_t.append(t)

X = np.array(X)
X_t = np.array(X_t)

print('X:', X.shape, X.dtype)
print('X_t:', X_t.shape, X_t.dtype)

# ----------------------------------------------------------------------

fname = 'model.h5'
print('Reading:', fname)
model = load_model(fname)    

model.summary()

# ----------------------------------------------------------------------

Y = model.predict(X, batch_size=500, verbose=1)
Y_t = X_t

print('Y:', Y.shape, Y.dtype)
print('Y_t:', Y_t.shape, Y_t.dtype)

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

while k < Y.shape[0]:
    k0 = k
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    for i in range(nrows):
        for j in range(ncols):
            if k < Y.shape[0]:
                im = ax[i,j].imshow(Y[k], cmap=get_cmap(),
                                    vmin=0., vmax=vmax,
                                    interpolation='bilinear')
                title = 't=%.*fs' % (digits, Y_t[k])
                ax[i,j].set_title(title, fontsize='small')
                ax[i,j].set_axis_off()
                k1 = k
                k += 1
            else:
                ax[i,j].set_axis_off()
    fig.set_size_inches(w, h)
    plt.subplots_adjust(left=0.001, right=1.-0.001, bottom=0.001, top=1.-0.028, wspace=0.01, hspace=0.14)
    fname = '%s/%s_%.*f_%.*f_%.*f.png' % (path, pulse, digits, Y_t[k0], digits, Y_t[k1], digits, dt)
    print('Writing:', fname, '(%d frames)' % (k-k0), '[total: %*d]' % (len(str(Y.shape[0])), k))
    plt.savefig(fname)
    plt.cla()
    plt.clf()
    plt.close()
