
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
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

fps = 15

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

path = 'movies'
if not os.path.exists(path):
    os.makedirs(path)

# ----------------------------------------------------------------------

fontsize = 'small'

R0 = 1.708 - 2*0.02
R1 = 3.988 + 3*0.02
Z0 = -1.77 - 2*0.02
Z1 = +2.13 + 2*0.02

im = plt.imshow(Y[0], cmap=get_cmap(),
                vmin=0., vmax=vmax,
                extent=[R0, R1, Z0, Z1],
                interpolation='bilinear',
                animated=True)

ticks = np.linspace(0., vmax, num=5)
labels = ['%.2f' % t for t in ticks]
labels[-1] = r'$\geq$' + labels[-1]
cb = plt.colorbar(im, fraction=0.26, ticks=ticks)
cb.ax.set_yticklabels(labels, fontsize=fontsize)
cb.ax.set_ylabel('MW/m3', fontsize=fontsize)

fig = plt.gcf()
ax = plt.gca()

title = 'Pulse %s t=%.*fs' % (pulse, digits, Y_t[0])
ax.set_title(title, fontsize=fontsize)
ax.tick_params(labelsize=fontsize)
ax.set_xlabel('R (m)', fontsize=fontsize)
ax.set_ylabel('Z (m)', fontsize=fontsize)
ax.set_xlim([R0, R1])
ax.set_ylim([Z0, Z1])

plt.setp(ax.spines.values(), linewidth=0.1)
plt.tight_layout()

def animate(k):
    title = 'Pulse %s t=%.*fs' % (pulse, digits, Y_t[k])
    ax.set_title(title, fontsize=fontsize)
    im.set_data(Y[k])

animation = ani.FuncAnimation(fig, animate, frames=range(Y.shape[0]))

fname = '%s/%s_%.*f_%.*f.mp4' % (path, pulse, digits, Y_t[0], digits, Y_t[-1])
print('Writing:', fname)
animation.save(fname, fps=fps, extra_args=['-vcodec', 'libx264'])
