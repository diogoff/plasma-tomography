
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

path = 'movies'
if not os.path.exists(path):
    os.makedirs(path)

# ----------------------------------------------------------------------

fontsize = 'small'

R0 = 1.708 - 2*0.02
R1 = 3.988 + 3*0.02
Z0 = -1.77 - 2*0.02
Z1 = +2.13 + 2*0.02

im = plt.imshow(frames[0], cmap=get_cmap(),
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

title = 'Pulse %s t=%.*fs' % (pulse, digits, frames_t[0])
ax.set_title(title, fontsize=fontsize)
ax.tick_params(labelsize=fontsize)
ax.set_xlabel('R (m)', fontsize=fontsize)
ax.set_ylabel('Z (m)', fontsize=fontsize)
ax.set_xlim([R0, R1])
ax.set_ylim([Z0, Z1])

plt.setp(ax.spines.values(), linewidth=0.1)
plt.tight_layout()

def animate(k):
    title = 'Pulse %s t=%.*fs' % (pulse, digits, frames_t[k])
    ax.set_title(title, fontsize=fontsize)
    im.set_data(frames[k])

animation = ani.FuncAnimation(fig, animate, frames=range(frames.shape[0]))

fname = '%s/%s_%.*f_%.*f.mp4' % (path, pulse, digits, frames_t[0], digits, frames_t[-1])
print('Writing:', fname)
animation.save(fname, fps=fps, extra_args=['-vcodec', 'libx264'])
