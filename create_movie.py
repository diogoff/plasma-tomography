from __future__ import print_function

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from cmap import *

# ----------------------------------------------------------------------

fname = 'test_data.hdf'
print('Reading:', fname)
f = h5py.File(fname, 'r')

pulse = f.keys()[0]

g = f[pulse]
tomo = g['tomo'][:]
tomo_t = g['tomo_t'][:]

print(pulse, 'tomo:', tomo.shape, tomo.dtype)
print(pulse, 'tomo_t:', tomo_t.shape, tomo_t.dtype)

f.close()

# ----------------------------------------------------------------------

vmax = 1.5
print('vmax:', vmax, 'MW/m3')

step = np.mean(tomo_t[1:]-tomo_t[:-1])
digits = 0
while round(step*10.**digits) == 0.:
    digits += 1

# ----------------------------------------------------------------------

fontsize = 'small'

R0 = 1.708 - 2*0.02
R1 = 3.988 + 3*0.02
Z0 = -1.77 - 2*0.02
Z1 = +2.13 + 2*0.02

im = plt.imshow(tomo[0], cmap=get_cmap(),
                vmin=0., vmax=vmax,
                extent=[R0, R1, Z0, Z1],
                interpolation='bilinear',
                animated=True)

ticks = np.linspace(0., vmax, num=7)
labels = [str(t) for t in ticks]
labels[-1] = r'$\geq$' + labels[-1]
cb = plt.colorbar(im, fraction=0.26, ticks=ticks)
cb.ax.set_yticklabels(labels, fontsize=fontsize)
cb.ax.set_ylabel(r'MW m$^{-3}$', fontsize=fontsize)

fig = plt.gcf()
ax = plt.gca()

title = 'Pulse %s t=%.*fs' % (pulse, digits, tomo_t[0])
ax.set_title(title, fontsize=fontsize)
ax.tick_params(labelsize=fontsize)
ax.set_xlabel('R (m)', fontsize=fontsize)
ax.set_ylabel('Z (m)', fontsize=fontsize)
ax.set_xlim([R0, R1])
ax.set_ylim([Z0, Z1])

plt.setp(ax.spines.values(), linewidth=0.1)
plt.tight_layout()

def animate(k):
    title = 'Pulse %s t=%.*fs' % (pulse, digits, tomo_t[k])
    ax.set_title(title, fontsize=fontsize)
    im.set_data(tomo[k])

animation = ani.FuncAnimation(fig, animate, frames=range(tomo.shape[0]))

fname = '%s_%.*f_%.*f.mp4' % (pulse, digits, tomo_t[0], digits, tomo_t[-1])
print('Writing:', fname)
animation.save(fname, fps=15, extra_args=['-vcodec', 'libx264'])
