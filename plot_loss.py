from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

fname = 'train.log'
print('Reading:', fname)
f = open(fname, 'r')

epoch = []
loss = []
val_loss = []

for line in f:
    line = line.strip()
    if len(line) == 0:
        continue
    parts = line.split()
    if parts[0] == 'time':
        continue
    epoch.append(int(parts[1]))
    loss.append(float(parts[2]))
    val_loss.append(float(parts[3]))

f.close()

linewidth = 1.5

plt.plot(epoch, loss, 'b', label='loss', linewidth=linewidth)
plt.plot(epoch, val_loss, 'r', label='val_loss', linewidth=linewidth)

plt.xlabel('epoch')
plt.ylabel('(MW m$^{-3}$)')

plt.legend()

plt.grid()

i = np.argmin(val_loss)
min_val_loss = val_loss[i]
min_epoch = epoch[i]

print('min_epoch:', min_epoch)
print('min_val_loss:', min_val_loss)

(x_min, x_max) = plt.xlim()
(y_min, y_max) = plt.ylim()

plt.plot([min_epoch, min_epoch], [y_min, min_val_loss], 'k:', linewidth=linewidth)

plt.xlim(0, x_max)
plt.ylim(y_min, y_max)

plt.tight_layout()
plt.show()
