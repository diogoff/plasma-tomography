from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fname = 'train.log'
print('Reading:', fname)
df = pd.read_csv(fname)

epoch = df['epoch'].values
loss = df['loss'].values
val_loss = df['val_loss'].values

plt.plot(epoch, loss, label='loss')
plt.plot(epoch, val_loss, label='val_loss')

plt.xlabel('epoch')
plt.ylabel('(MW m$^{-3}$)')

plt.legend()
plt.grid()

i = np.argmin(val_loss)
min_val_loss = val_loss[i]
min_val_epoch = epoch[i]

print('min_val_loss:', min_val_loss)
print('min_val_epoch:', min_val_epoch)

(x_min, x_max) = plt.xlim()
(y_min, y_max) = plt.ylim()

plt.plot([x_min, min_val_epoch], [min_val_loss, min_val_loss], 'k--')
plt.plot([min_val_epoch, min_val_epoch], [y_min, min_val_loss], 'k--')

plt.xlim(0, x_max)
plt.ylim(y_min, y_max)

plt.tight_layout()
plt.show()
