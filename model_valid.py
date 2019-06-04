from __future__ import print_function

import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import *

# ----------------------------------------------------------------------

i0 = 0
if len(sys.argv) > 1:
    i0 = int(sys.argv[1])
    print('i0:', i0)

i1 = 0
if len(sys.argv) > 2:
    i1 = int(sys.argv[2])
    print('i1:', i1)

j0 = 0
if len(sys.argv) > 3:
    j0 = int(sys.argv[3])
    print('j0:', j0)

j1 = 0
if len(sys.argv) > 4:
    j1 = int(sys.argv[4])
    print('j1:', j1)

# ----------------------------------------------------------------------

def load(fname):
    print('Reading:', fname)
    return np.load(fname)
    
X_valid = load('X_valid.npy')
Y_valid = load('Y_valid.npy')

print('X_valid:', X_valid.shape, X_valid.dtype)
print('Y_valid:', Y_valid.shape, Y_valid.dtype)

X_test = load('X_test.npy')
Y_test = load('Y_test.npy')

print('X_test:', X_test.shape, X_test.dtype)
print('Y_test:', Y_test.shape, Y_test.dtype)

# ----------------------------------------------------------------------

from keras.models import load_model

fname = 'model.hdf'
print('Reading:', fname)
model = load_model(fname)    

# ----------------------------------------------------------------------

for (X_pred, Y_true) in [(X_valid, Y_valid), (X_test, Y_test)]:

    Y_pred = model.predict(X_pred, batch_size=500, verbose=1)

    print('Y_pred:', Y_pred.shape)

    # ----------------------------------------------------------------------

    loss = np.mean(np.abs(Y_true-Y_pred))

    print('loss: %.6f' % loss)

    # ----------------------------------------------------------------------

    ssim = []
    psnr = []
    rmse = []

    for i in range(X_pred.shape[0]):
        y0 = np.clip(Y_true[i], 0., 1.)
        y1 = np.clip(Y_pred[i], 0., 1.)
        if (i0 != 0) or (i1 != 0) or (j0 != 0) or (j1 != 0):
            y0 = y0[i0:i1,j0:j1]
            y1 = y1[i0:i1,j0:j1]
        ssim.append(compare_ssim(y0, y1))
        psnr.append(compare_psnr(y0, y1))
        rmse.append(compare_nrmse(y0, y1))

    mean_ssim = np.mean(ssim)
    mean_psnr = np.mean(psnr)
    mean_rmse = np.mean(rmse)

    print('mean_ssim: %10.6f' % mean_ssim)
    print('mean_psnr: %10.6f' % mean_psnr)    
    print('mean_rmse: %10.6f' % mean_rmse)    

    std_ssim = np.std(ssim)
    std_psnr = np.std(psnr)
    std_rmse = np.std(rmse)

    print('std_ssim: %10.6f' % std_ssim)
    print('std_psnr: %10.6f' % std_psnr)    
    print('std_rmse: %10.6f' % std_rmse)    
