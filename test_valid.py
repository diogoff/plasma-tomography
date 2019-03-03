from __future__ import print_function

import tqdm
import numpy as np
np.random.seed(0)
from skimage.measure import *

# ----------------------------------------------------------------------

def load(fname):
    print('Reading:', fname)
    return np.load(fname)
    
X_valid = load('X_valid.npy')
Y_valid = load('Y_valid.npy')

print('X_valid:', X_valid.shape, X_valid.dtype)
print('Y_valid:', Y_valid.shape, Y_valid.dtype)

# ----------------------------------------------------------------------

from keras.models import load_model

fname = 'model.hdf'
print('Reading:', fname)
model = load_model(fname)    

# ----------------------------------------------------------------------

Y_pred = model.predict(X_valid, batch_size=500, verbose=1)

print('Y_pred:', Y_pred.shape)

# ----------------------------------------------------------------------

val_loss = np.mean(np.abs(Y_valid-Y_pred))

print('val_loss: %.6f' % val_loss)

# ----------------------------------------------------------------------

ssim = []
psnr = []
rmse = []

for i in tqdm.tqdm(range(X_valid.shape[0])):
    y0 = np.clip(Y_valid[i], 0., 1.)
    y1 = np.clip(Y_pred[i], 0., 1.)
    ssim.append(compare_ssim(y0, y1))
    psnr.append(compare_psnr(y0, y1))
    rmse.append(compare_nrmse(y0, y1))

mean_ssim = np.mean(ssim)
mean_psnr = np.mean(psnr)
mean_rmse = np.mean(rmse)

print('mean_ssim:', mean_ssim)
print('mean_psnr:', mean_psnr)    
print('mean_rmse:', mean_rmse)    

std_ssim = np.std(ssim)
std_psnr = np.std(psnr)
std_rmse = np.std(rmse)

print('std_ssim:', std_ssim)
print('std_psnr:', std_psnr)    
print('std_rmse:', std_rmse)    
