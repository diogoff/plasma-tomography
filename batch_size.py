
import h5py
import numpy as np

# ----------------------------------------------------------------------

def load(fname):
    print('Reading:', fname)
    return np.load(fname)
    
X_train = load('X_train.npy')
X_valid = load('X_valid.npy')

print('X_train:', X_train.shape, X_train.dtype)
print('X_valid:', X_valid.shape, X_valid.dtype)

# ----------------------------------------------------------------------

n_gpus = 8

print('%20s %20s %20s' % ('batch_size', 'train_ratio', 'valid_ratio'))
for batch_size in range(1, X_train.shape[0]+1):
    if batch_size % n_gpus != 0:
        continue
    train_ratio = X_train.shape[0] / batch_size
    valid_ratio = X_valid.shape[0] / batch_size
    train_ratio_round = int(round(train_ratio))
    valid_ratio_round = int(round(valid_ratio))
    if train_ratio_round < train_ratio:
        continue
    if train_ratio_round - train_ratio > 1e-2:
        continue
    print('%20d %20.6f %20.6f' % (batch_size, train_ratio, valid_ratio))
