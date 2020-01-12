
import numpy as np
import tensorflow as tf

# ----------------------------------------------------------------------

def load(fname):
    print('Reading:', fname)
    return np.load(fname)
    
X_valid = load('X_valid.npy')
Y_valid = load('Y_valid.npy')

print('X_valid:', X_valid.shape, X_valid.dtype)
print('Y_valid:', Y_valid.shape, Y_valid.dtype)

# ----------------------------------------------------------------------

fname = 'model.h5'
print('Reading:', fname)
model = tf.keras.models.load_model(fname)

model.summary()

# ----------------------------------------------------------------------

Y_pred = model.predict(X_valid, batch_size=1000, verbose=1)

print('Y_pred:', Y_pred.shape)

# ----------------------------------------------------------------------

loss = np.mean(np.abs(Y_valid-Y_pred))

print('loss: %.6f' % loss)
