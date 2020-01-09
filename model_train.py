
import time
import numpy as np
np.random.seed(0)

# ----------------------------------------------------------------------

def load(fname):
    print('Reading:', fname)
    return np.load(fname)
    
X_train = load('X_train.npy')
Y_train = load('Y_train.npy')

X_valid = load('X_valid.npy')
Y_valid = load('Y_valid.npy')

print('X_train:', X_train.shape, X_train.dtype)
print('Y_train:', Y_train.shape, Y_train.dtype)

print('X_valid:', X_valid.shape, X_valid.dtype)
print('Y_valid:', Y_valid.shape, Y_valid.dtype)

# ----------------------------------------------------------------------

from keras.models import *
from keras.layers import *

model = Sequential()

model.add(Dense(25*15*20, input_shape=(56,)))
model.add(Activation('relu'))

model.add(Dense(25*15*20))
model.add(Activation('relu'))

model.add(Reshape((25,15,20)))

model.add(Conv2DTranspose(20, kernel_size=(3,3), strides=(2,2), padding='same'))
model.add(Activation('relu'))

model.add(Conv2DTranspose(20, kernel_size=(3,3), strides=(2,2), padding='same'))
model.add(Activation('relu'))

model.add(Conv2DTranspose(20, kernel_size=(3,3), strides=(2,2), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(1, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))

model.add(Lambda(lambda t: t[:,2:-2,2:-3,0]))

model.summary()

# ----------------------------------------------------------------------

from keras.optimizers import *

opt = Adam(lr=1e-4)

model.compile(optimizer=opt, loss='mae')

# ----------------------------------------------------------------------

from keras.callbacks import *

class MyCallback(Callback):
    
    def on_train_begin(self, logs=None):
        self.min_val_loss = None
        self.min_val_epoch = None
        self.min_val_weights = None
        fname = 'train.log'
        print('Writing:', fname)
        self.log = open(fname, 'w')
        print('%-10s %10s %10s %10s' % ('time', 'epoch', 'loss', 'val_loss'))
        self.log.write('time,epoch,loss,val_loss\n')
        self.log.flush()
        
    def on_epoch_end(self, epoch, logs=None):
        t = time.strftime('%H:%M:%S')
        loss = logs['loss']
        val_loss = logs['val_loss']
        if (self.min_val_loss == None) or (val_loss < self.min_val_loss):
            self.min_val_loss = val_loss
            self.min_val_epoch = epoch
            self.min_val_weights = self.model.get_weights()
            print('%-10s %10d %10.6f %10.6f *' % (t, epoch, loss, val_loss))
        else:
            print('%-10s %10d %10.6f %10.6f' % (t, epoch, loss, val_loss))
        self.log.write('%s,%d,%f,%f\n' % (t, epoch, loss, val_loss))
        self.log.flush()
        if epoch > 2*self.min_val_epoch:
            print('Stop training.')
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        self.log.close()

    def get_weights(self):
        return self.min_val_weights

# ----------------------------------------------------------------------

batch_size = 436
print('batch_size:', batch_size)

batch_ratio_train = float(X_train.shape[0]) / float(batch_size)
batch_ratio_valid = float(X_valid.shape[0]) / float(batch_size)

print('batch_ratio_train:', batch_ratio_train)
print('batch_ratio_valid:', batch_ratio_valid)

epochs = 10000

mc = MyCallback()

try:
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              callbacks=[mc],
              validation_data=(X_valid, Y_valid))

except KeyboardInterrupt:
    print('\nTraining interrupted.')

print('Loading weights.')
model.set_weights(mc.get_weights())

# ----------------------------------------------------------------------

fname = 'model.hdf'
print('Writing:', fname)
model.save(fname)
