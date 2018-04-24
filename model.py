
from keras.models import *
from keras.layers import *

def create_model():

    model = Sequential()

    model.add(Dense(25*15*20, input_shape=(56,)))
    model.add(Activation('relu'))

    model.add(Dense(25*15*20))
    model.add(Activation('relu'))

    model.add(Reshape((25,15,20)))

    model.add(Conv2DTranspose(20, kernel_size=(5,5), strides=(2,2), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(20, kernel_size=(5,5), strides=(2,2), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(20, kernel_size=(5,5), strides=(2,2), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(1, kernel_size=(1,1), strides=(1,1)))
    model.add(Activation('relu'))
    
    model.summary()
    
    return model
