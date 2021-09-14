# In this module, we load data, train the model completely and
# save the model to the disk.

# imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
from keras.utils import np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.constraints import maxnorm
import numpy as np

# hyperparameters
BATCH_SIZE = 32
NUM_CLASSES = 10
LR = 0.01
NUM_EPOCHS = 5

def load_data(num_classes):
    '''
    load mnist data
    scale it to (0,1)
    return train, test
    '''
    # get data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # one hot vectors
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    
    # cast it float32
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    # scale pixels to (0,1)
    x_train /= 255.0
    x_test /= 255.0

    return x_train, y_train, x_test, y_test

def create_model(num_classes, activation='relu'):
    '''
    construct and compile model.
    return model.
    '''
    # Create the model
    model = Sequential()

    model.add(tf.keras.Input(shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation=activation, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation=activation, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation=activation))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    opt = tf.keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def train(model, x_train, y_train, x_test, y_test, num_epochs, batch_size):
    '''
    train model.
    returns model.
    '''
    model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0.1, shuffle=True)    
    return model

    
if __name__ == '__main__':
    # get data
    x_train, y_train, x_test, y_test = load_data(NUM_CLASSES)
    # create model
    model_with_relu = create_model(NUM_CLASSES)
    # train model
    model_with_relu = train(model_with_relu, x_train, y_train, x_test, y_test, NUM_EPOCHS, BATCH_SIZE)
    # save model
    model_with_relu.save('model_relu.h5')
    
