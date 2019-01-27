# In this module, we load data, train the model completely and
# save the model to the disk.

# imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.constraints import maxnorm

# hyperparameters
BATCH_SIZE = 32
NUM_CLASSES = 10
NUM_EPOCHS = 30

def load_data(num_classes):
    '''
    load cifar10 data
    scale it to (0,1)
    return train, test
    '''
    # get data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # one hot vectors
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    
    # cast it float32
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    # scale pixels to (0,1)
    x_train /= 255.0
    x_test /= 255.0

    return x_train, y_train, x_test, y_test

def create_model(num_classes):
    '''
    construct and compile model.
    return model.
    '''
    # Create the model
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dropout(0.2))
    
    model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    epochs = 30
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def train(model, x_train, y_train, x_test, y_test, num_epochs, batch_size):
    '''
    train model.
    returns model.
    '''
    model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_test, y_test), shuffle=True, verbose=2)
    
    return model

    
if __name__ == '__main__':
    # get data
    x_train, y_train, x_test, y_test = load_data(NUM_CLASSES)
    # create model
    model = create_model(NUM_CLASSES)
    # train model
    model = train(model, x_train, y_train, x_test, y_test, NUM_EPOCHS, BATCH_SIZE)
    # save model
    model.save('model_cifar10.h5')
    
