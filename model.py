# In this module, we load data, train the model completely and
# save the model to the disk.

# imports
from keras.datasets import cifar10
from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

# hyperparameters
BATCH_SIZE = 32
NUM_CLASSES = 10
NUM_EPOCHS = 20

def create_data_generator():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    y_train = np_utils.to_categorical(y_train, NUM_CLASSES)
    y_test = np_utils.to_categorical(y_test, NUM_CLASSES)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    x_train /= 255
    x_test /= 255

    return x_train, y_train, x_test, y_test

def create_model():
    img_input = Input(shape=(32, 32, 3))
    
    model = MobileNet(input_tensor=img_input, classes=NUM_CLASSES, weights=None)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(x_test, y_test), shuffle=True, verbose=1)
    
    return model

def save_model(model):
    model.save('mobile_net_cifar10.h5')
    
if __name__ == '__main__':

    x_train, y_train, x_test, y_test = create_data_generator()
    model = create_model()
    model = train(model, x_train, y_train, x_test, y_test)
    save_model(model)
    
