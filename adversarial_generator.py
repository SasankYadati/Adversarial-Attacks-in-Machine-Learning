# In this module we define 3 functions to generate adversarial examples
# 1-> fast gradient sign method
# 2-> basic iterative method
# 3-> iterative least likely class method

# imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
from keras.datasets import cifar10
from keras import backend as K
import numpy as np
import random

MODEL_NAME = 'model_cifar10.h5'
EPSILONS = [x/10 for x in range(0,20)] # co-efficients of perturbation
NUM_CLASSES = 10

def load_data(num_classes):
    '''
    returns x, a dictionary where keys are categories and values are list of images for 
    corresponding category.
    '''
    (x_train, y_train), (_, _) = cifar10.load_data()
    assert len(x_train) == len(y_train)
  
    x_train = x_train.astype('float32')
    x_train /= 255.0

    # each image is of shape 32x32x3
    total_images = len(y_train)
    
    x = dict() # key is category, value is list of images in that category
    
    # initialize each key's value as a list
    for j in range(0, num_classes):
        x[str(j)] = list()
    
    # for each image, append it to appropriate value in dictionary
    for i in range(0,total_images):
        category = y_train[i][0]
        x[str(category)].append(x_train[i])
    
    return x

def get_pure_examples(examples, num_examples, class_):
    '''
    num_examples defines the no. of images to be returned
    category defines the class to which the images belong
    '''
    low_limit = random.randint(0,len(examples)-num_examples-1)
    pure_examples = examples[str(class_)][low_limit:low_limit+num_examples]

    for i in range(0,len(pure_examples)):
        pure_examples[i] = np.array([pure_examples[i]])

    return pure_examples

def compute_gradient_sign(model, x, y_true, delta=1e-4):
    '''
    returns gradient of loss w.r.t the example => (x, y_true)
    '''
    
    # computing the gradients manually,
    # there could be a better way to do this

    x_delta = np.add(x,delta)

    y_pred = model.predict(x)
    y_pred_delta = model.predict(x_delta)
    
    loss = K.categorical_crossentropy(K.variable(y_true), K.variable(y_pred))
    loss_delta = K.categorical_crossentropy(K.variable(y_true), K.variable(y_pred_delta))

    if(K.eval(loss_delta) > K.eval(loss)):
        return 1 # positive gradient
    else:
        return -1 # negative gradient

def least_likely_class(model, examples, class_):
    '''
    returns least likely class for a given class by selecting a random example of class_, 
    returns an integer between [0,NUM_CLASSES).
    '''
    example = get_pure_examples(examples, 1, class_)
    example = example[0]
    preds = model.predict(example)
    ll_class = np.argmin(preds[0])
    return ll_class


def fast_gradient_sign(model, example, class_, epsilon):
    '''
    returns an adversarial example by morphing example with epsilon multiplied by gradient sign.
    '''
    gradient_sign = compute_gradient_sign(model, example, class_) # returns 1 or -1

    epsilon *= gradient_sign

    adv_example = np.add(example, epsilon)
    adv_example = np.clip(adv_example, 0, 1)

    return adv_example

def basic_iterative(model, example, class_, alpha, epsilon, num_iterations):
    '''
    returns an adversarial example similar to fast_gradient_sign() but iteratively updates to better the adversary.
    '''
    adv_example = example

    for _ in range(0, num_iterations):
        gradient_sign = compute_gradient_sign(model, adv_example, class_) # returns 1 or -1
        alpha *= gradient_sign
        adv_example = np.add(adv_example, alpha)

        # clip to keep it within epsilon neighborhood
        adv_example = np.clip(adv_example, 0, 1)
        
    return adv_example

def iterative_least_likely_class(model, examples, target_class, alpha, epsilon, num_iterations):
    '''
    returns an adversarial example.
    this is a targeted attack.
    adversairal example belongs to the least likely class for a given target category. we use least_likely_class() for this.
    '''
    # get l.l. class and l.l. example for target_class
    ll_class = least_likely_class(model, examples, target_class)
    ll_class_example = get_pure_examples(examples, 1, ll_class)

    # iteratively update ll_class_example making a better adversary
    adv_example = ll_class_example

    for _ in range(0, num_iterations):
        gradient_sign = compute_gradient_sign(model, adv_example, ll_class) # returns 1 or -1
        alpha *= gradient_sign
        adv_example = np.add(adv_example, alpha)

        # clip to keep it within epsilon neighborhood
        adv_example = np.clip(adv_example, 0, 1)
        
    return adv_example
    
if __name__ == '__main__':
    model = load_model(MODEL_NAME)
    examples = load_data(NUM_CLASSES)

    test_class = 2

    pure_examples = get_pure_examples(examples, 3, test_class)

    print("Testing on pure class {}".format(test_class))
    for i in range(0, len(pure_examples)):
        print("Prediction by the model on pure example: {}".format(np.argmax(model.predict(pure_examples[i]))))
        print("Make predictions on adversarial examples for this image")
        for epsilon in EPSILONS:
            adversarial_example = fast_gradient_sign(model, pure_examples[i], test_class, epsilon)
            print("Prediction on adversarial example using fast gradient sign with epsilon {} is {}".format(epsilon, np.argmax(model.predict(adversarial_example))))
        print("-------------------------------------------------------------------------------")