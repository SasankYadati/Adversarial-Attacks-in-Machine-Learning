# In this module we define 3 functions to generate adversarial examples
# 1-> fast gradient sign method
# 2-> basic iterative method
# 3-> iterative least likely class method

# imports
from keras.models import load_model
from keras.datasets import cifar10

MODEL_NAME = 'mobile_net_cifar10.h5'
EPSILON = 0.3 # co-efficient of perturbation

def load_data():
    (x_train, y_train), (_, _) = cifar10.load_data()
    assert len(x_train) == len(y_train)
  
    # each image is of shape 32x32x3
  
    total_images = len(y_train)
    num_classes = 10
    
    x = dict() # key is category, value is list of images in that category
    
    # initialize each key's value as a list
    for j in range(0, num_classes):
        x[str(j)] = list()
    
    # for each image, append it to appropriate value in dictionary
    for i in range(0,total_images):
        category = y_train[i][0]
        x[str(category)].append(x_train[i])
    
    return x

def get_pure_examples(examples, num_examples, category):
    '''
    num_examples defines the no. of images to be returned
    category defines the class to which the images belong;
   
    '''
    return examples[str(category)][0:num_examples]
    

def fast_gradient_sign(model, example, category, epsilon):
    pass

def basic_iterative(model, example, category, epsilon, num_iterations):
    pass

def iterative_least_likely_class(model, target_category, epsilon, num_iterations):
    pass

if __name__ == '__main__':
    model = load_model(MODEL_NAME)
    examples = load_data()
    pure_examples_0 = get_pure_examples(examples, 4, 9)
    pure_examples_1 = get_pure_examples(examples, 10, 2)
  