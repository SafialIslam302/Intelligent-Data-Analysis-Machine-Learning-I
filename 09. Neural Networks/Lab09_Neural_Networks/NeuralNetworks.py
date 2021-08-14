#!/usr/bin/env python
# coding: utf-8

# # Neural Networks
# 
# In this exercise you will learn how to implement a feedforward neural network and train it with backpropagation.

# In[ ]:


import numpy as np
from numpy.random import multivariate_normal
from numpy.random import uniform
from scipy.stats import zscore


# We define two helper functions "init_toy_data" and "init_model" to create a simple data set to work on and a 2 layer neural network. 

# First, we create toy data with categorical labels by sampling from different multivariate normal distributions for each class. 

# In[ ]:


def init_toy_data(num_samples,num_features, num_classes, seed=3):
    # num_samples: number of samples *per class*
    # num_features: number of features (excluding bias)
    # num_classes: number of class labels
    # seed: random seed
    np.random.seed(seed)
    X=np.zeros((num_samples*num_classes, num_features))
    y=np.zeros(num_samples*num_classes)
    for c in range(num_classes):
        # initialize multivariate normal distribution for this class:
        # choose a mean for each feature
        means = uniform(low=-10, high=10, size=num_features)
        # choose a variance for each feature
        var = uniform(low=1.0, high=5, size=num_features)
        # for simplicity, all features are uncorrelated (covariance between any two features is 0)
        cov = var * np.eye(num_features)
        # draw samples from normal distribution
        X[c*num_samples:c*num_samples+num_samples,:] = multivariate_normal(means, cov, size=num_samples)
        # set label
        y[c*num_samples:c*num_samples+num_samples] = c
    return X,y


# In[ ]:


def init_model(input_size,hidden_size,num_classes, seed=3):
    # input size: number of input features
    # hidden_size: number of units in the hidden layer
    # num_classes: number of class labels, i.e., number of output units
    np.random.seed(seed)
    model = {}
    # initialize weight matrices and biases randomly
    model['W1'] = uniform(low=-1, high=1, size=(input_size, hidden_size))
    model['b1'] = uniform(low=-1, high=1, size=hidden_size)
    model['W2'] = uniform(low=-1, high=1, size=(hidden_size, num_classes))
    model['b2'] = uniform(low=-1, high=1, size=num_classes)
    return model


# In[ ]:


# create toy data
X,y= init_toy_data(2,4,3) # 2 samples per class; 4 features, 3 classes
# Normalize data
X = zscore(X, axis=0)
print('X: ' + str(X))
print('y: ' + str(y))


# We now initialise our neural net with one hidden layer consisting of $10$ units and and an output layer consisting of $3$ units. Here we expect (any number of) training samples with $4$ features. We do not apply any activation functions yet. The following figure shows a graphical representation of this neuronal net. 
# <img src="nn.graphviz.png"  width="30%" height="30%">

# In[ ]:


# initialize model
model = init_model(input_size=4, hidden_size=10, num_classes=3)

print('model: ' + str(model))
print('model[\'W1\'].shape: ' + str(model['W1'].shape))
print('model[\'W2\'].shape: ' + str(model['W2'].shape))
print('model[\'b1\'].shape: ' + str(model['b1'].shape))
print('model[\'b12\'].shape: ' + str(model['b2'].shape))
print('number of parameters: ' + str((model['W1'].shape[0] * model['W1'].shape[1]) + 
     np.sum(model['W2'].shape[0] * model['W2'].shape[1]) + 
     np.sum(model['b1'].shape[0]) +
     np.sum(model['b2'].shape[0] )))


# <b>Exercise 1</b>: Implement softmax layer.
# 
# Implement the softmax function given by 
# 
# $softmax(x_i) = \frac{e^{x_i}}{{\sum_{j\in 1...J}e^{x_j}}}$, 
# 
# where $J$ is the total number of classes, i.e. the length of  **x** .
# 
# Note: Implement the function such that it takes a matrix X of shape (N, J) as input rather than a single instance **x**; N is the number of instances.

# In[ ]:


def softmax(X):
    #######################################
    # INSERT YOUR CODE HERE
    #######################################
    return None


# Check if everything is correct.

# In[ ]:


x = np.array([[0.1, 0.7],[0.7,0.4]])
exact_softmax = np.array([[ 0.35434369,  0.64565631],
                         [ 0.57444252,  0.42555748]])
sm = softmax(x)
difference = np.sum(np.abs(exact_softmax - sm))
try:
    assert difference < 0.000001   
    print("Testing successful.")
except:
    print("Tests failed.")


# <b>Exercise 2</b>: Implement the forward propagation algorithm for the model defined above.
# 
# The activation function of the hidden neurons is a Rectified Linear Unit $relu(x)=max(0,x)$ (to be applied element-wise to the hidden units)
# The activation function of the output layer is a softmax function as (as implemented in Exercise 1).
# 
# The function should return both the activation of the hidden units (after having applied the $relu$ activation function) (shape: $(N, num\_hidden)$) and the softmax model output (shape: $(N, num\_classes)$). 

# In[ ]:


def forward_prop(X,model):
    ###############################################
    # INSERT YOUR CODE HERE                       #
    ###############################################
    return None # hidden_activations,probs


# In[ ]:


acts,probs = forward_prop(X, model)
correct_probs = np.array([[0.22836388, 0.51816433, 0.25347179],
                            [0.15853289, 0.33057078, 0.51089632],
                            [0.40710319, 0.41765056, 0.17524624],
                            [0.85151353, 0.03656425, 0.11192222],
                            [0.66016592, 0.19839791, 0.14143618],
                            [0.70362036, 0.08667923, 0.20970041]])

# the difference should be very small.
difference =  np.sum(np.abs(probs - correct_probs))

try:
    assert probs.shape==(X.shape[0],len(set(y)))
    assert difference < 0.00001   
    print("Testing successful.")
except:
    print("Tests failed.")


# <b>Exercise 3:</b> How would you train the above defined neural network? Which loss-function would you use? You do not need to implement this.

# <b>Part 2 (Neural Net using Keras)</b>
# 
# Instead of implementing the model learning ourselves, we can use the neural network library Keras for Python (https://keras.io/). Keras is an abstraction layer that either builds on top of Theano or Google's Tensorflow. So please install Keras and Tensorflow/Theano for this lab.

# <b>Exercise 4:</b>
#     Implement the same model as above using Keras:
#     
#     ** 1 hidden layer à 10 units
#     ** softmax output layer à three units
#     ** 4 input features
#     
# Compile the model using categorical cross-entropy (also referred to as 'softmax-loss') as loss function and using categorical crossentropy together categorical accuracy as metrics for runtime evaluation during training.

# In[ ]:


from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Activation

# define the model 
################################################
# INSERT YOUR CODE HERE                        #
################################################


# compile the model
################################################
# INSERT YOUR CODE HERE                        #
################################################


# The description of the current network can always be looked at via the summary method. The layers can be accessed via model.layers and weights can be obtained with the method get_weights. Check if your model is as expected. 

# In[ ]:


# Check model architecture and initial weights.

#############################################
# INSERT YOUR CODE HERE                     #
#############################################


# <b>Exercise 5:</b> Train the model on the toy data set generated below: 
# 
# Hints: 
# 
# * Keras expects one-hot-coded labels 
# 
# * Don't forget to normalize the data

# In[ ]:


X, y = init_toy_data(1000,4,3, seed=3)

##################################
# INSERT YOUR CODE HERE          #
##################################

