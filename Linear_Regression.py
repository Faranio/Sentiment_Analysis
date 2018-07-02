#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import json
import re
import string
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


def read(file_name):
	path1 = file_name
	data_file1 = open(path1, 'r')
	data1 = data_file1.read()
	json_data = json.loads(data1)
	return json_data

def process_str(this_str):
	this_str = this_str.lower()
	cleanr = re.compile('<.*?>')
	this_str = re.sub(cleanr, '', this_str).replace(' в ', ' ').replace(' а ', ' ').replace(' и ', ' ').replace('\n', ' ').replace('«', ' ').replace('»', ' ').replace('\"', ' ').replace('\t', ' ').replace('–', ' ').replace('–-', ' ').replace('$', 'txttndollar').replace('тенге', 'txttndollar').replace('рубль', 'txttndollar').replace('рублей', 'txttndollar').replace('доллар', 'txttndollar').replace('долларов', 'txttndollar').replace('%', 'txttnperc').replace('процентов', 'txttnperc').replace('процент', 'txttnperc')
	this_str = re.sub(r'http\S+', '', this_str)
	translator = str.maketrans('', '', string.punctuation)
	this_str = this_str.translate(translator)
	this_str = re.sub('(\d+[-/]\d+[-/]\d+)', 'txttndate', this_str)
	this_str = re.sub('\d\d\d\d г', 'txttndate', this_str)
	this_str = re.sub('(\d+[./]\d+[./]\d+)', 'txttndate', this_str)
	this_str = this_str.replace('января', 'txttndate').replace('февраля', 'txttndate').replace('марта', 'txttndate').replace('апреля', 'txttndate').replace('мая', 'txttndate').replace('июня', 'txttndate').replace('июля', 'txttndate').replace('августа', 'txttndate').replace('сентября', 'txttndate').replace('октября', 'txttndate').replace('ноября', 'txttndate').replace('декабря', 'txttndate').replace('январь', 'txttndate').replace('февраль', 'txttndate').replace('март', 'txttndate').replace('апрель', 'txttndate').replace('май', 'txttndate').replace('июнь', 'txttndate').replace('июль', 'txttndate').replace('август', 'txttndate').replace('сентябрь', 'txttndate').replace('октябрь', 'txttndate').replace('ноябрь', 'txttndate').replace('декабрь', 'txttndate')
	ps = PorterStemmer()
	rs = SnowballStemmer("russian")
	l = this_str.split()
	l = [rs.stem(word) for word in l]
	l = [ps.stem(word) for word in l]
	this_str = ' '.join(l)
	while 'txttndate txttndate' in this_str or 'txttndatetxttndate' in this_str:
		this_str = this_str.replace('txttndatetxttndate', 'txttndate').replace('txttndate txttndate', 'txttndate')
	this_str = re.sub('\d', 'txttnnumb', this_str)
	while 'txttnnumbtxttnnumb' in this_str or 'txttnnumb txttnnumb' in this_str:
		this_str = this_str.replace('txttnnumbtxttnnumb', 'txttnnumb').replace('txttnnumb txttnnumb', 'txttnnumb')
	this_str = this_str.replace('txttnnumb txttnperc', 'txttnstat').replace('txttnnumbtxttnperc', 'txttnstat').replace('txttnperc txttnnumb', 'txttnstat').replace('txttnperctxttnnumb', 'txttnstat')
	while '  ' in this_str:
		this_str = this_str.replace('  ', ' ')
	return this_str

def save(this_data, this_file):
	with open(this_file, 'w') as jsonfile:
		json.dump(this_data, jsonfile, ensure_ascii=False)

def predict_hx(X_set, params):
	res = np.dot(X_set, np.transpose(params))
	return res

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        ### END CODE HERE ###
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters 

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    Z = np.dot(W, A) + b
    ### END CODE HERE ###
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        ### START CODE HERE ### (≈ 2 lines of code)
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        ### END CODE HERE ###
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        ### START CODE HERE ### (≈ 2 lines of code)
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        ### END CODE HERE ###
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        ### START CODE HERE ### (≈ 2 lines of code)
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation = "relu")
        caches.append(cache)
        ### END CODE HERE ###
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    ### START CODE HERE ### (≈ 2 lines of code)
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation = "sigmoid")
    caches.append(cache)
    ### END CODE HERE ###
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches

def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    ### START CODE HERE ### (≈ 1 lines of code)
    cost = (-1) / m * sum(sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL))))
    ### END CODE HERE ###
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    ### START CODE HERE ### (≈ 3 lines of code)
    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    ### END CODE HERE ###
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        ### END CODE HERE ###
        
    elif activation == "sigmoid":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        ### END CODE HERE ###
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    ### START CODE HERE ### (1 line of code)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    ### END CODE HERE ###
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    ### START CODE HERE ### (approx. 2 lines)
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    ### END CODE HERE ###
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        ### START CODE HERE ### (approx. 5 lines)
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        ### END CODE HERE ###

    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    ### START CODE HERE ### (≈ 3 lines of code)
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    ### END CODE HERE ###
    return parameters

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    ### START CODE HERE ###
    parameters = initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward(X, parameters)
        ### END CODE HERE ###
        
        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y)
        ### END CODE HERE ###
    
        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches)
        ### END CODE HERE ###
 
        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

def predict(X, y):

	output = 0
	for i in len(X):
		if X[i] == y[i]:
			output += 1
	output /= len(X)
	return output


def text_to_arr(this_text, this_features):
	this_text = this_text.split()
	output = []
	for features in this_features:
		if features in this_text:
			output.append(1)
		else:
			output.append(0)
	return output


# features matrix
read_features = read("top_words.json")
input_features = read_features.keys()
# end of features

# read data
training_data = read("proc_train.json")
texts = [d['text'] for d in training_data]
y = [d['sentiment'] for d in training_data]
index = [d['id'] for d in training_data]
# end of read data

# process text into input
print("process to input...")
X = []
for text_this in texts:
	X = np.append(X, text_to_arr(text_this, input_features))
# end of text into input

print("Training the model...")

train_X = X.reshape(X.shape[0], -1).T
parameters = L_layer_model(train_X, y, [train_X.size, 5, 5, 3], num_iterations = 100, print_cost = True)
AL, caches = L_model_forward(X, parameters)
pred_train = (AL > 0.5)

print("The training accuracy: ")

print(pred_train)

cv_data = read("proc_train.json")
cv_texts = [d['text'] for d in cv_data]
cv_y = [d['sentiment'] for d in cv_data]
cv_index = [d['id'] for d in cv_data]

print("process to input... (cv)")
cv_X = np.array([])
for text_this in texts:
	cv_X = np.append(cv_X, text_to_arr(text_this, input_features))

print("Training the CV model...")

test_X = cv_X.reshape(cv_X.shape[0], -1).T
cv_parameters = L_layer_model(test_X, cv_y, [test_X.size, 5, 5, 3], num_iterations = 100, print_cost = True)
cv_AL, cv_caches = L_model_forward(cv_X, cv_parameters)
pred_test = (cv_AL > 0.5)

print("The CV accuracy: ")

print(pred_test)

#print("predicting")
#for xs in cv_X:
#	print(classifier_linear.predict(xs))

# for i in range(0, len(y)):
# 	if y[i] == "negative":
# 		y[i] = 0
# 	elif y[i] == "positive":
# 		y[i] = 1
# 	else:
# 		y[i] = 0.5




# def train_svm(X, Y, C, kernelFunction):
# 	max_passes = 5
# 	tol = 0.001
# 	m = len(X);
# 	n = len(X[0]);
# 	for i in range(0, len(Y)):
# 		if Y[i] == 0:
# 			Y[i] = -1
# 	alphas = np.zeros(m, 1);
# 	b = 0;
# 	E = np.zeros(m, 1);
# 	passes = 0;
# 	eta = 0;
# 	L = 0;
# 	H = 0;

# 	# 'linearKernel'
# 	K = np.dot(X, np.transpose(X))
# 	dots = 12
# 	while passes < max_passes:
# 		num_changed_alphas = 0
# 		for i in range(1:m):
# 			E[i] = b + np.sum(np.muliply(np.muliply(alphas, Y), [el for el in K][i])) - Y[i]
# 			if (Y[i]*E[i] < -tol and alphas[i] < C) or (Y[i]*E[i] > tol and alphas[i] > 0):
# 				j = math.ceil(m * np.random.rand())
# 				while j == i:
# 					j = math.ceil(m * np.random.rand())
# 				E[j] = b + np.sum(np.muliply(np.muliply(alphas, Y), [el for el in K][j])) - Y[i]
# 				alpha_i_old = alphas[i]
# 				alpha_j_old = alphas[j]
# 				if Y[i] == Y[j]:
# 					L = np.max(alphas(j) + alphas(i) - C)#0, 
# 					H = min(C, alphas(j) + alphas(i))
# 				else:
# 					L = max(0, alphas(j) - alphas(i))
# 					H = min(C, C + alphas(j) - alphas(i))







# find top words
# top_words_pos = {}
# top_words_neg = {}
# top_words_neu = {}

# for i in range(0, len(y)):
# 	this_text = texts[i].split()
# 	for word in this_text:
# 		if word in top_words_neg.keys():
# 			top_words_neg[word] = top_words_neg[word] + 1
# 		else:
# 			top_words_neg[word] = 1

# top_one = {}
# top_two = {}
# top_three = {}

# for key in top_words_neg.keys():
# 	if top_words_neg[key] > 100:
# 		top_one[key] = top_words_neg[key]
# 	if top_words_neg[key] > 200:
# 		top_two[key] = top_words_neg[key]
# 	if top_words_neg[key] > 300:
# 		top_three[key] = top_words_neg[key]

# print(len(top_one), len(top_two), len(top_three))

# save(top_one, "top_words.json")
# save(top_two, "top_words2.json")
# save(top_three, "top_words3.json")
# end of top words



# process data
# for i in range(0, len(y)):
# 	if y[i] == "negative":
# 		y[i] = 0
# 	elif y[i] == "positive":
# 		y[i] = 1
# 	else:
# 		y[i] = 0.5


# dict_data = []

# for s in range(0, int(len(texts)*0.7)):
# 	proc_s = process_str(texts[s])
# 	dict_data.append({'text': proc_s, 'sentiment': y[s], 'id': index[s]})
# 	print("preprocessed: " + str(s))

# save(dict_data, "proc_train.json")

# dict_data = []

# for s in range(int(len(texts)*0.7), len(texts)):
# 	proc_s = process_str(texts[s])
# 	dict_data.append({'text': proc_s, 'sentiment': y[s], 'id': index[s]})
# 	print("preprocessed: " + str(s))

# save(dict_data, "proc_cv.json")
# end of process data
		



