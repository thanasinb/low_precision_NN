# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 12:59:19 2021

@author: tolbo
"""

import numpy as np

# import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, num_input_layer, num_hidden_layer, num_output_layer, max_value):
        self.hidden_weights = np.random.uniform(-1, 1, (num_hidden_layer + 1, num_input_layer  + 1))
        self.output_weights = np.random.uniform(-1, 1, (num_output_layer, num_hidden_layer + 1))
        self.hidden_weights = np.array([np.round(x*max_value, 0) for x in self.hidden_weights]).astype(int)
        self.output_weights = np.array([np.round(x*max_value, 0) for x in self.output_weights]).astype(int)

def feedForward (inputs, weights):
    rows = len(inputs)
    scale = rows * max_value
    dp = np.dot(weights, inputs)
    dp_scale = dp/scale
    result = ReLU(dp_scale)
    return result

def sigmoid (x):
    return 1/(1 + np.exp(-x))

def dSigmoid (x):
    return np.multiply (x, (1-x))

def ReLU (x):
    return np.maximum(0,x)

def dReLU (x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def loadMNIST(prefix):
    intType = np.dtype( 'int32' ).newbyteorder( '>' )
    nMetaDataBytes = 4 * intType.itemsize
    
    data = np.fromfile(prefix + '-images.idx3-ubyte', dtype = 'ubyte')
    magicBytes, nImages, width, height = np.frombuffer( data[:nMetaDataBytes].tobytes(), intType )
    data = data[nMetaDataBytes:].astype( dtype = 'float32' ).reshape( [ nImages, width, height ] )
    labels = np.fromfile(prefix + '-labels.idx1-ubyte', dtype = 'ubyte' )[2 * intType.itemsize:]
 
    return data, labels

## MNIST LOAD MNIST TRAIN AND TEST IMAGES
# train_images, train_labels = loadMNIST("train")
# test_images, test_labels = loadMNIST("t10k")
    
## MNIST: PLOT SAMPLE MNIST IMAGES
# img1_arr, img1_label = train_images[3], train_labels[3]
# plt.subplot(111) # 111 MEANS 1x1 GRID, FIRST SUBPLOT
# plt.imshow(img1_arr, cmap=plt.get_cmap('gray'))
# plt.show()

## PREPARE NN INPUT
# set_input = np.array([np.reshape(item,(784, 1)) for item in train_images])
# set_input = set_input/np.amax(set_input)
# number_input = set_input.shape[0]
# set_answer = np.zeros((set_input.shape[0], 10, 1))
# set_answer[range(set_answer.shape[0]), train_labels]=1

## XOR INPUT AND OUTPUT
set_input=np.array([np.matrix('0; 0'),np.matrix('0; 1'),np.matrix('1; 0'),np.matrix('1; 1')])
set_answer = np.matrix('0; 1; 1; 0')
number_input = set_input.shape[0]
number_answer = set_answer.shape[0]

## SET NN PARAMETERS
max_value = 7
scale = len(set_input[0])
nn = NeuralNetwork(2, 2, 1, max_value)
learning_rate = 0.1
epoch = 100000
count_correct = 0
count_error = 0

## TRAIN NEURAL NETWORK
for i in range(epoch):
    ## FEEDFORWARD
    input = set_input[i%number_input]
    input = np.concatenate((input, [[1]]), axis=0)
    
    result_hidden = feedForward(input, nn.hidden_weights)
    y_guess = feedForward(result_hidden, nn.output_weights)
    
    
    ## BACKPROPAGATION
    y_answer = set_answer[i%number_input]
    output_error = y_answer - y_guess
    output_delta = np.dot(learning_rate * np.multiply(dSigmoid(y_guess), output_error), result_hidden.T)
    
    hidden_error = np.dot(nn.output_weights.T, output_error)
    hidden_delta = np.dot(learning_rate * np.multiply(dSigmoid(result_hidden), hidden_error), input.T)
    
    nn.output_weights = np.add(nn.output_weights, output_delta)
    nn.hidden_weights = np.add(nn.hidden_weights, hidden_delta)
    
    ## XOR CALCULATE ACCURACY
    num_y_guess  = int(round(y_guess.item(0, 0),0))
    num_y_answer = y_answer.item(0, 0)
    
    ## NMIST CALCULATE ACCURACY
    # num_y_answer = np.argmax(y_answer, axis=0)
    # num_y_guess = np.argmax(y_guess, axis=0)
    
    if (num_y_answer == num_y_guess):
        count_correct += 1
    else:
        count_error += 1
    if (i%10000==0):
        print ('epoch: ', i, 'accuracy: ', round((count_correct/10000)*100, 2), '%')
        count_correct = 0
        count_error = 0
        
count_correct = 0
count_error = 0

## MNIST: SETUP TEST INPUT
# set_input = np.array([np.reshape(item,(784, 1)) for item in test_images])
# set_input = set_input/np.amax(set_input)
# number_input = set_input.shape[0]
# set_answer = np.zeros((set_input.shape[0], 10, 1))
# set_answer[range(set_answer.shape[0]), test_labels]=1

## TEST NEURAL NETWORK
for i in range (100):
    ## FEEDFORWARD
    input = set_input[i%number_input]
    input = np.concatenate((input, [[1]]), axis=0)
    
    result_hidden = feedForward(input, nn.hidden_weights)
    y_guess = feedForward(result_hidden, nn.output_weights)
    y_answer = set_answer[i%number_input]
    
    ## XOR CALCULATE ACCURACY
    num_y_guess  = int(round(y_guess.item(0, 0),0))
    num_y_answer = y_answer.item(0, 0)
    
    ## NMIST CALCULATE ACCURACY
    # num_y_answer = np.argmax(y_answer, axis=0)
    # num_y_guess = np.argmax(y_guess, axis=0)
    
    if (num_y_answer == num_y_guess):
        count_correct += 1
    else:
        count_error += 1
        # print ('(' , input , ',' , y_guess, ',' , num_y_guess ,')')
    
print ('error: ', count_error)
print ('correct: ', count_correct)
print ('accuracy: ', round((count_correct/100)*100, 2), '%')
