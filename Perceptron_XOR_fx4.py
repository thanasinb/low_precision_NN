# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 12:59:19 2021

@author: tolbo
"""

import numpy as np
import time
from fxpmath import Fxp as fxp

# import matplotlib.pyplot as plt

word = 4
frac = 3
lim = fxp(0, signed=True, n_word=word, n_frac=frac)
Logic_1 = fxp(1, signed=True, n_word=word, n_frac=frac, rounding='around')
Logic_1n = fxp(Logic_1() * (-1), signed=True, n_word=word, n_frac=frac, rounding='around')
Logic_0 = fxp(0, signed=True, n_word=word, n_frac=frac, rounding='around')


class NeuralNetwork:
    def __init__(self, num_input_layer, num_hidden_layer, num_output_layer):
        self.hidden_weights = fxp(np.random.uniform(Logic_1n, Logic_1, (num_hidden_layer + 1, num_input_layer + 1)),
                                  signed=True, n_word=word, n_frac=frac, rounding='around')
        self.output_weights = fxp(np.random.uniform(Logic_1n, Logic_1, (num_output_layer, num_hidden_layer + 1)),
                                  signed=True, n_word=word, n_frac=frac, rounding='around')


def feedForward(inputs, weights):
    dot_product = np.dot(weights, inputs)
    result_act = sigmoid(np.array(dot_product))
    result_fxp = fxp(result_act, signed=True, n_word=word, n_frac=frac, rounding='around')

    return result_fxp


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dSigmoid(x):
    return np.multiply(x, (1 - x))


def ReLU(x):
    return np.maximum(0, x)


def dReLU(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


t = time.time()

## XOR INPUT AND OUTPUT
set_input = np.array([np.matrix('0; 0'), np.matrix('0; 1'), np.matrix('1; 0'), np.matrix('1; 1')])
set_answer = fxp(np.array([[0], [1], [1], [0]]), signed=True, n_word=word, n_frac=frac, rounding='around')

number_input = set_input.shape[0]
number_answer = set_answer.shape[0]

## SET NN PARAMETERS
nn = NeuralNetwork(2, 2, 1)
learning_rate = 0.1
epoch = 50000
count_correct = 0
count_error = 0
round_print = 500

## TRAIN NEURAL NETWORK
for i in range(epoch):
    ## FEEDFORWARD
    input = set_input[i % number_input]
    input = np.concatenate((input, [[Logic_1()]]), axis=0)  # ATTACH BIAS
    input = fxp(input, signed=True, n_word=word, n_frac=frac, rounding='around')

    result_hidden = feedForward(input, nn.hidden_weights)
    y_guess = feedForward(result_hidden, nn.output_weights)
    y_answer = set_answer[i % number_input]

    ## XOR CALCULATE ACCURACY
    num_y_guess = y_guess()
    num_y_answer = y_answer()
    if (num_y_guess > (Logic_1() / 2)):
        num_y_guess = Logic_1()
    else:
        num_y_guess = Logic_0()

    ## NMIST CALCULATE ACCURACY
    # num_y_answer = np.argmax(y_answer, axis=0)
    # num_y_guess = np.argmax(y_guess, axis=0)

    if (num_y_answer == num_y_guess):
        count_correct += 1
    else:
        count_error += 1

        ## BACKPROPAGATION
        output_error = y_answer() - y_guess()
        output_delta = np.dot(learning_rate * np.multiply(dSigmoid(y_guess()), output_error), result_hidden().T)

        hidden_error = np.dot(nn.output_weights().T, output_error)
        hidden_delta = np.dot(learning_rate * np.multiply(dSigmoid(result_hidden()), hidden_error), input().T)

        output_weights_real = np.add(nn.output_weights(), output_delta)
        hidden_weights_real = np.add(nn.hidden_weights(), hidden_delta)
        nn.output_weights = fxp(output_weights_real, signed=True, n_word=word, n_frac=frac, rounding='around')
        nn.hidden_weights = fxp(hidden_weights_real, signed=True, n_word=word, n_frac=frac, rounding='around')

        # nn.output_weights[nn.output_weights>Logic_1 ]=Logic_1
        # nn.output_weights[nn.output_weights<Logic_1n]=Logic_1n
        # nn.hidden_weights[nn.hidden_weights>Logic_1 ]=Logic_1
        # nn.hidden_weights[nn.hidden_weights<Logic_1n]=Logic_1n

    if (i % round_print == 0):
        print('epoch: ', i, 'accuracy: ', round((count_correct / round_print) * 100.0, 2), '%')
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
for i in range(100):
    ## FEEDFORWARD
    input = set_input[i % number_input]
    input = np.concatenate((input, [[Logic_1]]), axis=0)
    input = fxp(input, signed=True, n_word=word, n_frac=frac, rounding='around')

    result_hidden = feedForward(input, nn.hidden_weights)
    y_guess = feedForward(result_hidden, nn.output_weights)
    y_answer = set_answer[i % number_input]

    ## XOR CALCULATE ACCURACY
    num_y_guess = y_guess()
    num_y_answer = y_answer()
    if (num_y_guess > (Logic_1() / 2)):
        num_y_guess = Logic_1()
    else:
        num_y_guess = Logic_0()

    ## NMIST CALCULATE ACCURACY
    # num_y_answer = np.argmax(y_answer, axis=0)
    # num_y_guess = np.argmax(y_guess, axis=0)

    if (num_y_answer == num_y_guess):
        count_correct += 1
    else:
        count_error += 1

print('error: ', count_error)
print('correct: ', count_correct)
print('accuracy: ', round((count_correct / 100.0) * 100.0, 2), '%')
