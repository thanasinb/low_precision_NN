# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 12:59:19 2021

@author: tolbo
"""

import numpy as np
import time
from fxpmath import Fxp as fxp
import pandas as pd

# import matplotlib.pyplot as plt

word = 4
frac = 3
lim = fxp(0, signed=True, n_word=word, n_frac=frac)
Logic_1 = fxp(1, signed=True, n_word=word, n_frac=frac, rounding='around')
Logic_1n = fxp(Logic_1() * (-1), signed=True, n_word=word, n_frac=frac, rounding='around')
Logic_0 = fxp(0, signed=True, n_word=word, n_frac=frac, rounding='around')
Logic_half = Logic_1() / 2


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
set_answer = fxp(np.array([np.matrix('0'), np.matrix('1'), np.matrix('1'), np.matrix('0')]), signed=True, n_word=word,
                 n_frac=frac, rounding='around')

number_input = set_input.shape[0]
number_answer = set_answer.shape[0]

## SET NN PARAMETERS
nn = NeuralNetwork(2, 10, 1)
learning_rate = 5
epoch = 50000
count_correct = 0
count_error = 0
count_error_00 = 0
count_error_01 = 0
count_error_10 = 0
count_error_11 = 0
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
    num_y_guess = y_guess[0][0]
    num_y_answer = y_answer()[0][0]

    if num_y_guess > Logic_half:
        num_y_guess = 1
    else:
        num_y_guess = 0

    if num_y_answer > Logic_half:
        num_y_answer = 1
    else:
        num_y_answer = 0

    if num_y_answer == num_y_guess:
        count_correct += 1
    else:
        count_error += 1
        A = input[0][0]()
        B = input[1][0]()
        if A == 0 and B == 0:
            count_error_00 += 1
        elif A == 0 and B > Logic_half:
            count_error_01 += 1
        elif A > Logic_half and B == 0:
            count_error_10 += 1
        elif A > Logic_half and B > Logic_half:
            count_error_11 += 1

        num_y_guess = np.matrix(num_y_guess)
        num_y_answer = np.matrix(num_y_answer)
        with open("xor.csv", "ab") as f:
            f.write(b"Hidden weights\n")
            np.savetxt(f, nn.hidden_weights, delimiter=",")
            f.write(b"Input vector\n")
            np.savetxt(f, input, delimiter=",")
            f.write(b"Output weights\n")
            np.savetxt(f, nn.output_weights, delimiter=",")
            f.write(b"Hidden vector\n")
            np.savetxt(f, result_hidden, delimiter=",")
            f.write(b"Y guess\n")
            np.savetxt(f, y_guess, delimiter=",")
            f.write(b"Y answer\n")
            np.savetxt(f, y_answer, delimiter=",")
            f.write(b"Y guess logic\n")
            np.savetxt(f, num_y_guess, delimiter=",")
            f.write(b"Y answer logic\n")
            np.savetxt(f, num_y_answer, delimiter=",")
            f.write(b"\n")

        ## BACKPROPAGATION
        output_error = y_answer() - y_guess()
        output_delta = np.dot(learning_rate * np.multiply(dSigmoid(y_guess()), output_error), result_hidden().T)

        hidden_error = np.dot(nn.output_weights().T, output_error)
        hidden_delta = np.dot(learning_rate * np.multiply(dSigmoid(result_hidden()), hidden_error), input().T)

        output_weights_real = np.add(nn.output_weights(), output_delta)
        hidden_weights_real = np.add(nn.hidden_weights(), hidden_delta)
        nn.output_weights = fxp(output_weights_real, signed=True, n_word=word, n_frac=frac, rounding='around')
        nn.hidden_weights = fxp(hidden_weights_real, signed=True, n_word=word, n_frac=frac, rounding='around')

        with open("xor.csv", "ab") as f:
            f.write(b"Output error\n")
            np.savetxt(f, output_error, delimiter=",")
            f.write(b"Output delta\n")
            np.savetxt(f, output_delta, delimiter=",")
            f.write(b"Hidden error\n")
            np.savetxt(f, hidden_error, delimiter=",")
            f.write(b"Hidden delta\n")
            np.savetxt(f, hidden_delta, delimiter=",")
            f.write(b"New hidden weights real\n")
            np.savetxt(f, hidden_weights_real, delimiter=",")
            f.write(b"New output weights real\n")
            np.savetxt(f, output_weights_real, delimiter=",")
            f.write(b"New hidden weights fix\n")
            np.savetxt(f, nn.hidden_weights, delimiter=",")
            f.write(b"New output weights fix\n")
            np.savetxt(f, nn.output_weights, delimiter=",")
            f.write(b"\n")

    if (i % round_print == 0):
        print('epoch: ', i, 'accuracy: ', round((count_correct / round_print) * 100.0, 2), '%')
        print('error 00: ', count_error_00)
        print('error 01: ', count_error_01)
        print('error 10: ', count_error_10)
        print('error 11: ', count_error_11)
        print('Wh: ', nn.hidden_weights)
        print('Wo: ', nn.output_weights)
        count_correct = 0
        count_error = 0
        count_error_00 = 0
        count_error_01 = 0
        count_error_10 = 0
        count_error_11 = 0

count_correct = 0
count_error = 0
count_error_00 = 0
count_error_01 = 0
count_error_10 = 0
count_error_11 = 0

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

    if (num_y_answer == num_y_guess):
        count_correct += 1
    else:
        count_error += 1

print('error: ', count_error)
print('correct: ', count_correct)
print('accuracy: ', round((count_correct / 100.0) * 100.0, 2), '%')
