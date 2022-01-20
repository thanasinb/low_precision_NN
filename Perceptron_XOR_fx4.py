# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 12:59:19 2021

@author: tolbo
"""

import numpy as np
import time
from fxpmath import Fxp as fxp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import collections

fxp_sign = True
word = 12
frac = 8
lim = fxp(0, signed=fxp_sign, n_word=word, n_frac=frac)
Logic_1 = fxp(1, signed=fxp_sign, n_word=word, n_frac=frac, rounding='around')
Logic_1n = fxp(Logic_1() * (-1), signed=fxp_sign, n_word=word, n_frac=frac, rounding='around')
Logic_0 = fxp(0, signed=fxp_sign, n_word=word, n_frac=frac, rounding='around')
Logic_half = Logic_1() / 2
epoch = 100000
learning_rate = 0.1
num_hidden = 2
idx_input = np.random.randint(4, size=epoch)

wih_00 = collections.deque(np.zeros(epoch))
wih_01 = collections.deque(np.zeros(epoch))
wih_02 = collections.deque(np.zeros(epoch))
wih_10 = collections.deque(np.zeros(epoch))
wih_11 = collections.deque(np.zeros(epoch))
wih_12 = collections.deque(np.zeros(epoch))
wih_20 = collections.deque(np.zeros(epoch))
wih_21 = collections.deque(np.zeros(epoch))
wih_22 = collections.deque(np.zeros(epoch))
who_00 = collections.deque(np.zeros(epoch))
who_01 = collections.deque(np.zeros(epoch))
who_02 = collections.deque(np.zeros(epoch))


class NeuralNetwork:
    def __init__(self, num_input_layer, num_hidden_layer, num_output_layer):
        # self.hidden_weights = fxp(np.random.uniform(Logic_1n, Logic_1, (num_hidden_layer + 1, num_input_layer + 1)),
        #                           signed=fxp_sign, n_word=word, n_frac=frac, rounding='around')
        # self.output_weights = fxp(np.random.uniform(Logic_1n, Logic_1, (num_output_layer, num_hidden_layer + 1)),
        #                           signed=fxp_sign, n_word=word, n_frac=frac, rounding='around')
        self.hidden_weights = np.random.uniform(-1, 1, (num_hidden_layer + 1, num_input_layer + 1))  # fp
        self.output_weights = np.random.uniform(-1, 1, (num_output_layer, num_hidden_layer + 1))  # fp


def feedForward(inputs, weights):
    dot_product = np.dot(weights, inputs)
    result_act = sigmoid(np.array(dot_product))
    # result_act = ReLU(np.array(dot_product))
    # result_fxp = fxp(result_act, signed=fxp_sign, n_word=word, n_frac=frac, rounding='around')

    return result_act  # fp
    # return result_fxp # fx


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
# set_answer = fxp(np.array([np.matrix('0'), np.matrix('1'), np.matrix('1'), np.matrix('0')]), signed=fxp_sign, n_word=word,
#                  n_frac=frac, rounding='around')
# set_answer = fxp(np.array([np.matrix('0'), np.matrix('1'), np.matrix('1'), np.matrix('0')]), signed=fxp_sign, n_word=word,
#                  n_frac=frac, rounding='around')
set_answer = np.array([np.matrix('0'), np.matrix('1'), np.matrix('1'), np.matrix('0')])  # fp

number_input = set_input.shape[0]
number_answer = set_answer.shape[0]

## SET NN PARAMETERS
nn = NeuralNetwork(2, num_hidden, 1)
count_correct = 0
count_error = 0
count_error_00 = 0
count_error_01 = 0
count_error_10 = 0
count_error_11 = 0
round_print = 500

fig = plt.figure(figsize=(20, 10), facecolor='#DEDEDE', dpi=200)
aih_00 = plt.subplot(431)
aih_01 = plt.subplot(432)
aih_02 = plt.subplot(433)
aih_10 = plt.subplot(434)
aih_11 = plt.subplot(435)
aih_12 = plt.subplot(436)
aih_20 = plt.subplot(437)
aih_21 = plt.subplot(438)
aih_22 = plt.subplot(439)
aho_00 = plt.subplot(4, 3, 10)
aho_01 = plt.subplot(4, 3, 11)
aho_02 = plt.subplot(4, 3, 12)
aih_00.set_facecolor('#DEDEDE')
aih_01.set_facecolor('#DEDEDE')
aih_02.set_facecolor('#DEDEDE')
aih_10.set_facecolor('#DEDEDE')
aih_11.set_facecolor('#DEDEDE')
aih_12.set_facecolor('#DEDEDE')
aih_20.set_facecolor('#DEDEDE')
aih_21.set_facecolor('#DEDEDE')
aih_22.set_facecolor('#DEDEDE')
aho_00.set_facecolor('#DEDEDE')
aho_01.set_facecolor('#DEDEDE')
aho_02.set_facecolor('#DEDEDE')

## TRAIN NEURAL NETWORK
for i in range(epoch):
    ## FEEDFORWARD
    input = set_input[idx_input[i]]
    # input = np.concatenate((input, [[Logic_1()]]), axis=0)  # ATTACH BIAS # fx
    # input = fxp(input, signed=fxp_sign, n_word=word, n_frac=frac, rounding='around') # fx
    input = np.concatenate((input, [[1]]), axis=0)  # ATTACH BIAS # fp

    wih_00.popleft()
    wih_01.popleft()
    wih_02.popleft()
    wih_10.popleft()
    wih_11.popleft()
    wih_12.popleft()
    wih_20.popleft()
    wih_21.popleft()
    wih_22.popleft()
    who_00.popleft()
    who_01.popleft()
    who_02.popleft()

    # wih_00.append(nn.hidden_weights[0][0]()) # fx
    # wih_01.append(nn.hidden_weights[0][1]()) # fx
    # wih_02.append(nn.hidden_weights[0][2]()) # fx
    # wih_10.append(nn.hidden_weights[1][0]()) # fx
    # wih_11.append(nn.hidden_weights[1][1]()) # fx
    # wih_12.append(nn.hidden_weights[1][2]()) # fx
    # wih_20.append(nn.hidden_weights[2][0]()) # fx
    # wih_21.append(nn.hidden_weights[2][1]()) # fx
    # wih_22.append(nn.hidden_weights[2][2]()) # fx
    # who_00.append(nn.output_weights[0][0]()) # fx
    # who_01.append(nn.output_weights[0][1]()) # fx
    # who_02.append(nn.output_weights[0][2]()) # fx

    wih_00.append(nn.hidden_weights[0][0])  # fp
    wih_01.append(nn.hidden_weights[0][1])  # fp
    wih_02.append(nn.hidden_weights[0][2])  # fp
    wih_10.append(nn.hidden_weights[1][0])  # fp
    wih_11.append(nn.hidden_weights[1][1])  # fp
    wih_12.append(nn.hidden_weights[1][2])  # fp
    wih_20.append(nn.hidden_weights[2][0])  # fp
    wih_21.append(nn.hidden_weights[2][1])  # fp
    wih_22.append(nn.hidden_weights[2][2])  # fp
    who_00.append(nn.output_weights[0][0])  # fp
    who_01.append(nn.output_weights[0][1])  # fp
    who_02.append(nn.output_weights[0][2])  # fp

    result_hidden = feedForward(input, nn.hidden_weights)
    y_guess = feedForward(result_hidden, nn.output_weights)
    y_answer = set_answer[idx_input[i]]

    ## XOR CALCULATE ACCURACY
    # num_y_guess = y_guess[0][0]  # fx
    # num_y_answer = y_answer()[0][0]  # fx
    num_y_guess = int(round(y_guess.item(0, 0), 0))  # fp
    num_y_answer = y_answer.item(0, 0)  # fp

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
        # A = input[0][0]()  # fx
        # B = input[1][0]()  # fx
        A = input[0][0]  # fp
        B = input[1][0]  # fp
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
        # with open("xor.csv", "ab") as f:
        #     f.write(b"Hidden weights\n")
        #     np.savetxt(f, nn.hidden_weights, delimiter=",")
        #     f.write(b"Input vector\n")
        #     np.savetxt(f, input, delimiter=",")
        #     f.write(b"Output weights\n")
        #     np.savetxt(f, nn.output_weights, delimiter=",")
        #     f.write(b"Hidden vector\n")
        #     np.savetxt(f, result_hidden, delimiter=",")
        #     f.write(b"Y guess\n")
        #     np.savetxt(f, y_guess, delimiter=",")
        #     f.write(b"Y answer\n")
        #     np.savetxt(f, y_answer, delimiter=",")
        #     f.write(b"Y guess logic\n")
        #     np.savetxt(f, num_y_guess, delimiter=",")
        #     f.write(b"Y answer logic\n")
        #     np.savetxt(f, num_y_answer, delimiter=",")
        #     f.write(b"\n")

        ## BACKPROPAGATION
        # output_error = y_answer() - y_guess()  # fx
        output_error = y_answer - y_guess  # fp
        # output_delta = np.dot(learning_rate * np.multiply(dSigmoid(y_guess()), output_error), result_hidden().T)  # fx
        # output_delta = np.dot(learning_rate * np.multiply(dReLU(y_guess()), output_error), result_hidden().T)  # Relu
        output_delta = np.dot(learning_rate * np.multiply(dSigmoid(y_guess), output_error), result_hidden.T)  # fp

        # hidden_error = np.dot(nn.output_weights().T, output_error)  # fx
        hidden_error = np.dot(nn.output_weights.T, output_error)  # fp

        # hidden_delta = np.dot(learning_rate * np.multiply(dSigmoid(result_hidden()), hidden_error), input().T)  # fx
        # hidden_delta = np.dot(learning_rate * np.multiply(dReLU(result_hidden()), hidden_error), input().T)
        hidden_delta = np.dot(learning_rate * np.multiply(dSigmoid(result_hidden), hidden_error), input.T)  # fp

        # output_weights_real = np.add(nn.output_weights(), output_delta) # fx
        # hidden_weights_real = np.add(nn.hidden_weights(), hidden_delta) # fx
        # nn.output_weights = fxp(output_weights_real, signed=fxp_sign, n_word=word, n_frac=frac, rounding='around') # fx
        # nn.hidden_weights = fxp(hidden_weights_real, signed=fxp_sign, n_word=word, n_frac=frac, rounding='around') # fx

        output_weights = np.add(nn.output_weights, output_delta)  # fp
        hidden_weights = np.add(nn.hidden_weights, hidden_delta)  # fp

        # with open("xor.csv", "ab") as f:
        #     f.write(b"Output error\n")
        #     np.savetxt(f, output_error, delimiter=",")
        #     f.write(b"Output delta\n")
        #     np.savetxt(f, output_delta, delimiter=",")
        #     f.write(b"Hidden error\n")
        #     np.savetxt(f, hidden_error, delimiter=",")
        #     f.write(b"Hidden delta\n")
        #     np.savetxt(f, hidden_delta, delimiter=",")
        #     f.write(b"New hidden weights real\n")
        #     np.savetxt(f, hidden_weights_real, delimiter=",")
        #     f.write(b"New output weights real\n")
        #     np.savetxt(f, output_weights_real, delimiter=",")
        #     f.write(b"New hidden weights fix\n")
        #     np.savetxt(f, nn.hidden_weights, delimiter=",")
        #     f.write(b"New output weights fix\n")
        #     np.savetxt(f, nn.output_weights, delimiter=",")
        #     f.write(b"\n")

    if (i == epoch - 1):
        print('epoch: ', i, 'accuracy: ', round((count_correct / i) * 100.0, 2), '%')
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

aih_00.plot(wih_00)
aih_01.plot(wih_01)
aih_02.plot(wih_02)
aih_10.plot(wih_10)
aih_11.plot(wih_11)
aih_12.plot(wih_12)
aih_20.plot(wih_20)
aih_21.plot(wih_21)
aih_22.plot(wih_22)
aho_00.plot(who_00)
aho_01.plot(who_01)
aho_02.plot(who_02)
plt.show()

## TEST NEURAL NETWORK
# for i in range(100):
#     ## FEEDFORWARD
#     input = set_input[i % number_input]
#     input = np.concatenate((input, [[Logic_1]]), axis=0)
#     input = fxp(input, signed=True, n_word=word, n_frac=frac, rounding='around')
#
#     result_hidden = feedForward(input, nn.hidden_weights)
#     y_guess = feedForward(result_hidden, nn.output_weights)
#     y_answer = set_answer[i % number_input]
#
#     ## XOR CALCULATE ACCURACY
#     num_y_guess = y_guess()
#     num_y_answer = y_answer()
#     if (num_y_guess > (Logic_1() / 2)):
#         num_y_guess = Logic_1()
#     else:
#         num_y_guess = Logic_0()
#
#     if (num_y_answer == num_y_guess):
#         count_correct += 1
#     else:
#         count_error += 1
#
# print('error: ', count_error)
# print('correct: ', count_correct)
# print('accuracy: ', round((count_correct / 100.0) * 100.0, 2), '%')
