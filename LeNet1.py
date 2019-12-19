# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:50:24 2019

@author: anton
"""

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from keras.utils import np_utils

# Obtaining MNIST dataset
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

# Reshaping the data to be used by keras
xTrain = xTrain.reshape(xTrain.shape[0], 28, 28, 1)
xTest = xTest.reshape(xTest.shape[0], 28, 28, 1)
xTrain = xTrain.astype('float32')
xTest = xTest.astype('float32')
xTrain /= 255
xTest /= 255

# Preparing categories class as CNN output
yTrain = np_utils.to_categorical(yTrain, 10)
yTest = np_utils.to_categorical(yTest, 10)

# Initial xTrain data shape
input_shape=(28, 28, 1)

# Creating LeNet model
lenet = Sequential()
lenet.add(Conv2D(12, (5, 5), activation='relu', input_shape=input_shape))
lenet.add(Conv2D(12, (5, 5), activation='relu'))
lenet.add(Flatten())
lenet.add(Dense(30, activation='relu'))
lenet.add(Dense(10, activation='softmax'))

# Compiling LeNet model to configure the learning process
lenet.compile(optimizer=optimizers.Adam(0.01), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

lenet.fit(xTrain, yTrain, batch_size=256, epochs=30, verbose=1)

final_loss, final_acc = lenet.evaluate(xTest, yTest, batch_size=128, verbose=2)

print('Final loss = ', final_loss)
print('Final accuracy = ', final_acc)