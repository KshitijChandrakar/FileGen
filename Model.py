import tensorflow as tf
from Filenames import *
import numpy as np
from os import getcwd as cwd
from MyFunctions import *
# Read the dictionary from the file
x_train, y_train, poolSize, pool, X, Y, DataDetails, inputSize, N = genData()
# printData()
def printData():
    for i in range(len(x_train)):
        print(x_train[i], y_train[i], sep=' : ')

model = tf.keras.models.Sequential([
  tf.keras.layers.LSTM(128, input_shape=(inputSize, 1) ),
  tf.keras.layers.Dense(poolSize, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=500)
model.save(cwd() + '/' + generateFilename('NN.keras'))
