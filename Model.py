import tensorflow as tf
from Filenames import *
import numpy as np
from os import getcwd as cwd
# Read the dictionary from the file
with open('DataDetails.json', 'r') as file1:
    import json
    DataDetails = json.load(file1)
    print(DataDetails)
inputSize = 20


#Read Data from the file
file = open('testfile.txt', 'r')
fileData = file.read()
#Generate and Format the Training data from substrings of length inputSize to lists of inputSize integers
def genData(fileData):
    global x_train, y_train, poolSize, pool
    X,Y = [],[]
    for j in range(0,len(fileData) - inputSize):
        k = j + inputSize
        X.append(fileData[j:k])
        Y.append(fileData[k])
    poolSize = DataDetails['PoolSize']
    pool = sorted(set(DataDetails['pool']))
    char_to_index = {char: i for i, char in enumerate(pool)}
    num_chars = poolSize
    x_train = np.array([[char_to_index[char] for char in substring] for substring in X])
    y_train = np.array([char_to_index[char] for char in Y])
genData(fileData)
def printData():
    for i in range(len(x_train)):
        print(x_train[i], y_train[i], sep=' : ')
# printData()

model = tf.keras.models.Sequential([
  tf.keras.layers.LSTM(512, input_shape=(inputSize, 1) ),
  tf.keras.layers.Dense(poolSize, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)
model.save(cwd() + '/' + generateFilename('Model.keras'))
'''
model.evaluate(x_test, y_test)
'''
