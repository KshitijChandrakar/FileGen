from Filenames import *
import tensorflow as tf
import numpy as np
from MyFunctions import *
import sys
LatestModel = getFilename('NN.keras')
model = tf.keras.models.load_model(LatestModel)
print(f"Using Model {LatestModel}")

#Read Data from the file
x_train, y_train, poolSize, pool, X, Y, DataDetails, inputSize, N, data = genData(N = 11)
runs = 989
total = N + runs
# print(x_train)
# x_train = x_train[0][1:]
# print(x_train)
# x_train = np.append(x_train[0], maxIndex(predictions[0])[0])

# print(x_train[0][1:].append(maxIndex(predictions[0])[0]))
# print(pool[maxiumProbab], maxiumProbab, pool[y_train[0]], y_train[0])
data1 = x_train.flatten()
def x(N):
    global x_train, data1
    for i in range(N):
        predictions = model.predict(x_train)
        maxProbab = maxIndex(predictions[0])[0]
        x_train = np.array([np.append(x_train[0][1:], maxProbab)])
        data1 = np.array(np.append(data1, maxProbab))
    # data[1:] + predicted
x(runs)
x_train, y_train, poolSize, pool, X, Y, DataDetails, inputSize, N, data = genData(total)
print(data1)
print(data)
def compare(A, B):
    loss = 0
    accurate = 0
    for i in range(len(A)):
        if A[i] == B[i]:
            accurate += 1
        else:
            loss += 1
    return loss, accurate
print(compare(data1, data))
# Print the evaluation results
# x_train, y_train, poolSize, pool, X, Y, DataDetails, inputSize, N, data = genData()
# loss, accuracy = model.evaluate(x_train, y_train)
# print("Test Loss:", loss)
# print("Test Accuracy:", accuracy)
