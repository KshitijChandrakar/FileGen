from Filenames import *
import tensorflow as tf
import numpy as np
from MyFunctions import *

LatestModel = getFilename('NN.keras')
model = tf.keras.models.load_model(LatestModel)
print(f"Using Model {LatestModel}")

#Read Data from the file
x_train, y_train, poolSize, pool, X, Y, DataDetails, inputSize, N, data = genData(N = 11)
print(pool)
predictions = model.predict(x_train)
maxiumProbab = maxIndex(predictions[0])[0][0]
print(pool[maxiumProbab], maxiumProbab, pool[y_train[0]], y_train[0])


# Print the evaluation results
x_train, y_train, poolSize, pool, X, Y, DataDetails, inputSize, N, data = genData()
loss, accuracy = model.evaluate(x_train, y_train)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
