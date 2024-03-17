from Filenames import *
import tensorflow as tf
import numpy as np
from MyFunctions import *
LatestModel = getFilename('Model.keras')
model = tf.keras.models.load_model(LatestModel)
with open('DataDetails.json', 'r') as file1:
    import json
    DataDetails = json.load(file1)
    print(DataDetails)
print(f"Using Model {LatestModel}")
#Read Data from the file
file = open('testfile.txt', 'r')
fileData = file.read(11)
def genData(fileData):
    global x_train, y_train, poolSize, pool, X, Y
    X,Y = [],[]
    for j in range(0,len(fileData) - 10):
        k = j + 10
        X.append(fileData[j:k])
        Y.append(fileData[k])
    poolSize = DataDetails['PoolSize']
    pool = sorted(set(DataDetails['pool']))
    char_to_index = {char: i for i, char in enumerate(pool)}
    num_chars = poolSize
    x_train = np.array([[char_to_index[char] for char in substring] for substring in X])
    y_train = np.array([char_to_index[char] for char in Y])
genData(fileData)
pool2 = DataDetails['pool']
# print(x_train, y_train,  X, Y, sep='\n\n')
# loss, accuracy = model.evaluate(x_train, y_train)
#
# # Print the evaluation results
# print("Test Loss:", loss)
# print("Test Accuracy:", accuracy)
predictions = model.predict(x_train)
print(pool2[maxIndex(predictions[0])[0][0]], pool2[y_train[0]])
