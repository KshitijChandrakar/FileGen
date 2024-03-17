def maxIndex(input_list):
    import numpy as np
    max_value = max(input_list)  # Find the maximum value in the list
    max_index = np.where(input_list == max_value)  # Find the index of the maximum value
    return max_index
#Read Data from the file
#Generate and Format the Training data from substrings of length inputSize to lists of inputSize integers
def genData(N = -1, inputSize = 10):
    global x_train, y_train, poolSize, pool, X, Y, DataDetails
    import numpy as np
    with open('DataDetails.json', 'r') as file1:
        import json
        DataDetails = json.load(file1)
    file = open('testfile.txt', 'r')
    fileData = file.read(N) if N != -1 else file.read()
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
    dataEncoded = np.array([char_to_index[substring] for substring in fileData])
    return x_train, y_train, poolSize, pool, X, Y, DataDetails, inputSize, N, dataEncoded
