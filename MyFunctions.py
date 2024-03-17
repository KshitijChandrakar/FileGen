def maxIndex(input_list):
    import numpy as np
    max_value = max(input_list)  # Find the maximum value in the list
    max_index = np.where(input_list == max_value)  # Find the index of the maximum value
    return max_index
