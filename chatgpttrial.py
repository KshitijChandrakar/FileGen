import numpy as np
import string, random
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

# Parameters
num_chars = 26 + 26 + 10  # 26 uppercase letters + 26 lowercase letters + 10 digits
sequence_length = 10
batch_size = 32
epochs = 10

# Generate random characters
def generate_random_characters(length):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

# Generate dataset
def generate_dataset(num_samples):
    X = []
    y = []
    for _ in range(num_samples):
        random_string = generate_random_characters(sequence_length + 1)
        X.append([ord(c) for c in random_string[:-1]])  # Convert characters to ASCII values
        y.append(ord(random_string[-1]))  # Convert character to ASCII value
    return np.array(X), np.array(y)

X_train, y_train = generate_dataset(1000)
X_val, y_val = generate_dataset(200)

# Convert to one-hot encoding
X_train = tf.one_hot(X_train, num_chars)
X_val = tf.one_hot(X_val, num_chars)

# Define the model
model = Sequential([
    LSTM(128, input_shape=(sequence_length, num_chars)),
    Dense(num_chars, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
