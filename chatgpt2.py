import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

# Example string
N = "dhvbjkclsvblkbvdkjbv,.lsajkbvksbdjvbkshjbvkhsbvbkslviuwehvnbhfuoivsiubvuhjvdlsjiklbkdsviusavdjlbjkslvbhdyoibulvubaoslivbsvuyvbysvbuvsb"
sequence_length = 10  # Length of substrings
num_samples = len(N) - sequence_length  # Number of samples

# Generate dataset
X = []
y = []
for i in range(num_samples):
    substring = N[i:i+sequence_length]
    target_character = N[i+sequence_length]
    X.append(substring)
    y.append(target_character)

# Convert characters to numerical representation (one-hot encoding)
characters = sorted(set(N))
char_to_index = {char: i for i, char in enumerate(characters)}
num_chars = len(characters)
X_encoded = np.array([[char_to_index[char] for char in substring] for substring in X])
y_encoded = np.array([char_to_index[char] for char in y])
print(X)
# Define the model
model = Sequential([
    LSTM(128, input_shape=(sequence_length, num_chars)),
    Dense(num_chars, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_encoded, y_encoded, batch_size=32, epochs=10, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_encoded, y_encoded)
print('Test accuracy:', test_acc)
