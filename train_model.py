import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define gestures
GESTURES = ["hello", "you", "me", "money", "go", "drink", "eat", "good", "like",
            "dislike", "1", "2", "3"]

# Load data
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Define the model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 21 * 3)),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(len(GESTURES), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))  # Adjust epochs as needed

# Save the model
model.save('gesture_recognition_model.keras')
