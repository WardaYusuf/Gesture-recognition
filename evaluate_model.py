import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf

# Define gestures
GESTURES = ["hello", "you", "me", "money", "go", "drink", "eat", "good", "like",
            "dislike", "1", "2", "3"]

# Load data and model
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
model = tf.keras.models.load_model('gesture_recognition_model.keras')

# Make predictions
y_pred = np.argmax(model.predict(X_test), axis=1)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

class_report = classification_report(y_test, y_pred, target_names=GESTURES)
print('Classification Report:')
print(class_report)
