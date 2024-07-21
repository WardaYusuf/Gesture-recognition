import numpy as np
import os
from sklearn.model_selection import train_test_split

DATA_PATH = 'sign_language_data'
GESTURES = ["hello", "you", "me", "money", "go", "drink", "eat", "good", "like",
            "dislike", "1", "2", "3"]

num_frames = 30  # Number of frames per sample
num_landmarks = 21  # Number of hand landmarks
num_features = 3  # Number of coordinates per landmark

def load_data():
    X, y = [], []
    for label, gesture in enumerate(GESTURES):
        gesture_path = os.path.join(DATA_PATH, gesture)
        for file in os.listdir(gesture_path):
            if file.endswith('.npy'):
                data = np.load(os.path.join(gesture_path, file))
                # Ensure each sample is (30, 21, 3)
                if data.shape[0] > num_frames:
                    data = data[:num_frames]
                elif data.shape[0] < num_frames:
                    data = np.pad(data, ((0, num_frames - data.shape[0]), (0, 0), (0, 0)), mode='constant')
                X.append(data)
                y.append(label)
    X = np.array(X)  # Shape: (num_samples, 30, 21, 3)
    y = np.array(y)
    # Reshape X to (num_samples, 30, 63)
    X = X.reshape(X.shape[0], X.shape[1], num_landmarks * num_features)
    return X, y

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
