import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up directories
DATA_PATH = 'sign_language_data'
GESTURES = ["hello", "you", "me", "money", "go", "drink", "eat", "good", "like",
            "dislike", "1", "2", "3"]

os.makedirs(DATA_PATH, exist_ok=True)
for gesture in GESTURES:
    os.makedirs(os.path.join(DATA_PATH, gesture), exist_ok=True)

def collect_data():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        for gesture in GESTURES:
            print(f"Collecting data for: {gesture}")
            for sample_num in range(15):  # Collect only 15 samples per gesture
                frames = []
                while len(frames) < 30:  # Collect 30 frames per sample
                    success, image = cap.read()
                    if not success:
                        continue

                    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                    results = hands.process(image)
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]
                            frames.append(landmarks)
                            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    cv2.imshow('Hand Tracking', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frames = np.array(frames)
                np.save(os.path.join(DATA_PATH, gesture, f'{gesture}_{sample_num}.npy'), frames)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    collect_data()
