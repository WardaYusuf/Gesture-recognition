import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Define gestures
GESTURES = ["hello", "you", "me", "money", "go", "drink", "eat", "good", "like",
            "dislike", "1", "2", "3"]

# Load the trained model
model = tf.keras.models.load_model('gesture_recognition_model.h5')
# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def preprocess_frame(frame):
    # Preprocess the frame as per the model's requirement
    # Assume the frame is of shape (30, 21, 3) or similar
    if frame.shape[0] > 30:
        frame = frame[:30]
    elif frame.shape[0] < 30:
        frame = np.pad(frame, ((0, 30 - frame.shape[0]), (0, 0), (0, 0)), mode='constant')
    frame = frame.reshape(1, 30, 63)  # Reshape to (1, 30, 63)
    return frame


def main():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        frames = []
        while True:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]
                    frames.append(landmarks)
                    if len(frames) > 30:
                        frames.pop(0)  # Keep only the last 30 frames

                if len(frames) == 30:
                    frame_array = np.array(frames)
                    frame_array = preprocess_frame(frame_array)
                    predictions = model.predict(frame_array)
                    gesture = GESTURES[np.argmax(predictions)]

                    cv2.putText(image, f'Predicted: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Real-Time Gesture Recognition', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
