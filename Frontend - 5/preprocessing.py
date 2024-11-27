import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands

def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    hand_motions = []

    # Reuse the Mediapipe Hands object
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

    frame_count = 0
    process_every_nth_frame = 5  # Adjust this value as needed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process every nth frame to reduce computation
        if frame_count % process_every_nth_frame != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_motion_frame = []
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    h, w, _ = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    hand_motion_frame.extend([x, y])
            # Ensure we always get 42 features
            if len(hand_motion_frame) == 42:
                hand_motions.append(hand_motion_frame)
            else:
                print("Error: Data length mismatch")

    cap.release()
    hands.close()
    return np.array(hand_motions)
