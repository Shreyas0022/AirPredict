import cv2
import numpy as np
import mediapipe as mp

# --- Global Initializations ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# --- NEW: Refactored Gesture Detection Function (with SWAPPED logic) ---
def get_gesture_mode(hand_landmarks):
    """
    Analyzes hand landmarks to determine the current gesture mode.
    Args:
        hand_landmarks: The detected landmarks for a single hand.
    Returns:
        A string: "DRAW", "MOVE", or "NONE".
    """
    # Get landmarks for index and middle fingers
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

    # Check if fingers are raised
    index_finger_up = index_tip.y < index_pip.y
    middle_finger_up = middle_tip.y < middle_pip.y

    # --- SWAPPED LOGIC ---
    # We must check for the more specific gesture (two fingers) FIRST.
    if index_finger_up and middle_finger_up:
        return "MOVE"  # Two fingers up is now MOVE mode
    elif index_finger_up:
        return "DRAW"  # One finger up is now DRAW mode
    else:
        return "NONE"

# --- Main Application Loop ---
points = []
canvas = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            height, width, _ = frame.shape
            
            mode = get_gesture_mode(hand_landmarks)
            
            index_tip_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            cx, cy = int(index_tip_landmark.x * width), int(index_tip_landmark.y * height)

            if mode == "DRAW":
                cv2.circle(frame, (cx, cy), 12, (0, 255, 0), cv2.FILLED)
                points.append((cx, cy))
                if len(points) > 1:
                    cv2.line(canvas, points[-2], points[-1], (0, 255, 0), 5)

            elif mode == "MOVE":
                cv2.circle(frame, (cx, cy), 12, (255, 0, 0), cv2.FILLED)
                points.clear()
            
            else:
                points.clear()

    frame = cv2.add(frame, canvas)
    cv2.imshow('AirPredict - Gestures Swapped', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
        points.clear()

cap.release()
cv2.destroyAllWindows()
hands.close()