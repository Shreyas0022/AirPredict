import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Video capture
cap = cv2.VideoCapture(0)

# Drawing variables
last_x, last_y = None, None
cursor_color = (0, 0, 255)  # red = idle, green = drawing
canvas = None

# Smoothing variables
smooth_x, smooth_y = 0, 0
alpha = 0.3  # smoothing factor


# ---------------- Palm Orientation Check ---------------- #
def is_palm_facing_camera(landmarks):
    """ Returns True if palm faces the camera, False if back of hand. """
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    index_mcp = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
    pinky_mcp = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])

    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist
    palm_normal = np.cross(v1, v2)

    return palm_normal[2] > 0  # palm toward camera


# ---------------- Finger State Check ---------------- #
def fingers_up(hand_landmarks):
    """ Return list of booleans for [Index, Middle, Ring, Pinky]. """
    fingers = []
    # Index
    fingers.append(hand_landmarks[8].y < hand_landmarks[6].y)
    # Middle
    fingers.append(hand_landmarks[12].y < hand_landmarks[10].y)
    # Ring
    fingers.append(hand_landmarks[16].y < hand_landmarks[14].y)
    # Pinky
    fingers.append(hand_landmarks[20].y < hand_landmarks[18].y)
    return fingers


# ---------------- Main Loop ---------------- #
with mp_hands.Hands(max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Index fingertip
                index_tip = hand_landmarks.landmark[8]
                x_index, y_index = int(index_tip.x * w), int(index_tip.y * h)

                # Palm orientation filter
                if is_palm_facing_camera(hand_landmarks.landmark):
                    # Finger states
                    finger_states = fingers_up(hand_landmarks.landmark)

                    # Smooth cursor
                    if smooth_x == 0 and smooth_y == 0:
                        smooth_x, smooth_y = x_index, y_index
                    else:
                        smooth_x = int(alpha * x_index + (1 - alpha) * smooth_x)
                        smooth_y = int(alpha * y_index + (1 - alpha) * smooth_y)

                    #  Strict Index-only mode: ONLY index = True
                    if finger_states == [True, False, False, False]:
                        cursor_color = (0, 255, 0)  # green
                        if last_x is not None and last_y is not None:
                            cv2.line(canvas, (last_x, last_y), (smooth_x, smooth_y), (0, 255, 0), 4, cv2.LINE_AA)
                        last_x, last_y = smooth_x, smooth_y
                    else:
                        cursor_color = (0, 0, 255)  # red
                        last_x, last_y = None, None

                    # Cursor always visible
                    cv2.circle(frame, (smooth_x, smooth_y), 8, cursor_color, -1)

                # Debug landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Merge
        combined = cv2.addWeighted(frame, 0.7, canvas, 1, 0)

        cv2.imshow("AirDraw - Strict Index Only", combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()
