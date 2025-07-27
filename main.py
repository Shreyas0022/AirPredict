import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# NEW: A list to store the points of our drawing line
points = []

# NEW: A variable for our drawing canvas
canvas = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror-like view
    frame = cv2.flip(frame, 1)

    # NEW: Initialize the canvas on the first frame
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convert the BGR image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB frame to find hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the frame dimensions
            height, width, _ = frame.shape
            
            # Access the specific landmark for the index fingertip
            index_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Convert normalized coordinates to pixel coordinates
            cx, cy = int(index_fingertip.x * width), int(index_fingertip.y * height)
            
            # Draw a circle on the index fingertip to show the "pen"
            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

            # NEW: Add the current point to our list of points
            points.append((cx, cy))

            # NEW: Draw lines on the canvas if we have at least two points
            if len(points) > 1:
                # Draw a line from the previous point to the current point
                cv2.line(canvas, points[-2], points[-1], (0, 255, 0), 5) # Green line, 5px thick

    # NEW: Combine the live frame with our canvas
    # This overlays the drawing onto the video
    # We use cv2.add to merge them.
    frame = cv2.add(frame, canvas)

    # Display the resulting frame
    cv2.imshow('AirPredict - Drawing Canvas', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()