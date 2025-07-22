import cv2
import numpy as np
import mediapipe as mp # NEW: Import MediaPipe

# NEW: Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam, explicitly telling it to use the DirectShow backend.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # If the frame was read correctly, ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Flip the frame horizontally for a more intuitive, mirror-like view
    frame = cv2.flip(frame, 1)

    # NEW: Process the frame with MediaPipe
    # Convert the BGR image to RGB, as MediaPipe requires RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB frame to find hands
    results = hands.process(rgb_frame)

    # NEW: Draw the hand annotations on the frame
    if results.multi_hand_landmarks:
        # Loop through each detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            # Use MediaPipe's drawing utility to draw the landmarks and connections
            mp_drawing.draw_landmarks(
                frame,                  # The frame to draw on
                hand_landmarks,         # The detected hand landmarks
                mp_hands.HAND_CONNECTIONS # The connections between landmarks
            )

    # Display the resulting frame in a window
    cv2.imshow('AirPredict - Hand Tracking', frame)

    # Wait for 1ms and check if the 'q' key is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
hands.close() # NEW: Release the MediaPipe hands resources