import cv2

# Initialize the webcam. 0 is the default camera.
cap = cv2.VideoCapture(0)

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

    # Display the resulting frame in a window
    cv2.imshow('AirPredict - Webcam Feed', frame)

    # Wait for 1ms and check if the 'q' key is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()