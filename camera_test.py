import cv2

# Try camera index 0
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera opened successfully! Press ESC to exit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    # Display the frame
    cv2.imshow('Camera Test', frame)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()