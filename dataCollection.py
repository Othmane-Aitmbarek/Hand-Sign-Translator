import cv2
import mediapipe as mp
import numpy as np
import math
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
offset = 20
imgSize = 300
folder = "Data/A"
counter = 0


def draw_hand_landmarks(img, landmarks, connections):
    """Draw hand landmarks manually on the image with green lines and red dots"""
    h, w, c = img.shape
    landmarks_px = []

    # Convert normalized coordinates to pixel coordinates
    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        landmarks_px.append((x, y))

    # Draw connections with green lines
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(landmarks_px) and end_idx < len(landmarks_px):
            cv2.line(img, landmarks_px[start_idx], landmarks_px[end_idx], (0, 255, 0), 2)

    # Draw red dots at key landmarks
    for point in landmarks_px:
        cv2.circle(img, point, 4, (0, 0, 255), cv2.FILLED)

    return img


while True:
    success, img = cap.read()
    if not success:
        continue

    # Convert to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image for hand detection
    results = hands.process(img_rgb)

    # Draw the hand landmarks on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Use custom drawing function instead of MediaPipe's default
            img = draw_hand_landmarks(img, hand_landmarks.landmark, mp_hands.HAND_CONNECTIONS)

            # Get bounding box
            h, w, c = img.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Add offset to bounding box
            x = x_min - offset
            y = y_min - offset
            w = x_max - x_min + 2 * offset
            h = y_max - y_min + 2 * offset

            # Make sure crop area is valid
            y_start = max(0, y)
            y_end = min(img.shape[0], y + h)
            x_start = max(0, x)
            x_end = min(img.shape[1], x + w)

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y_start:y_end, x_start:x_end]

            # Check if the crop is valid
            if imgCrop.size == 0:
                continue

            aspectRatio = h / w if w > 0 else 1

            # Transform landmarks to white background coordinates
            whiteImgLandmarks = []

            if aspectRatio > 1:
                k = imgSize / h if h > 0 else 1
                wCal = math.ceil(k * w)
                if wCal > 0:
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize

                    # Transform landmarks for white image
                    for landmark in hand_landmarks.landmark:
                        # Calculate relative position in crop
                        rel_x = (landmark.x * img.shape[1] - x_start) / (x_end - x_start)
                        rel_y = (landmark.y * img.shape[0] - y_start) / (y_end - y_start)

                        # Transform to white image
                        white_x = rel_x * wCal + wGap
                        white_y = rel_y * imgSize

                        # Create new landmark object
                        new_landmark = type('obj', (object,), {
                            'x': white_x / imgSize,
                            'y': white_y / imgSize,
                            'z': landmark.z
                        })
                        whiteImgLandmarks.append(new_landmark)
            else:
                k = imgSize / w if w > 0 else 1
                hCal = math.ceil(k * h)
                if hCal > 0:
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                    # Transform landmarks for white image
                    for landmark in hand_landmarks.landmark:
                        # Calculate relative position in crop
                        rel_x = (landmark.x * img.shape[1] - x_start) / (x_end - x_start)
                        rel_y = (landmark.y * img.shape[0] - y_start) / (y_end - y_start)

                        # Transform to white image
                        white_x = rel_x * imgSize
                        white_y = rel_y * hCal + hGap

                        # Create new landmark object
                        new_landmark = type('obj', (object,), {
                            'x': white_x / imgSize,
                            'y': white_y / imgSize,
                            'z': landmark.z
                        })
                        whiteImgLandmarks.append(new_landmark)

            # Draw landmarks on the white image
            if whiteImgLandmarks:
                imgWhite = draw_hand_landmarks(imgWhite, whiteImgLandmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
    elif key == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
hands.close()