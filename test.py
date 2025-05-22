import cv2
import mediapipe as mp
import numpy as np
import math
import tensorflow as tf
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mediapipe_hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Initialize HandDetector
from cvzone.HandTrackingModule import HandDetector

detector = HandDetector(maxHands=1, detectionCon=0.7)

cap = cv2.VideoCapture(0)


class SimpleClassifier:
    def __init__(self, model_path, labels_path):
        try:
            with tf.keras.utils.custom_object_scope({
                'DepthwiseConv2D': lambda **kwargs: tf.keras.layers.DepthwiseConv2D(
                    **{k: v for k, v in kwargs.items() if k != 'groups'}
                )
            }):
                self.model = tf.keras.models.load_model(model_path, compile=False)

            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]

            print(f"Model loaded successfully with labels: {self.labels}")
            self.model_loaded = True

        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
            self.labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
                           "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
            print("Using fallback mode")

    def getPrediction(self, img, confidence_threshold=0.7):
        if not self.model_loaded:
            fake_pred = np.random.rand(len(self.labels))
            fake_pred = fake_pred / fake_pred.sum()
            return fake_pred, np.argmax(fake_pred), np.max(fake_pred)

        try:
            img = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            # Use the model in eager mode
            prediction = self.model.predict(img, verbose=0)[0]
            index = np.argmax(prediction)
            confidence = prediction[index]

            return prediction, index, confidence

        except Exception as e:
            print(f"Prediction error: {e}")
            return None, None, 0.0


# Initialize classifier
classifier = SimpleClassifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300


def draw_hand_landmarks(img, landmarks, connections):
    """Draw hand landmarks manually on the image"""
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


def transform_landmarks_to_crop(landmarks, x_start, y_start, crop_width, crop_height):
    """Transform landmarks from original image to cropped image coordinates"""
    transformed_landmarks = []
    for landmark in landmarks:
        # Calculate pixel coordinates in original image
        x_px = landmark.x
        y_px = landmark.y
        z_px = landmark.z

        # Transform to cropped image coordinates (normalized 0-1)
        x_crop = (x_px - x_start) / crop_width
        y_crop = (y_px - y_start) / crop_height

        # Create a new landmark with transformed coordinates
        new_landmark = type('obj', (object,), {
            'x': x_crop,
            'y': y_crop,
            'z': z_px
        })
        transformed_landmarks.append(new_landmark)

    return transformed_landmarks


def transform_landmarks_to_white_bg(landmarks, x_start, y_start, crop_w, crop_h, white_w, white_h, aspect_ratio, gap=0):
    """Transform landmarks from original image to white background coordinates"""
    # First transform to crop coordinates
    crop_landmarks = transform_landmarks_to_crop(landmarks, x_start, y_start, crop_w, crop_h)

    # Then transform to white background coordinates
    white_landmarks = []

    if aspect_ratio > 1:  # Tall hand
        k = white_h / crop_h
        w_cal = math.ceil(k * crop_w)
        w_gap = (white_w - w_cal) // 2

        for landmark in crop_landmarks:
            x_white = (landmark.x * w_cal + w_gap) / white_w
            y_white = landmark.y
            z_white = landmark.z

            new_landmark = type('obj', (object,), {
                'x': x_white,
                'y': y_white,
                'z': z_white
            })
            white_landmarks.append(new_landmark)
    else:  # Wide hand
        k = white_w / crop_w
        h_cal = math.ceil(k * crop_h)
        h_gap = (white_h - h_cal) // 2

        for landmark in crop_landmarks:
            x_white = landmark.x
            y_white = (landmark.y * h_cal + h_gap) / white_h
            z_white = landmark.z

            new_landmark = type('obj', (object,), {
                'x': x_white,
                'y': y_white,
                'z': z_white
            })
            white_landmarks.append(new_landmark)

    return white_landmarks


# Define dark orange color for text (R=255, G=140, B=0)
TEXT_COLOR = (0, 140, 255)  # Note: OpenCV uses BGR format
WORD_COLOR = (0, 200, 0)  # Green color for the formed word

# Variables for word formation
current_word = ""
last_detected_letter = None
letter_start_time = 0
letter_confirmed = False
stable_time_required = 3.0  # Time in seconds required for letter to be confirmed
last_hand_detected = False
no_hand_start_time = 0
space_time_required = 2.0  # Time in seconds with no hand to add a space
last_coordinates = None
movement_threshold = 50  # Pixel threshold to detect significant movement

while True:
    success, img = cap.read()
    if not success:
        print("Failed to get frame from camera")
        break

    # Create copies of the original image for different displays
    imgOutput = img.copy()

    # Process with MediaPipe for hand detection
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mediapipe_hands.process(img_rgb)

    # Process with CVZone HandDetector for bounding box
    hands, img_with_hands = detector.findHands(img, draw=False)

    # Initialize empty placeholder images
    imgCrop = np.zeros((100, 100, 3), np.uint8)
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

    # Check if hand is detected
    hand_detected = hands and results.multi_hand_landmarks

    # If hand was detected before but not now, start timing for possible space
    if last_hand_detected and not hand_detected:
        no_hand_start_time = time.time()
    # If hand wasn't detected before but is now, reset letter timing
    elif not last_hand_detected and hand_detected:
        letter_start_time = time.time()
        letter_confirmed = False

    # Check for space condition (no hand detected for space_time_required seconds)
    if not hand_detected and last_hand_detected:
        time_without_hand = time.time() - no_hand_start_time
        if time_without_hand >= space_time_required:
            current_word += " "
            last_hand_detected = False

    if hand_detected:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Check for significant movement
        hand_center = (x + w // 2, y + h // 2)
        if last_coordinates is not None:
            distance = math.sqrt((hand_center[0] - last_coordinates[0]) ** 2 +
                                 (hand_center[1] - last_coordinates[1]) ** 2)

            if distance > movement_threshold:
                # Significant movement detected, reset letter timing
                letter_start_time = time.time()
                letter_confirmed = False

        last_coordinates = hand_center

        # Get hand crop with boundary checks
        y_start, y_end = max(0, y - offset), min(img.shape[0], y + h + offset)
        x_start, x_end = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y_start:y_end, x_start:x_end].copy()

        if imgCrop.size == 0:
            imgCrop = np.zeros((100, 100, 3), np.uint8)
            continue

        # Get pixel-based landmarks from original image
        hand_landmarks = results.multi_hand_landmarks[0].landmark
        orig_landmarks = []

        for i, landmark in enumerate(hand_landmarks):
            # Convert normalized coordinates to absolute pixel coordinates
            x_px = landmark.x * img.shape[1]
            y_px = landmark.y * img.shape[0]
            z_px = landmark.z

            # Store as objects with x, y, z properties
            orig_landmarks.append(type('obj', (object,), {
                'x': x_px,
                'y': y_px,
                'z': z_px
            }))

        # Transform landmarks to crop coordinates
        crop_landmarks = transform_landmarks_to_crop(
            orig_landmarks,
            x_start,
            y_start,
            imgCrop.shape[1],
            imgCrop.shape[0]
        )

        # Create white background and center the hand
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        aspectRatio = h / max(w, 1)

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wGap + wCal] = imgResize

            # Transform landmarks to white background
            white_landmarks = transform_landmarks_to_white_bg(
                orig_landmarks,
                x_start,
                y_start,
                imgCrop.shape[1],
                imgCrop.shape[0],
                imgSize,
                imgSize,
                aspectRatio,
                wGap
            )
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hGap + hCal, :] = imgResize

            # Transform landmarks to white background
            white_landmarks = transform_landmarks_to_white_bg(
                orig_landmarks,
                x_start,
                y_start,
                imgCrop.shape[1],
                imgCrop.shape[0],
                imgSize,
                imgSize,
                aspectRatio,
                hGap
            )

        # Draw landmarks on both white background AND cropped image
        imgWhite = draw_hand_landmarks(imgWhite, white_landmarks, mp_hands.HAND_CONNECTIONS)
        imgCrop = draw_hand_landmarks(imgCrop, crop_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get prediction
        prediction, index, confidence = classifier.getPrediction(imgWhite)

        # Display prediction if confident
        if index is not None and confidence > 0.4:
            label = classifier.labels[index]
            # Extract just the letter without any index
            # Labels are in format '0 A', '1 B', etc. - split by space and take the second part
            letter = label.split(' ')[1] if isinstance(label, str) and ' ' in label else label
            prob_percentage = int(confidence * 100)

            # Display current letter being recognized (without index)
            cv2.putText(imgOutput, f"Sign: {letter}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

            # Check if the letter has been stable for the required time
            current_time = time.time()
            if not letter_confirmed:
                if last_detected_letter == letter:
                    if current_time - letter_start_time >= stable_time_required:
                        current_word += letter
                        letter_confirmed = True

                        # Draw a confirmation indicator
                        cv2.putText(imgOutput, "Letter added!",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Reset the timer if the detected letter changes
                    letter_start_time = current_time
                    last_detected_letter = letter

            # Show progress towards letter confirmation
            if not letter_confirmed:
                elapsed_time = current_time - letter_start_time
                progress = min(elapsed_time / stable_time_required, 1.0)
                bar_width = int(200 * progress)
                cv2.rectangle(imgOutput, (10, 70), (10 + bar_width, 90), (0, 255, 0), -1)
                cv2.rectangle(imgOutput, (10, 70), (210, 90), (0, 0, 0), 2)
                cv2.putText(imgOutput, f"Confirming: {letter}",
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    else:
        # Reset letter detection if no hand is detected
        last_detected_letter = None

    # Display the current word
    cv2.putText(imgOutput, f"Word: {current_word}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, WORD_COLOR, 2)

    # Controls display - positioned at bottom left
    h, w, _ = imgOutput.shape
    margin = 10
    line_height = 25
    
    # Starting y position (from bottom)
    y_pos = h - margin - line_height
    
    # Controls text
    cv2.putText(imgOutput, "Press ESC to exit", (margin, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y_pos -= line_height
    
    cv2.putText(imgOutput, "Press 'r' to reset word", (margin, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y_pos -= line_height
    
    cv2.putText(imgOutput, "Press 'd' to delete last letter", (margin, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y_pos -= line_height
    
    cv2.putText(imgOutput, "Hold sign for 3s to add letter", (margin, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y_pos -= line_height
    
    cv2.putText(imgOutput, "Controls:", (margin, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Update last_hand_detected for next iteration
    last_hand_detected = hand_detected

    # Display all windows
    cv2.imshow("Main Output", imgOutput)
    cv2.imshow("Hand Crop", imgCrop)
    cv2.imshow("White Background", imgWhite)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break
    elif key == ord('r'):  # Reset word
        current_word = ""
        letter_confirmed = False
        cv2.putText(imgOutput, "Word Reset!", (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif key == ord('d'):  # Delete last letter
        if current_word:
            current_word = current_word[:-1]  # Remove the last character
            letter_confirmed = False
            cv2.putText(imgOutput, "Letter Deleted!", (10, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cap.release()
cv2.destroyAllWindows()
mediapipe_hands.close()