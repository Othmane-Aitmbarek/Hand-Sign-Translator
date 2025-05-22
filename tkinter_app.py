import cv2
import mediapipe as mp
import numpy as np
import math
import tensorflow as tf
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mediapipe_hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Initialize HandDetector
from cvzone.HandTrackingModule import HandDetector
detector = HandDetector(maxHands=1, detectionCon=0.7)

# Define colors
TEXT_COLOR = (0, 140, 255)  # Note: OpenCV uses BGR format
WORD_COLOR = (0, 200, 0)  # Green color for the formed word
BG_COLOR = "#E0E0E0"  # Light gray background color for Tkinter

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

    if aspect_ratio > 1:  # width > height (wide hand)
        # Calculate scaling factor
        k = white_w / crop_w
        h_cal = math.ceil(k * crop_h)
        h_gap = (white_h - h_cal) // 2 + gap

        for landmark in crop_landmarks:
            x_white = landmark.x  # Keep x as is since we're scaling to full width
            y_white = (landmark.y * h_cal + h_gap) / white_h
            z_white = landmark.z

            new_landmark = type('obj', (object,), {
                'x': x_white,
                'y': y_white,
                'z': z_white
            })
            white_landmarks.append(new_landmark)
    else:  # height >= width (tall hand)
        # Calculate scaling factor
        k = white_h / crop_h
        w_cal = math.ceil(k * crop_w)
        w_gap = (white_w - w_cal) // 2 + gap

        for landmark in crop_landmarks:
            x_white = (landmark.x * w_cal + w_gap) / white_w
            y_white = landmark.y  # Keep y as is since we're scaling to full height
            z_white = landmark.z

            new_landmark = type('obj', (object,), {
                'x': x_white,
                'y': y_white,
                'z': z_white
            })
            white_landmarks.append(new_landmark)
    
    return white_landmarks


class HandSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Sign Detection")
        self.root.configure(bg=BG_COLOR)
        
        # Initialize classifier
        self.classifier = SimpleClassifier("Model/keras_model.h5", "Model/labels.txt")
        self.offset = 20
        self.imgSize = 300
        
        # Variables for word formation
        self.current_word = ""
        self.last_detected_letter = None
        self.letter_start_time = 0
        self.letter_confirmed = False
        self.stable_time_required = 3.0  # Time in seconds required for letter to be confirmed
        self.last_hand_detected = False
        self.no_hand_start_time = 0
        self.space_time_required = 2.0  # Time in seconds with no hand to add a space
        self.last_coordinates = None
        self.movement_threshold = 50  # Pixel threshold to detect significant movement
        
        # Create the main frame
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a frame for the video feeds
        self.video_frame = ttk.Frame(self.main_frame)
        self.video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Create labels for the video feeds (without white background)
        self.main_video_label = ttk.Label(self.video_frame)
        self.main_video_label.grid(row=0, column=0, padx=5, pady=5)
        
        self.crop_video_label = ttk.Label(self.video_frame)
        self.crop_video_label.grid(row=0, column=1, padx=5, pady=5)
        
        # Create a frame for the word display
        self.word_frame = ttk.Frame(self.main_frame)
        self.word_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        # Create a label for the word display
        self.word_label = ttk.Label(self.word_frame, text="Word: ", font=("Helvetica", 16))
        self.word_label.pack(side=tk.LEFT, padx=5)
        
        # Create a frame for the confidence percentages
        self.confidence_frame = ttk.Frame(self.main_frame)
        self.confidence_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # Create a label for the confidence percentages title
        ttk.Label(self.confidence_frame, text="Pourcentages de confiance:", font=("Helvetica", 12, "bold")).pack(anchor=tk.W, padx=5)
        
        # Create labels for the top 5 predictions
        self.confidence_labels = []
        for i in range(5):  # Show top 5 predictions
            label = ttk.Label(self.confidence_frame, text="", font=("Helvetica", 10))
            label.pack(anchor=tk.W, padx=20)
            self.confidence_labels.append(label)
        
        # Create a frame for controls
        self.controls_frame = ttk.Frame(self.main_frame, style="Controls.TFrame")
        self.controls_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        # Create labels for controls
        ttk.Label(self.controls_frame, text="Controls:", font=("Helvetica", 12, "bold")).pack(anchor=tk.W, padx=5)
        ttk.Label(self.controls_frame, text="• Hold sign for 3s to add letter").pack(anchor=tk.W, padx=20)
        ttk.Label(self.controls_frame, text="• Press 'd' to delete last letter").pack(anchor=tk.W, padx=20)
        ttk.Label(self.controls_frame, text="• Press 'r' to reset word").pack(anchor=tk.W, padx=20)
        ttk.Label(self.controls_frame, text="• Press ESC to exit").pack(anchor=tk.W, padx=20)
        
        # Create buttons for controls
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        self.reset_button = ttk.Button(self.button_frame, text="Reset Word", command=self.reset_word)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        self.delete_button = ttk.Button(self.button_frame, text="Delete Last Letter", command=self.delete_last_letter)
        self.delete_button.pack(side=tk.LEFT, padx=5)
        
        self.exit_button = ttk.Button(self.button_frame, text="Exit", command=self.exit_app)
        self.exit_button.pack(side=tk.RIGHT, padx=5)
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
        # Bind keyboard shortcuts
        self.root.bind('<Escape>', lambda e: self.exit_app())
        self.root.bind('r', lambda e: self.reset_word())
        self.root.bind('d', lambda e: self.delete_last_letter())
        
        # Start the video loop
        self.update()
    
    def update(self):
        success, img = self.cap.read()
        if not success:
            print("Failed to get frame from camera")
            return
        
        # Create copies of the original image for different displays
        imgOutput = img.copy()
        
        # Process with MediaPipe for hand detection
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = mediapipe_hands.process(img_rgb)
        
        # Process with CVZone HandDetector for bounding box
        hands, img_with_hands = detector.findHands(img, draw=False)
        
        # Initialize empty placeholder images
        imgCrop = np.zeros((100, 100, 3), np.uint8)
        imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
        
        # Check if hand is detected
        hand_detected = hands and results.multi_hand_landmarks
        
        # If hand was detected before but not now, start timing for possible space
        if self.last_hand_detected and not hand_detected:
            self.no_hand_start_time = time.time()
        # If hand wasn't detected before but is now, reset letter timing
        elif not self.last_hand_detected and hand_detected:
            self.letter_start_time = time.time()
            self.letter_confirmed = False
        
        # Check for space condition (no hand detected for space_time_required seconds)
        if not hand_detected and self.last_hand_detected:
            time_without_hand = time.time() - self.no_hand_start_time
            if time_without_hand >= self.space_time_required:
                self.current_word += " "
                self.last_hand_detected = False
        
        # Process hand landmarks if detected
        if hand_detected:
            # Get hand information from CVZone
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            # Ensure the bounding box is within the image boundaries
            x_start = max(0, x - self.offset)
            y_start = max(0, y - self.offset)
            x_end = min(img.shape[1], x + w + self.offset)
            y_end = min(img.shape[0], y + h + self.offset)
            
            # Crop the hand region
            imgCrop = img[y_start:y_end, x_start:x_end]
            
            # Check if crop is valid
            if imgCrop.size == 0 or imgCrop.shape[0] == 0 or imgCrop.shape[1] == 0:
                # Skip this frame if crop is invalid
                pass
            else:
                # Calculate aspect ratio for the white background
                aspectRatio = imgCrop.shape[1] / imgCrop.shape[0]
                
                # Prepare the white background image properly
                if aspectRatio > 1:  # width > height
                    # Calculate new dimensions
                    wCal = self.imgSize
                    hCal = int(self.imgSize / aspectRatio)
                    # Resize the cropped image
                    imgResize = cv2.resize(imgCrop, (wCal, hCal))
                    # Calculate vertical gap
                    hGap = (self.imgSize - hCal) // 2
                    # Place the resized image on the white background
                    imgWhite[hGap:hGap + hCal, :] = imgResize
                else:  # height >= width
                    # Calculate new dimensions
                    wCal = int(self.imgSize * aspectRatio)
                    hCal = self.imgSize
                    # Resize the cropped image
                    imgResize = cv2.resize(imgCrop, (wCal, hCal))
                    # Calculate horizontal gap
                    wGap = (self.imgSize - wCal) // 2
                    # Place the resized image on the white background
                    imgWhite[:, wGap:wGap + wCal] = imgResize
                    hGap = 0
                
                # Get landmarks from MediaPipe
                orig_landmarks = results.multi_hand_landmarks[0].landmark
                
                # Transform landmarks to cropped image coordinates
                crop_landmarks = transform_landmarks_to_crop(
                    orig_landmarks,
                    x_start / img.shape[1],
                    y_start / img.shape[0],
                    (x_end - x_start) / img.shape[1],
                    (y_end - y_start) / img.shape[0]
                )
                
                # Transform landmarks to white background coordinates
                white_landmarks = transform_landmarks_to_white_bg(
                    orig_landmarks,
                    x_start / img.shape[1],
                    y_start / img.shape[0],
                    (x_end - x_start) / img.shape[1],
                    (y_end - y_start) / img.shape[0],
                    self.imgSize,
                    self.imgSize,
                    aspectRatio,
                    hGap if aspectRatio > 1 else wGap
                )
                
                # Draw landmarks on both white background AND cropped image
                imgWhite = draw_hand_landmarks(imgWhite, white_landmarks, mp_hands.HAND_CONNECTIONS)
                imgCrop = draw_hand_landmarks(imgCrop, crop_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get prediction
                prediction, index, confidence = self.classifier.getPrediction(imgWhite)
                
                # Update confidence percentage labels
                self.update_confidence_labels(prediction)
                
                # Display prediction if confident
                if index is not None and confidence > 0.4:
                    label = self.classifier.labels[index]
                    # Extract just the letter without any index
                    # Labels are in format '0 A', '1 B', etc. - split by space and take the second part
                    letter = label.split(' ')[1] if isinstance(label, str) and ' ' in label else label
                    prob_percentage = int(confidence * 100)
                    
                    # Display current letter being recognized (without index)
                    cv2.putText(imgOutput, f"Sign: {letter}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
                    
                    # Check if the letter has been stable for the required time
                    current_time = time.time()
                    if not self.letter_confirmed:
                        if self.last_detected_letter == letter:
                            if current_time - self.letter_start_time >= self.stable_time_required:
                                self.current_word += letter
                                self.letter_confirmed = True
                                
                                # Draw a confirmation indicator
                                cv2.putText(imgOutput, "Letter added!",
                                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            # Reset the timer if the detected letter changes
                            self.letter_start_time = current_time
                            self.last_detected_letter = letter
                    
                    # Show progress towards letter confirmation
                    if not self.letter_confirmed:
                        elapsed_time = current_time - self.letter_start_time
                        progress = min(elapsed_time / self.stable_time_required, 1.0)
                        bar_width = int(200 * progress)
                        cv2.rectangle(imgOutput, (10, 70), (10 + bar_width, 90), (0, 255, 0), -1)
                        cv2.rectangle(imgOutput, (10, 70), (210, 90), (0, 0, 0), 2)
                        cv2.putText(imgOutput, f"Confirming: {letter}",
                                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
        else:
            # Reset letter detection if no hand is detected
            self.last_detected_letter = None
        
        # Display the current word
        cv2.putText(imgOutput, f"Word: {self.current_word}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, WORD_COLOR, 2)
        
        # Update the word label in the Tkinter UI
        self.word_label.config(text=f"Word: {self.current_word}")
        
        # Update last_hand_detected for next iteration
        self.last_hand_detected = hand_detected
        
        # Convert the OpenCV images to PIL format for Tkinter
        imgOutput_rgb = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
        imgOutput_pil = Image.fromarray(imgOutput_rgb)
        imgOutput_tk = ImageTk.PhotoImage(image=imgOutput_pil)
        self.main_video_label.config(image=imgOutput_tk)
        self.main_video_label.image = imgOutput_tk
        
        # Display the crop image if it's valid
        if imgCrop.size > 0 and imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            imgCrop_rgb = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2RGB)
            imgCrop_pil = Image.fromarray(imgCrop_rgb)
            imgCrop_tk = ImageTk.PhotoImage(image=imgCrop_pil)
            self.crop_video_label.config(image=imgCrop_tk)
            self.crop_video_label.image = imgCrop_tk
        
        # White background image is still used for prediction but not displayed
        
        # Schedule the next update
        self.root.after(10, self.update)
    
    def reset_word(self):
        self.current_word = ""
        self.letter_confirmed = False
        self.word_label.config(text=f"Word: {self.current_word}")
    
    def delete_last_letter(self):
        if self.current_word:
            self.current_word = self.current_word[:-1]
            self.letter_confirmed = False
            self.word_label.config(text=f"Word: {self.current_word}")
    
    def update_confidence_labels(self, prediction):
        if prediction is None:
            # Clear all labels if no prediction
            for label in self.confidence_labels:
                label.config(text="")
            return
            
        # Get the indices of the top 5 predictions
        top_indices = np.argsort(prediction)[-5:]
        top_indices = top_indices[::-1]  # Reverse to get descending order
        
        # Update the labels with the letter and percentage
        for i, label_widget in enumerate(self.confidence_labels):
            if i < len(top_indices):
                idx = top_indices[i]
                letter_label = self.classifier.labels[idx]
                # Extract just the letter without any index
                letter = letter_label.split(' ')[1] if isinstance(letter_label, str) and ' ' in letter_label else letter_label
                percentage = int(prediction[idx] * 100)
                label_widget.config(text=f"{letter}: {percentage}%")
            else:
                label_widget.config(text="")
    
    def exit_app(self):
        if self.cap is not None:
            self.cap.release()
        mediapipe_hands.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x700")
    
    # Apply a theme
    style = ttk.Style()
    style.configure("TFrame", background=BG_COLOR)
    style.configure("Controls.TFrame", background=BG_COLOR)
    style.configure("TLabel", background=BG_COLOR)
    style.configure("TButton", background=BG_COLOR, font=("Helvetica", 10))
    
    app = HandSignApp(root)
    root.mainloop()
