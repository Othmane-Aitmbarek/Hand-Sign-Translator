# Hand Sign Language Translator 

More than 70 million deaf people around the world use sign languages to communicate. Sign language allows them to learn, work, access services and be integrated into communities.
It is difficult to teach sign language to everyone to ensure that people with disabilities can enjoy their rights on an equal footing with others.
Thus, the objective is to develop a user-friendly human-machine interface (HMI) where the computer understands American Sign Language. This project will help deaf and mute people by making their daily lives easier.

**Objective**: Create computer software and train a model that takes as input a real-time image of an American Sign Language gesture and displays the text translation as output.

**Scope**: This system will be beneficial for both deaf/mute people and those who do not understand sign language. Simply perform sign language gestures, and this system will identify what the person is trying to say, then provide the output in text form.

## ✨ Features

- **Real-time hand gesture recognition** using webcam
- **High accuracy detection** with MediaPipe hand landmarks
- **Customizable gesture library** through Teachable Machine integration
- **User-friendly interface** with live video feed
- **Text translation output** for recognized gestures
- **Support for multiple hand signs** and gestures

## 🛠️ Technologies Used

- **Python 3.10+** - Core programming language
- **OpenCV** - Computer vision and image processing
- **MediaPipe** - Hand landmark detection and tracking
- **TensorFlow** - Machine learning model inference
- **NumPy** - Numerical computations and array operations
- **Math** - Mathematical calculations for hand pose analysis
- **Teachable Machine** - Custom gesture model training

## 📋 Prerequisites

Before running this project, make sure you have Python 3.10 or higher installed on your system.

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Othmane-Aitmbarek/Hand-Sign-Translator.git
   cd Hand-Sign-Translator
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Dependencies

```
opencv-python>=4.5.0
mediapipe>=0.8.0
tensorflow>=2.0.0
numpy>=1.21.0
```

Create a `requirements.txt` file with the above dependencies.

## 🎯 Usage

1. **Run the main application**
   ```bash
   python test.py
   ```

2. **Position your hand** in front of the camera
3. **Make hand gestures** - the application will detect and translate them in real-time
4. **Press 'esc'** to quit the application

## 🏗️ Project Structure

```
Hand-Sign-Translator/
│
├── Model/                  # Trained models directory
│   └── keras_model.h5      # Teachable Machine model
│   └── labels.txt          # Class labels
├── Data/                   # Data directory
│   ├── A/    
│       └── 0.jpg           
│       └── 1.jpg
│   ├── B/    
│       └── 0.jpg         
│       └── 1.jpg       
├── test.py                 # Application file
├── camera_test.py          # Camera Testing file
├── dataCollection.py       # Data Collection file
└── tkinter_app.py          # Main application file
```

## 🤖 How It Works

1. **Hand Detection**: MediaPipe identifies hand landmarks in real-time from the webcam feed
2. **Feature Extraction**: Key hand pose features are extracted and normalized
3. **Gesture Classification**: The trained Teachable Machine model classifies the gesture
4. **Translation**: Recognized gestures are translated to corresponding text/meaning
5. **Display**: Results are displayed in a user-friendly Tkinter GUI interface with confidence scores

## 🎓 Training Your Own Model

To train your own gesture recognition model:

1. Visit [Teachable Machine](https://teachablemachine.withgoogle.com/)
2. Choose "Image Project"
3. Create classes for different hand signs
4. Upload training images for each gesture class
5. Train the model and download it
6. Replace the model files in the `model/` directory

## 📊 Supported Gestures

Currently supported hand signs:
- A-Z Alphabet signs
- Custom gestures (can be added through training)


⭐ If you found this project helpful, please give it a star!
