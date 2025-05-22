# Hand Sign Language Translator 🤟

Plus de 70 millions de personnes sourdes dans le monde utilisent les langues des signes pour communiquer. La langue des signes leur permet d’apprendre, de travailler, d’accéder aux services et d’être intégrées dans les communautés.
Il est difficile de faire apprendre la langue des signes à tout le monde afin de garantir que les personnes en situation de handicap puissent jouir de leurs droits sur un pied d’égalité avec les autres.
Ainsi, l’objectif est de développer une interface homme-machine (IHM) conviviale où l’ordinateur comprend la langue des signes américaine. Ce projet aidera les personnes sourdes et muettes en facilitant leur quotidien.

**Objectif** : Créer un logiciel informatique et entraîner un modèle qui prend en entrée une image en temps réel d’un geste de la langue des signes américaine et affiche en sortie la traduction en texte.

**Scope** : Ce système sera bénéfique à la fois pour les personnes sourdes/muettes et pour celles qui ne comprennent pas la langue des signes. Il suffit de réaliser les gestes de la langue des signes, et ce système identifiera ce que la personne essaie de dire, puis fournira la sortie sous forme de texte.

![image]((https://www.researchgate.net/publication/328396430/figure/fig1/AS:11431281391526962@1745332210693/The-26-letters-and-10-digits-of-American-Sign-Language-ASL.tif))

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
