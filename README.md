# ✋Hand Gesture Recognition using CNN+OpenCV

📌 **Project Overview**

This project implements a real-time hand gesture recognition system using a Convolutional Neural Network (CNN) trained on the [LeapGestRecog Dataset](https://www.kaggle.com/datasets/gti-upm/leapgestrecog).

The trained model is integrated with OpenCV to detect gestures live from a webcam.

The workflow includes:

- Image loading & preprocessing

- CNN model training on gesture images

- Model evaluation (accuracy, classification report, confusion matrix)

- Real-time gesture detection using webcam

## 📊 Dataset

- Source:

[LeapGestRecog Dataset](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)

- Folder Structure:

leapGestRecog/

00/

01/

...

09/

-Classes (subset used):

- ✊ Fist

- 👌 OK

- 👍 Thumbs Up

- 🖐 Palm

- Image Preprocessing:

- Grayscale conversion

- Resized to 64 × 64

- Normalized to \[0,1]

## 🧮 Model Features

- Architecture: CNN (3 convolutional layers + dense layers)

- Input size: 64 × 64 (grayscale)

- Output labels: 4 gestures (fist, palm, thumbs\_up, ok)

- Training:

- Optimizer: Adam

- Loss: Categorical Crossentropy

- Epochs: 20

- Batch size: 32

## ⚙️ Requirements

- Python 3.x

- Libraries:

- tensorflow

- opencv-python

- numpy

- matplotlib

- scikit-learn

- seaborn

Install via:

pip install -r requirements.txt

 Methodology

1️⃣ Data Loading

Load images from dataset folders

Assign labels based on folder names

2️⃣ Preprocessing

Convert to grayscale

Resize to 64×64

Normalize pixel values

3️⃣ Model Training

Train CNN on gesture images

Save trained model as hand_gesture_model_trained.h5

4️⃣ Evaluation

Accuracy & loss curves

Confusion matrix

Classification report

5️⃣ Real-Time Detection

Capture webcam feed with OpenCV

Draw ROI on screen

Predict gestures inside ROI

Display gesture label

📈 Results

Accuracy: ~99% on test set (with 4 gesture classes)

Performance: Stable real-time detection at ~15–20 FPS

Visualization:

Correct predictions ✅ displayed above ROI

Unknown/misclassified gestures shown as ...

🔮 Future Improvements

Train on more gestures (expand class set)

Improve ROI detection using skin segmentation or Mediapipe

Use transfer learning (MobileNetV2, EfficientNet) for higher accuracy

Deploy with Flask/Streamlit for interactive demo

👤 Author=>

SUBHADIP KARMAKAR

Machine Learning Intern @ Skillcraft Technology
