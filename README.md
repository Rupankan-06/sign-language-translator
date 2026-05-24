# Sign Language Translator using CNN

Real-time hand gesture recognition system that classifies 
ASL signs using a Convolutional Neural Network.

## Results
- Test Accuracy: 93%+
- Signs Recognised: 24 words
- Model: Trained CNN saved as asl_model.pkl

## How It Works
1. Extract hand landmarks from images using MediaPipe
2. Train CNN classifier on landmark features
3. Real-time prediction via webcam

## Files
- `extract_landmarks.py` — extracts hand landmarks from dataset
- `train.py` — trains the CNN model
- `webcam_demo.py` — real-time sign recognition via webcam
- `asl_model.pkl` — pre-trained model
- `requirements.txt` — dependencies

## Tech Stack
Python, PyTorch, OpenCV, MediaPipe, scikit-learn, NumPy

## How to Run
pip install -r requirements.txt
python webcam_demo.py

## Note
Raw dataset not included. Run extract_landmarks.py 
to generate landmarks from source images.
