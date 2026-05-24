import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import joblib

# Load model, scaler, label encoder
model = joblib.load("asl_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Load MediaPipe
MODEL_PATH = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Start webcam
cap = cv2.VideoCapture(0)
print("Webcam started! Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame so it feels like a mirror
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    if len(result.hand_landmarks) > 0:
        # Extract landmarks
        row = []
        for lm in result.hand_landmarks[0]:
            row.extend([lm.x, lm.y, lm.z])

        # Predict
        X = scaler.transform([row])
        pred = model.predict(X)[0]
        letter = le.inverse_transform([pred])[0]
        confidence = max(model.predict_proba(X)[0]) * 100

        # Display
        cv2.putText(frame, f"{letter} ({confidence:.1f}%)",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    2.5, (0, 255, 0), 4)
        cv2.putText(frame, "Hand Detected",
                    (30, 130), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No hand detected",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

    cv2.imshow("ASL Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()