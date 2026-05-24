import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd

MODEL_PATH = "hand_landmarker.task"
DATA_DIR = "data/raw/asl_alphabet_train/asl_alphabet_train"
data = []

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

for label in sorted(os.listdir(DATA_DIR)):
    label_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_path):
        continue

    count = 0
    images = os.listdir(label_path)

    for img_file in images:
        if count >= 200:
            break

        img_path = os.path.join(label_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = detector.detect(mp_image)
        except Exception as e:
            continue

        if len(result.hand_landmarks) > 0:
            row = []
            for lm in result.hand_landmarks[0]:
                row.extend([lm.x, lm.y, lm.z])
            row.append(label)
            data.append(row)
            count += 1

    print(f"{label} done - {count} samples")

cols = [f"{axis}{i}" for i in range(21) for axis in ["x","y","z"]] + ["label"]
pd.DataFrame(data, columns=cols).to_csv("data/landmarks.csv", index=False)
print(f"Done! {len(data)} total samples saved to landmarks.csv")