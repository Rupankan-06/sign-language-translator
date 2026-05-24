import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load Data
df = pd.read_csv("data/landmarks.csv")
X = df.drop("label", axis=1).values
y = df["label"].values
print(f"Data loaded - {X.shape[0]} samples")

# 2. Encode Labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4. Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# 5. Build + Train
print("Training started...")
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    max_iter=50,
    verbose=True
)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"\nVal Accuracy: {acc*100:.2f}%")
print(classification_report(y_val, y_pred))

# 7. Save
joblib.dump(model, "asl_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")
print("Model saved!")