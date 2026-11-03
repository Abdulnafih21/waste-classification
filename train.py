import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

DATASET_DIR = r"C:\Users\abdul\Downloads\archive\balanced_waste_images"
MODEL_DIR = r"Waste-Classification-Using-Machine-Learning\models"
MODEL_PATH = os.path.join(MODEL_DIR, "waste_classifier_rf.joblib")
IMG_SIZE = 64

os.makedirs(MODEL_DIR, exist_ok=True)

classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
print(f"Found classes: {classes}")

X = []
y = []

print("Loading dataset images...")
for idx, class_name in enumerate(classes):
    class_folder = os.path.join(DATASET_DIR, class_name)
    count = 0
    for img_file in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_file)
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img.flatten())
            y.append(idx)
            count += 1
        except Exception:
            continue
    print(f"Loaded {count} images for class '{class_name}'")

X = np.array(X, dtype=np.float32) / 255.0
y = np.array(y)

print(f"Total samples: {len(y)}, Feature shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training Random Forest classifier (100 estimators)...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Validation Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=classes))

# Save model and metadata
model_bundle = {
    "model": clf,
    "classes": classes,
    "img_size": IMG_SIZE
}
joblib.dump(model_bundle, MODEL_PATH)
print(f"Model saved successfully to: {MODEL_PATH}")

