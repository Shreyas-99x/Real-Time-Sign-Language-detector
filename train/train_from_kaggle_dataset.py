"""
Train SignSpeak MLP on Kaggle Sign Language Gesture Images Dataset
==================================================================
Dataset: ahmedkhanak1995/sign-language-gesture-images-dataset
Location: C:\\Users\\shrey\\.cache\\kagglehub\\datasets\\ahmedkhanak1995\\sign-language-gesture-images-dataset\\versions\\1\\Gesture Image Data

Extracts real MediaPipe landmarks, applies canonical normalization, augments
with jitter and left/right mirroring, trains a 4-layer MLP, and exports:
- web/model/labels.json
- web/model/landmarks.json
- web/model/model.json
"""

import os
import sys
import json
import time
import cv2
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

KAGGLE_PATH = r"C:\Users\shrey\.cache\kagglehub\datasets\ahmedkhanak1995\sign-language-gesture-images-dataset\versions\1\Gesture Image Data"
CLASSES = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
SAMPLES_PER_CLASS = 150

print(f"--- 1. Extracting MediaPipe Landmarks from Kaggle Dataset ({len(CLASSES)} classes) ---", flush=True)

hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.05
)

def canonical_normalize(landmarks_21x3):
    wrist = landmarks_21x3[0]
    mid_mcp = landmarks_21x3[9]
    dx = mid_mcp[0] - wrist[0]
    dy = mid_mcp[1] - wrist[1]
    angle = np.arctan2(dy, dx)
    rot_angle = -np.pi / 2.0 - angle
    cos_t = np.cos(rot_angle)
    sin_t = np.sin(rot_angle)
    
    rel = landmarks_21x3 - wrist
    rot_x = rel[:, 0] * cos_t - rel[:, 1] * sin_t
    rot_y = rel[:, 0] * sin_t + rel[:, 1] * cos_t
    rot_z = rel[:, 2]
    
    rot = np.column_stack([rot_x, rot_y, rot_z])
    max_d = np.max(np.linalg.norm(rot, axis=1))
    if max_d < 1e-6:
        max_d = 1.0
    return (rot / max_d).flatten()

raw_landmarks_by_class = {}
t0 = time.time()

for letter in CLASSES:
    folder = os.path.join(KAGGLE_PATH, letter)
    if not os.path.isdir(folder):
        print(f"Warning: folder for {letter} not found!", flush=True)
        continue
    
    files = os.listdir(folder)
    # Take a balanced sample
    step = max(1, len(files) // SAMPLES_PER_CLASS)
    selected_files = files[::step][:SAMPLES_PER_CLASS]
    
    extracted = []
    for f in selected_files:
        img_path = os.path.join(folder, f)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Add padding to give palm detector breathing room
        padded = cv2.copyMakeBorder(img, 40, 80, 60, 60, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        up = cv2.resize(padded, (256, 256))
        rgb = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
        
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            lm_obj = res.multi_hand_landmarks[0]
            pts = np.array([[p.x, p.y, p.z] for p in lm_obj.landmark], dtype=np.float32)
            norm_vec = canonical_normalize(pts)
            extracted.append(norm_vec)
    
    raw_landmarks_by_class[letter] = extracted
    print(f"  Class {letter}: extracted {len(extracted)} valid samples in {time.time() - t0:.1f}s", flush=True)

print(f"\n--- 2. Building Dataset with Augmentation ---", flush=True)

X_list = []
y_list = []
centroids = {}
rng = np.random.RandomState(42)

for cid, letter in enumerate(CLASSES):
    samples = raw_landmarks_by_class.get(letter, [])
    if not samples:
        print(f"Error: No samples for class {letter}!")
        continue
    
    # Calculate real centroid
    centroid = np.mean(samples, axis=0)
    centroids[letter] = centroid.tolist()
    
    for vec in samples:
        # Original
        X_list.append(vec)
        y_list.append(cid)
        
        # Jitter variations
        for _ in range(5):
            jitter = rng.normal(0, 0.015, vec.shape).astype(np.float32)
            X_list.append(vec + jitter)
            y_list.append(cid)
            
        # Left/Right Mirrored
        pts = vec.reshape(21, 3).copy()
        pts[:, 0] = -pts[:, 0]
        m_vec = pts.flatten()
        X_list.append(m_vec)
        y_list.append(cid)
        for _ in range(5):
            jitter = rng.normal(0, 0.015, m_vec.shape).astype(np.float32)
            X_list.append(m_vec + jitter)
            y_list.append(cid)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)

print(f"Total Augmented Training Dataset: X={X.shape}, y={y.shape}", flush=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

print("\n--- 3. Training 4-Layer MLP Neural Network (63 -> 256 -> 128 -> 64 -> 26) ---", flush=True)
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    batch_size=128,
    max_iter=150,
    random_state=42,
    early_stopping=True,
    n_iter_no_change=10,
    verbose=True
)

mlp.fit(X_train, y_train)

test_preds = mlp.predict(X_test)
acc = accuracy_score(y_test, test_preds)
print(f"\n==========================================", flush=True)
print(f"✓ Test Accuracy on Kaggle ASL Dataset: {acc * 100:.2f}%", flush=True)
print(f"==========================================\n", flush=True)

# Test centroids directly
centroid_acc = 0
print("--- Centroid Verification ---")
for letter, c_vec in centroids.items():
    pred_cid = mlp.predict([c_vec])[0]
    pred_letter = CLASSES[pred_cid]
    prob = np.max(mlp.predict_proba([c_vec])[0])
    match = '✓' if pred_letter == letter else '✗'
    if pred_letter == letter:
        centroid_acc += 1
    print(f"  {match} {letter} -> {pred_letter} ({prob * 100:.1f}%)")

print(f"\nCentroid Accuracy: {centroid_acc}/{len(CLASSES)} ({centroid_acc/len(CLASSES)*100:.1f}%)\n")

print("--- 4. Exporting Web Model Artifacts ---", flush=True)

# Export labels.json
labels_path = "web/model/labels.json"
with open(labels_path, "w", encoding="utf-8") as f:
    json.dump(CLASSES, f)
print(f"✓ Exported {labels_path}")

# Export landmarks.json
landmarks_path = "web/model/landmarks.json"
landmarks_data = {
    "classes": CLASSES,
    "centroids": centroids
}
with open(landmarks_path, "w", encoding="utf-8") as f:
    json.dump(landmarks_data, f)
print(f"✓ Exported {landmarks_path}")

# Export model.json
model_path = "web/model/model.json"
model_export = {
    "input_dim": 63,
    "layer_sizes": [256, 128, 64, len(CLASSES)],
    "weights": [coef.tolist() for coef in mlp.coefs_],
    "biases": [intercept.tolist() for intercept in mlp.intercepts_],
    "classes": CLASSES,
    "accuracy": float(acc)
}
with open(model_path, "w", encoding="utf-8") as f:
    json.dump(model_export, f)
print(f"✓ Exported {model_path} ({os.path.getsize(model_path) // 1024} KB)")

print("\nModel trained successfully and exported to web/model/!")
