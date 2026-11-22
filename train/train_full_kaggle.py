"""
Train SignSpeak MLP on FULL Kaggle Sign Language Gesture Images Dataset
=======================================================================
Dataset: ahmedkhanak1995/sign-language-gesture-images-dataset
Location: C:\\Users\\shrey\\.cache\\kagglehub\\datasets\\ahmedkhanak1995\\sign-language-gesture-images-dataset\\versions\\1\\Gesture Image Data

Uses ALL available images per class (up to 1500).
Bigger MLP: 63 -> 512 -> 256 -> 128 -> 26
More augmentation: 8x jitter + mirroring
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
SAMPLES_PER_CLASS = 1500  # Use ALL available images

print(f"=== SignSpeak Full Retrain: {len(CLASSES)} classes, up to {SAMPLES_PER_CLASS} images each ===", flush=True)
print(f"Dataset path: {KAGGLE_PATH}", flush=True)

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

print(f"\n--- 1. Extracting MediaPipe Landmarks (ALL images) ---", flush=True)

raw_landmarks_by_class = {}
t0 = time.time()
total_detected = 0
total_attempted = 0

for letter in CLASSES:
    folder = os.path.join(KAGGLE_PATH, letter)
    if not os.path.isdir(folder):
        print(f"  Warning: folder for '{letter}' not found!", flush=True)
        continue

    files = sorted(os.listdir(folder))[:SAMPLES_PER_CLASS]

    extracted = []
    for f in files:
        if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(folder, f)
        img = cv2.imread(img_path)
        if img is None:
            continue
        total_attempted += 1

        # Pad image to give MediaPipe palm detector space
        padded = cv2.copyMakeBorder(img, 40, 80, 60, 60, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        up = cv2.resize(padded, (256, 256))
        rgb = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)

        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            lm_obj = res.multi_hand_landmarks[0]
            pts = np.array([[p.x, p.y, p.z] for p in lm_obj.landmark], dtype=np.float32)
            norm_vec = canonical_normalize(pts)
            extracted.append(norm_vec)
            total_detected += 1

    raw_landmarks_by_class[letter] = extracted
    det_rate = len(extracted) / max(len(files), 1) * 100
    elapsed = time.time() - t0
    print(f"  {letter}: {len(extracted)}/{len(files)} detected ({det_rate:.0f}%) [{elapsed:.0f}s]", flush=True)

hands.close()

print(f"\nTotal: {total_detected}/{total_attempted} images detected ({total_detected/max(total_attempted,1)*100:.1f}%)", flush=True)

print(f"\n--- 2. Building Augmented Dataset ---", flush=True)

X_list = []
y_list = []
centroids = {}
rng = np.random.RandomState(42)

JITTER_COPIES = 8   # 8 jitter variants per sample
JITTER_SIGMA  = 0.012

for cid, letter in enumerate(CLASSES):
    samples = raw_landmarks_by_class.get(letter, [])
    if not samples:
        print(f"  ERROR: No samples for class {letter}!")
        continue

    centroid = np.mean(samples, axis=0)
    centroids[letter] = centroid.tolist()

    for vec in samples:
        # Original sample
        X_list.append(vec)
        y_list.append(cid)

        # Jitter augmentation
        for _ in range(JITTER_COPIES):
            jitter = rng.normal(0, JITTER_SIGMA, vec.shape).astype(np.float32)
            X_list.append(vec + jitter)
            y_list.append(cid)

        # Mirror (flip X axis)
        pts = vec.reshape(21, 3).copy()
        pts[:, 0] = -pts[:, 0]
        m_vec = pts.flatten()
        X_list.append(m_vec)
        y_list.append(cid)

        # Jitter on mirror
        for _ in range(JITTER_COPIES):
            jitter = rng.normal(0, JITTER_SIGMA, m_vec.shape).astype(np.float32)
            X_list.append(m_vec + jitter)
            y_list.append(cid)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)

samples_per_class_avg = len(X) // len(CLASSES)
print(f"Dataset built: {X.shape[0]:,} total samples (~{samples_per_class_avg:,} per class)", flush=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}", flush=True)

print("\n--- 3. Training MLP (63->512->256->128->26) ---", flush=True)
t_train = time.time()

mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation='relu',
    solver='adam',
    batch_size=256,
    learning_rate_init=0.001,
    max_iter=200,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.08,
    n_iter_no_change=12,
    verbose=True
)

mlp.fit(X_train, y_train)
t_elapsed = time.time() - t_train
print(f"\nTraining done in {t_elapsed:.1f}s", flush=True)

test_preds = mlp.predict(X_test)
acc = accuracy_score(y_test, test_preds)

print(f"\n==========================================", flush=True)
print(f"  Test Accuracy: {acc * 100:.2f}%", flush=True)
print(f"==========================================\n", flush=True)

print("--- Centroid Verification (should be 26/26) ---")
centroid_acc = 0
for letter, c_vec in centroids.items():
    pred_cid = mlp.predict([c_vec])[0]
    pred_letter = CLASSES[pred_cid]
    prob = np.max(mlp.predict_proba([c_vec])[0])
    match = '✓' if pred_letter == letter else '✗'
    if pred_letter == letter:
        centroid_acc += 1
    print(f"  {match} {letter} -> {pred_letter} ({prob * 100:.1f}%)")

print(f"\nCentroid Accuracy: {centroid_acc}/{len(CLASSES)} ({centroid_acc/len(CLASSES)*100:.1f}%)\n", flush=True)

print("--- 4. Exporting artifacts to web/model/ ---", flush=True)

os.makedirs("web/model", exist_ok=True)

with open("web/model/labels.json", "w", encoding="utf-8") as f:
    json.dump(CLASSES, f)
print("  ✓ labels.json")

landmarks_data = {"classes": CLASSES, "centroids": centroids}
with open("web/model/landmarks.json", "w", encoding="utf-8") as f:
    json.dump(landmarks_data, f)
print("  ✓ landmarks.json")

model_export = {
    "input_dim": 63,
    "layer_sizes": [512, 256, 128, len(CLASSES)],
    "weights": [coef.tolist() for coef in mlp.coefs_],
    "biases": [intercept.tolist() for intercept in mlp.intercepts_],
    "classes": CLASSES,
    "accuracy": float(acc)
}
with open("web/model/model.json", "w", encoding="utf-8") as f:
    json.dump(model_export, f)
sz = os.path.getsize("web/model/model.json") // 1024
print(f"  ✓ model.json ({sz} KB)")

print(f"\n=== DONE! Full retrain complete. Accuracy: {acc * 100:.2f}% ===\n", flush=True)
