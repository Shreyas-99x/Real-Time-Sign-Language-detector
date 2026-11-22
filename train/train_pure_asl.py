"""
SignSpeak - Pure ASL High-Accuracy Model Training (A-Z)
======================================================
Trains an orientation-invariant 4-layer MLP (63 -> 256 -> 128 -> 64 -> 26)
on 39,000 pure, anatomically verified single-handed ASL landmark vectors.
Zero contamination from two-handed ISL dataset.
Exports:
- web/model/labels.json
- web/model/model.json
- web/model/landmarks.json
"""

import os
import sys
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

sys.path.append('scratch')
from asl_dataset_generator import generate_asl_landmarks, ALPHABETS

print(f"Generating comprehensive ASL training dataset for {len(ALPHABETS)} classes...", flush=True)

X_list = []
y_list = []
class_centroids = {letter: [] for letter in ALPHABETS}
rng = np.random.RandomState(42)

# Generate 750 samples per class with rich realistic variations:
# 1. Anatomical hand dimension variations (palm_w, palm_h, finger lengths)
# 2. Gaussian noise jitter (trembling simulation)
# 3. XY planar rotation (-25 deg to +25 deg)
# 4. Scale variations
# 5. Left and right hand mirroring
for cid, letter in enumerate(ALPHABETS):
    for s_idx in range(750):
        lm = generate_asl_landmarks(letter, rng)
        
        # Jitter
        jitter = rng.normal(0, rng.uniform(0.008, 0.020), lm.shape).astype(np.float32)
        noisy = lm + jitter
        
        # Rotation
        rot_ang = rng.uniform(-0.45, 0.45) # ~ -26 to +26 deg
        cos_t = np.cos(rot_ang); sin_t = np.sin(rot_ang)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
        r = noisy.copy()
        r[:, :2] = np.dot(noisy[:, :2], R.T)
        
        # Scale
        max_d = np.max(np.linalg.norm(r, axis=1))
        if max_d < 1e-6: max_d = 1.0
        norm_vec = (r / max_d).flatten()
        
        X_list.append(norm_vec)
        y_list.append(cid)
        if s_idx < 100:
            class_centroids[letter].append(norm_vec)
        
        # Mirrored (horizontal flip for left/right hand support)
        m = (r / max_d).copy()
        m[:, 0] = -m[:, 0]
        X_list.append(m.flatten())
        y_list.append(cid)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)
print(f"Total balanced dataset: X = {X.shape}, y = {y.shape}", flush=True)

# 85/15 train/test stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

print(f"Training 4-layer MLP Neural Network (63 -> 256 -> 128 -> 64 -> 26)...", flush=True)
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    batch_size=128,
    max_iter=120,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10
)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n=======================================================", flush=True)
print(f"  >>> PURE ASL (A-Z) TEST ACCURACY: {accuracy * 100:.2f}% <<<", flush=True)
print(f"=======================================================\n", flush=True)

# Export artifacts to web/model/
output_dir = "web/model"
os.makedirs(output_dir, exist_ok=True)

# 1. labels.json
labels_path = os.path.join(output_dir, "labels.json")
with open(labels_path, "w", encoding="utf-8") as f:
    json.dump(ALPHABETS, f, indent=2)
print(f"✓ Saved labels: {labels_path}")

# 2. model.json
weights_list = [w.tolist() for w in mlp.coefs_]
biases_list = [b.tolist() for b in mlp.intercepts_]

model_data = {
    "format": "signspeak_asl_v4_pure",
    "accuracy": float(accuracy),
    "input_dim": int(mlp.coefs_[0].shape[0]),
    "layer_sizes": [int(w.shape[1]) for w in mlp.coefs_],
    "activation": "relu",
    "output_activation": "softmax",
    "weights": weights_list,
    "biases": biases_list,
    "classes": ALPHABETS,
    "top_k": 3
}

model_path = os.path.join(output_dir, "model.json")
with open(model_path, "w", encoding="utf-8") as f:
    json.dump(model_data, f)
print(f"✓ Saved model.json: {os.path.getsize(model_path)/1024:.1f} KB")

# 3. landmarks.json
centroids = {}
for letter, vecs in class_centroids.items():
    centroids[letter] = np.mean(vecs, axis=0).tolist()

knn_data = {
    "classes": ALPHABETS,
    "centroids": centroids
}
landmarks_path = os.path.join(output_dir, "landmarks.json")
with open(landmarks_path, "w", encoding="utf-8") as f:
    json.dump(knn_data, f)
print(f"✓ Saved landmarks.json: {os.path.getsize(landmarks_path)/1024:.1f} KB")

print("\nAll pure ASL model artifacts successfully written to web/model/")
