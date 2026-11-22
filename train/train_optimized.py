"""
SignSpeak - Optimized High-Accuracy Training Pipeline
=====================================================
Trains a 4-layer MLP (63 -> 256 -> 128 -> 64 -> 43) with:
- Wrist-relative & scale-invariant landmark normalization
- Left/right hand mirroring
- Gaussian noise jitter augmentation (std=0.008) for hand trembling robustness
- Exports generic multi-layer model.json, labels.json, and landmarks.json
"""

import os
import sys
import json
import pickle
import numpy as np

os.environ['KERAS_HOME'] = os.path.abspath('.keras')
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def normalize_landmarks(landmarks_21x3):
    wrist = landmarks_21x3[0]
    relative = landmarks_21x3 - wrist
    distances = np.linalg.norm(relative, axis=1)
    max_dist = np.max(distances)
    if max_dist < 1e-6:
        max_dist = 1.0
    return relative / max_dist

print("Loading dataset from cnn_dataset.pickle...", flush=True)
with open("cnn_dataset.pickle", "rb") as f:
    data = pickle.load(f)

features = data["features"][:, 0, :, :] # (16425, 21, 3)
labels = data["labels"] # (16425,)
label_map = data["label_map"]
id_to_class = {v: k for k, v in label_map.items()}
classes = [id_to_class[i] for i in range(len(label_map))]

print(f"Loaded {len(features)} original samples across {len(classes)} classes.", flush=True)

X_list = []
y_list = []
unmirrored_by_class = {cls: [] for cls in classes}

np.random.seed(42)

for i in range(len(features)):
    cls_name = id_to_class[labels[i]]
    norm_lm = normalize_landmarks(features[i].astype(np.float32))
    feat = norm_lm.flatten() # 63
    
    # 1. Original sample
    X_list.append(feat)
    y_list.append(labels[i])
    unmirrored_by_class[cls_name].append(feat)
    
    # 2. Mirrored sample (flip X)
    mirrored = norm_lm.copy()
    mirrored[:, 0] = -mirrored[:, 0]
    X_list.append(mirrored.flatten())
    y_list.append(labels[i])
    
    # 3. Small jitter noise for robustness to trembling
    noisy = norm_lm.copy() + np.random.normal(0, 0.008, norm_lm.shape).astype(np.float32)
    X_list.append(noisy.flatten())
    y_list.append(labels[i])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)

print(f"Total augmented dataset: X = {X.shape}, y = {y.shape}", flush=True)

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training 4-layer MLP (63 -> 256 -> 128 -> 64 -> {len(classes)})...", flush=True)
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    batch_size=256,
    max_iter=80,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=8
)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n=======================================================", flush=True)
print(f"  >>> OPTIMIZED TEST ACCURACY: {accuracy * 100:.2f}% <<<", flush=True)
print(f"=======================================================\n", flush=True)

# Export artifacts to web/model/
output_dir = "web/model"
os.makedirs(output_dir, exist_ok=True)

# 1. labels.json
labels_path = os.path.join(output_dir, "labels.json")
with open(labels_path, "w", encoding="utf-8") as f:
    json.dump(classes, f, indent=2)

# 2. model.json (generic multi-layer MLP)
weights_list = [w.tolist() for w in mlp.coefs_]
biases_list = [b.tolist() for b in mlp.intercepts_]

model_data = {
    "format": "signspeak_mlp_v2",
    "accuracy": float(accuracy),
    "input_dim": int(mlp.coefs_[0].shape[0]),
    "layer_sizes": [int(w.shape[1]) for w in mlp.coefs_],
    "activation": "relu",
    "output_activation": "softmax",
    "weights": weights_list,
    "biases": biases_list,
    "classes": classes,
    "top_k": 3
}

model_path = os.path.join(output_dir, "model.json")
with open(model_path, "w", encoding="utf-8") as f:
    json.dump(model_data, f)
print(f"Saved optimized model.json: {os.path.getsize(model_path)/1024:.1f} KB", flush=True)

# 3. landmarks.json (true un-mirrored centroids)
centroids = {}
for cls_name, vecs in unmirrored_by_class.items():
    if len(vecs) > 0:
        centroids[cls_name] = np.mean(vecs, axis=0).tolist()

knn_data = {
    "classes": classes,
    "centroids": centroids
}
landmarks_path = os.path.join(output_dir, "landmarks.json")
with open(landmarks_path, "w", encoding="utf-8") as f:
    json.dump(knn_data, f)
print(f"Saved updated landmarks.json: {os.path.getsize(landmarks_path)/1024:.1f} KB", flush=True)

print("Model training and export complete!", flush=True)
