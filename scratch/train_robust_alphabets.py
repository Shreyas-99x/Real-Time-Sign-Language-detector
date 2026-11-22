import os
import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

ALPHABETS = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
print(f"Target classes ({len(ALPHABETS)}): {', '.join(ALPHABETS)}")

def rotate_to_canonical_upright(landmarks_21x3):
    """
    Aligns hand so vector from wrist (0) to middle MCP (9) points straight UP (angle = -90 deg: dx=0, dy=-1).
    """
    wrist = landmarks_21x3[0]
    rel = landmarks_21x3 - wrist
    mid_mcp = rel[9]
    angle = np.arctan2(mid_mcp[1], mid_mcp[0])
    rot_angle = -np.pi / 2 - angle
    
    cos_t = np.cos(rot_angle)
    sin_t = np.sin(rot_angle)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    
    rot = rel.copy()
    rot[:, :2] = np.dot(rel[:, :2], R.T)
    max_d = np.max(np.linalg.norm(rot, axis=1))
    if max_d < 1e-6: max_d = 1.0
    return rot / max_d

def rotate_xy(landmarks_21x3, angle_rad):
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    rot = landmarks_21x3.copy()
    rot[:, :2] = np.dot(landmarks_21x3[:, :2], R.T)
    return rot

# 1. Load dataset samples
with open("cnn_dataset.pickle", "rb") as f:
    data = pickle.load(f)

feats = data["features"][:, 0, :, :].astype(np.float32)
labels = data["labels"]
label_map = data["label_map"]
inv_map = {v: k for k, v in label_map.items()}

class_to_id = {letter: idx for idx, letter in enumerate(ALPHABETS)}
alphabet_indices = [i for i, lbl in enumerate(labels) if inv_map[lbl] in ALPHABETS]

X_list = []
y_list = []

np.random.seed(42)

for i in alphabet_indices:
    letter = inv_map[labels[i]]
    cid = class_to_id[letter]
    raw_lm = feats[i]
    
    # Canonical upright rotation
    can_lm = rotate_to_canonical_upright(raw_lm)
    
    # 1. Base upright
    X_list.append(can_lm.flatten())
    y_list.append(cid)
    
    # 2. Mirrored
    m = can_lm.copy()
    m[:, 0] = -m[:, 0]
    X_list.append(m.flatten())
    y_list.append(cid)
    
    # 3. Small rotation augmentations (-15, +15 deg)
    for rot in [-0.25, 0.25]:
        r = rotate_xy(can_lm, rot)
        X_list.append(r.flatten())
        y_list.append(cid)

print(f"Dataset samples after rotation and mirroring: {len(X_list)}")

# 2. Generate synthetic ASL canonical poses for all 26 letters
# to ensure both standard ASL and dataset are recognized perfectly
def build_asl_canonical(letter):
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.0, 0.0, 0.0] # wrist
    
    # Base MCPs (pointing straight up)
    lm[1] = [-0.15, -0.15, -0.05]
    lm[2] = [-0.25, -0.35, -0.05]
    lm[5] = [-0.12, -0.55, -0.02]
    lm[9] = [0.00,  -0.58,  0.00]
    lm[13] = [0.12, -0.55, -0.02]
    lm[17] = [0.22, -0.48, -0.04]
    
    # Finger configs for each letter
    # Helper to set finger: (mcp, dx, dy_pip, dy_dip, dy_tip, dz)
    def ext_finger(base_idx, x_off, pip_y=-0.72, dip_y=-0.86, tip_y=-1.00):
        lm[base_idx + 1] = [x_off, pip_y, -0.02]
        lm[base_idx + 2] = [x_off, dip_y, -0.03]
        lm[base_idx + 3] = [x_off, tip_y, -0.04]
        
    def curl_finger(base_idx, x_off):
        lm[base_idx + 1] = [x_off, -0.62, 0.08]
        lm[base_idx + 2] = [x_off * 0.8, -0.48, 0.12]
        lm[base_idx + 3] = [x_off * 0.6, -0.38, 0.08]
        
    def curve_finger(base_idx, x_off):
        lm[base_idx + 1] = [x_off * 1.1, -0.68, 0.05]
        lm[base_idx + 2] = [x_off * 1.2, -0.74, 0.12]
        lm[base_idx + 3] = [x_off * 1.0, -0.68, 0.18]

    # Thumb positions
    def thumb_up():
        lm[3] = [-0.22, -0.52, -0.02]
        lm[4] = [-0.20, -0.68,  0.02]
    def thumb_out():
        lm[3] = [-0.42, -0.40, -0.02]
        lm[4] = [-0.58, -0.42, -0.02]
    def thumb_across():
        lm[3] = [-0.08, -0.42,  0.10]
        lm[4] = [ 0.05, -0.42,  0.12]
    def thumb_tuck():
        lm[3] = [-0.05, -0.48,  0.06]
        lm[4] = [ 0.00, -0.52,  0.04]

    # Specific letter poses
    if letter == 'A':
        curl_finger(5, -0.12); curl_finger(9, 0.0); curl_finger(13, 0.12); curl_finger(17, 0.22)
        thumb_up()
    elif letter == 'B':
        ext_finger(5, -0.12); ext_finger(9, 0.0); ext_finger(13, 0.12); ext_finger(17, 0.22)
        thumb_across()
    elif letter == 'C':
        curve_finger(5, -0.12); curve_finger(9, 0.0); curve_finger(13, 0.12); curve_finger(17, 0.22)
        lm[3] = [-0.32, -0.45, 0.10]; lm[4] = [-0.22, -0.58, 0.15]
    elif letter == 'D':
        ext_finger(5, -0.12); curve_finger(9, 0.0); curve_finger(13, 0.12); curve_finger(17, 0.22)
        lm[3] = [-0.15, -0.48, 0.08]; lm[4] = [-0.02, -0.55, 0.10]
    elif letter == 'E':
        for b, x in [(5, -0.12), (9, 0.0), (13, 0.12), (17, 0.22)]:
            lm[b+1] = [x, -0.65, 0.08]; lm[b+2] = [x, -0.58, 0.14]; lm[b+3] = [x, -0.52, 0.10]
        thumb_across()
    elif letter == 'F':
        lm[6] = [-0.10, -0.65, 0.05]; lm[7] = [-0.08, -0.58, 0.10]; lm[8] = [-0.05, -0.52, 0.12]
        ext_finger(9, 0.0); ext_finger(13, 0.14); ext_finger(17, 0.26)
        lm[3] = [-0.15, -0.45, 0.05]; lm[4] = [-0.06, -0.52, 0.10]
    elif letter == 'G':
        lm[6] = [0.15, -0.58, 0.0]; lm[7] = [0.35, -0.58, 0.0]; lm[8] = [0.55, -0.58, 0.0]
        curl_finger(9, 0.0); curl_finger(13, 0.12); curl_finger(17, 0.22)
        lm[3] = [0.10, -0.45, 0.0]; lm[4] = [0.28, -0.45, 0.0]
    elif letter == 'H':
        lm[6] = [0.15, -0.60, 0.0]; lm[7] = [0.35, -0.60, 0.0]; lm[8] = [0.55, -0.60, 0.0]
        lm[10] = [0.15, -0.52, 0.0]; lm[11] = [0.35, -0.52, 0.0]; lm[12] = [0.55, -0.52, 0.0]
        curl_finger(13, 0.12); curl_finger(17, 0.22); thumb_across()
    elif letter == 'I':
        curl_finger(5, -0.12); curl_finger(9, 0.0); curl_finger(13, 0.12); ext_finger(17, 0.22)
        thumb_across()
    elif letter == 'J':
        curl_finger(5, -0.12); curl_finger(9, 0.0); curl_finger(13, 0.12); ext_finger(17, 0.22)
        thumb_across()
    elif letter == 'K':
        ext_finger(5, -0.12); lm[10] = [0.0, -0.68, 0.15]; lm[11] = [0.0, -0.80, 0.25]; lm[12] = [0.0, -0.92, 0.35]
        curl_finger(13, 0.12); curl_finger(17, 0.22); thumb_up()
    elif letter == 'L':
        ext_finger(5, -0.12); curl_finger(9, 0.0); curl_finger(13, 0.12); curl_finger(17, 0.22)
        thumb_out()
    elif letter == 'M':
        for b, x in [(5, -0.12), (9, 0.0), (13, 0.12)]:
            lm[b+1] = [x, -0.60, 0.08]; lm[b+2] = [x, -0.50, 0.12]; lm[b+3] = [x, -0.40, 0.08]
        curl_finger(17, 0.22); thumb_tuck()
    elif letter == 'N':
        for b, x in [(5, -0.12), (9, 0.0)]:
            lm[b+1] = [x, -0.60, 0.08]; lm[b+2] = [x, -0.50, 0.12]; lm[b+3] = [x, -0.40, 0.08]
        curl_finger(13, 0.12); curl_finger(17, 0.22); thumb_tuck()
    elif letter == 'O':
        for b, x in [(5, -0.10), (9, 0.0), (13, 0.10), (17, 0.18)]:
            lm[b+1] = [x, -0.65, 0.08]; lm[b+2] = [x*0.7, -0.58, 0.14]; lm[b+3] = [0.0, -0.50, 0.15]
        lm[3] = [-0.18, -0.45, 0.08]; lm[4] = [0.0, -0.50, 0.15]
    elif letter == 'P':
        lm[6] = [-0.12, -0.50, 0.0]; lm[7] = [-0.12, -0.35, 0.0]; lm[8] = [-0.12, -0.20, 0.0]
        curl_finger(9, 0.0); curl_finger(13, 0.12); curl_finger(17, 0.22); thumb_out()
    elif letter == 'Q':
        lm[6] = [-0.12, -0.45, 0.0]; lm[7] = [-0.12, -0.32, 0.0]; lm[8] = [-0.12, -0.20, 0.0]
        curl_finger(9, 0.0); curl_finger(13, 0.12); curl_finger(17, 0.22)
        lm[3] = [-0.20, -0.40, 0.0]; lm[4] = [-0.20, -0.25, 0.0]
    elif letter == 'R':
        # Crossed index & middle
        lm[6] = [-0.08, -0.72, -0.02]; lm[7] = [-0.02, -0.86, -0.02]; lm[8] = [0.05, -1.00, -0.02]
        lm[10] = [-0.02, -0.72, 0.02]; lm[11] = [-0.08, -0.86, 0.02]; lm[12] = [-0.12, -1.00, 0.02]
        curl_finger(13, 0.12); curl_finger(17, 0.22); thumb_across()
    elif letter == 'S':
        curl_finger(5, -0.12); curl_finger(9, 0.0); curl_finger(13, 0.12); curl_finger(17, 0.22)
        lm[3] = [-0.05, -0.42, 0.15]; lm[4] = [0.08, -0.42, 0.16]
    elif letter == 'T':
        lm[6] = [-0.12, -0.62, 0.08]; lm[7] = [-0.10, -0.50, 0.12]; lm[8] = [-0.08, -0.40, 0.08]
        curl_finger(9, 0.0); curl_finger(13, 0.12); curl_finger(17, 0.22)
        lm[3] = [-0.10, -0.46, 0.08]; lm[4] = [-0.05, -0.50, 0.10]
    elif letter == 'U':
        # Index & mid straight up touching tight
        ext_finger(5, -0.04); ext_finger(9, 0.04)
        curl_finger(13, 0.12); curl_finger(17, 0.22); thumb_across()
    elif letter == 'V':
        # Index & mid spread in V
        ext_finger(5, -0.18); ext_finger(9, 0.12)
        curl_finger(13, 0.12); curl_finger(17, 0.22); thumb_across()
    elif letter == 'W':
        ext_finger(5, -0.20); ext_finger(9, 0.0); ext_finger(13, 0.20)
        curl_finger(17, 0.25); thumb_across()
    elif letter == 'X':
        lm[6] = [-0.12, -0.68, 0.0]; lm[7] = [-0.12, -0.62, 0.12]; lm[8] = [-0.12, -0.52, 0.10]
        curl_finger(9, 0.0); curl_finger(13, 0.12); curl_finger(17, 0.22); thumb_across()
    elif letter == 'Y':
        curl_finger(5, -0.12); curl_finger(9, 0.0); curl_finger(13, 0.12); ext_finger(17, 0.28)
        thumb_out()
    elif letter == 'Z':
        ext_finger(5, -0.12); curl_finger(9, 0.0); curl_finger(13, 0.12); curl_finger(17, 0.22)
        thumb_across()
        
    max_d = np.max(np.linalg.norm(lm, axis=1))
    return lm / max_d

# Augment synthetic ASL poses (150 samples per letter with jitter and rotation)
for letter in ALPHABETS:
    cid = class_to_id[letter]
    base = build_asl_canonical(letter)
    for _ in range(120):
        # jitter
        noisy = base + np.random.normal(0, 0.015, base.shape).astype(np.float32)
        # random rotation (-18 to +18 deg)
        rot_ang = np.random.uniform(-0.30, 0.30)
        r = rotate_xy(noisy, rot_ang)
        # re-normalize
        maxd = np.max(np.linalg.norm(r, axis=1))
        norm_r = r / maxd
        X_list.append(norm_r.flatten())
        y_list.append(cid)
        # mirrored
        m = norm_r.copy()
        m[:, 0] = -m[:, 0]
        X_list.append(m.flatten())
        y_list.append(cid)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)

print(f"Total balanced dataset: X = {X.shape}, y = {y.shape}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

print(f"Training robust 4-layer MLP (63 -> 256 -> 128 -> 64 -> 26)...")
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    batch_size=256,
    max_iter=100,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10
)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"=======================================================")
print(f"  ROBUST BALANCED TEST ACCURACY: {acc * 100:.2f}%")
print(f"=======================================================")

# Test on clean ASL canonical letters
print("\n--- Validation on Pure ASL Canonical Signs ---")
asl_correct = 0
for letter in ALPHABETS:
    cid = class_to_id[letter]
    clean_asl = build_asl_canonical(letter).flatten().reshape(1, -1)
    pred_cid = mlp.predict(clean_asl)[0]
    pred_letter = ALPHABETS[pred_cid]
    prob = np.max(mlp.predict_proba(clean_asl)[0])
    is_ok = (pred_letter == letter)
    if is_ok: asl_correct += 1
    mark = "✓" if is_ok else "✗"
    print(f"{mark} ASL '{letter}' -> Predicted: {pred_letter} ({prob*100:.1f}%)")

print(f"\nASL Signs Accuracy: {asl_correct}/26 ({asl_correct/26*100:.1f}%)")
