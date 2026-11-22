import numpy as np
from asl_dataset_generator import generate_asl_landmarks, ALPHABETS
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X_list = []
y_list = []
rng = np.random.RandomState(42)

for cid, letter in enumerate(ALPHABETS):
    for _ in range(600):
        lm = generate_asl_landmarks(letter, rng)
        # Jitter
        noisy = lm + rng.normal(0, 0.015, lm.shape).astype(np.float32)
        # Slight rotation (-20 to +20 deg)
        rot_ang = rng.uniform(-0.35, 0.35)
        cos_t = np.cos(rot_ang); sin_t = np.sin(rot_ang)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
        r = noisy.copy()
        r[:, :2] = np.dot(noisy[:, :2], R.T)
        max_d = np.max(np.linalg.norm(r, axis=1))
        norm_r = r / max_d
        
        X_list.append(norm_r.flatten())
        y_list.append(cid)
        
        # Mirrored
        m = norm_r.copy()
        m[:, 0] = -m[:, 0]
        X_list.append(m.flatten())
        y_list.append(cid)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)
print(f"Generated pure ASL dataset: X={X.shape}, y={y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    batch_size=128,
    max_iter=100,
    random_state=42,
    early_stopping=True
)

mlp.fit(X_train, y_train)
acc = accuracy_score(y_test, mlp.predict(X_test))
print(f"Pure ASL Test Accuracy: {acc * 100:.2f}%")

# Test on test cases
from test_clean_run import make_hand
test_cases = {
    'A (Fist, thumb up)': ('up', False, False, False, False),
    'B (4 fingers up)': ('across', True, True, True, True),
    'D (Index up, thumb in)': ('across', True, False, False, False),
    'I (Pinky up)': ('across', False, False, False, True),
    'L (Index & Thumb)': ('out', True, False, False, False),
    'V (Index & Mid)': ('across', True, True, False, False),
    'W (Index, Mid, Ring)': ('across', True, True, True, False),
    'Y (Thumb & Pinky)': ('out', False, False, False, True),
    'S (Fist, thumb across)': ('across', False, False, False, False),
}

print("\n--- Test on Real Hand Shapes ---")
for name, params in test_cases.items():
    lm = make_hand(*params)
    # canonical rotation
    wrist = lm[0]; mid_mcp = lm[9]
    dx = mid_mcp[0] - wrist[0]; dy = mid_mcp[1] - wrist[1]
    rot_angle = -np.pi / 2.0 - np.arctan2(dy, dx)
    cos_t = np.cos(rot_angle); sin_t = np.sin(rot_angle)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    rel = lm - wrist
    rot = rel.copy()
    rot[:, :2] = np.dot(rel[:, :2], R.T)
    max_d = np.max(np.linalg.norm(rot, axis=1))
    vec = (rot / max_d).flatten().reshape(1, -1)
    
    pred_idx = mlp.predict(vec)[0]
    prob = np.max(mlp.predict_proba(vec)[0])
    print(f"{name:<25} -> {ALPHABETS[pred_idx]} ({prob*100:.1f}%)")
