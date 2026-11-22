import json
import numpy as np

with open('web/model/model.json') as f:
    m = json.load(f)

weights = [np.array(w, dtype=np.float32) for w in m['weights']]
biases = [np.array(b, dtype=np.float32) for b in m['biases']]
classes = m['classes']

def forward(vec):
    curr = np.array(vec, dtype=np.float32)
    for i in range(len(weights)):
        out = np.dot(curr, weights[i]) + biases[i]
        curr = np.maximum(0, out) if i < len(weights)-1 else out
    e = np.exp(curr - np.max(curr))
    probs = e / np.sum(e)
    top_indices = np.argsort(probs)[::-1]
    return classes[top_indices[0]], probs[top_indices[0]], [(classes[i], probs[i]) for i in top_indices[:3]]

# Let's test what forward predicts on actual upright hand gestures:
# MediaPipe coordinates for real hand in front of camera (center of 640x480 frame):
def make_real_mediapipe_hand(thumb_ext, index_ext, mid_ext, ring_ext, pinky_ext):
    lm = np.zeros((21, 3), dtype=np.float32)
    # Wrist at (0.50, 0.75, 0.0)
    lm[0] = [0.50, 0.75, 0.0]
    
    # MCPs
    lm[1] = [0.46, 0.70, -0.01]
    lm[2] = [0.43, 0.65, -0.02]
    lm[5] = [0.46, 0.55, -0.03]
    lm[9] = [0.50, 0.53, -0.03]
    lm[13] = [0.54, 0.55, -0.03]
    lm[17] = [0.58, 0.58, -0.03]
    
    # Thumb
    if thumb_ext: # thumb extended out (L or Y)
        lm[3] = [0.38, 0.64, -0.02]
        lm[4] = [0.32, 0.63, -0.02]
    else: # thumb folded across palm
        lm[3] = [0.46, 0.60, 0.02]
        lm[4] = [0.48, 0.58, 0.03]
        
    fingers = [(5, index_ext, 0.46), (9, mid_ext, 0.50), (13, ring_ext, 0.54), (17, pinky_ext, 0.58)]
    for mcp, is_ext, x in fingers:
        mcp_y = lm[mcp][1]
        if is_ext:
            lm[mcp + 1] = [x, mcp_y - 0.07, -0.03]
            lm[mcp + 2] = [x, mcp_y - 0.13, -0.04]
            lm[mcp + 3] = [x, mcp_y - 0.19, -0.05] # tip is higher up
        else:
            lm[mcp + 1] = [x, mcp_y - 0.03, 0.02]
            lm[mcp + 2] = [x, mcp_y + 0.02, 0.04]
            lm[mcp + 3] = [x, mcp_y + 0.05, 0.03] # tip curled down
            
    # Now apply the EXACT normalizeLandmarks from classifier.js:
    wrist = lm[0]
    mid_mcp = lm[9]
    dx = mid_mcp[0] - wrist[0]
    dy = mid_mcp[1] - wrist[1]
    angle = np.arctan2(dy, dx)
    rot_angle = -np.pi / 2.0 - angle
    cos_t = np.cos(rot_angle)
    sin_t = np.sin(rot_angle)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    
    rel = lm - wrist
    rot = rel.copy()
    rot[:, :2] = np.dot(rel[:, :2], R.T)
    max_d = np.max(np.linalg.norm(rot, axis=1))
    norm_vec = (rot / max_d).flatten()
    return norm_vec

test_cases = {
    'A (Fist)': (False, False, False, False, False),
    'B (4 fingers up)': (False, True, True, True, True),
    'D (Index up)': (False, True, False, False, False),
    'I (Pinky up)': (False, False, False, False, True),
    'L (Index & Thumb)': (True, True, False, False, False),
    'V (Index & Mid)': (False, True, True, False, False),
    'W (Index, Mid, Ring)': (False, True, True, True, False),
    'Y (Thumb & Pinky)': (True, False, False, False, True)
}

print(f"{'Gesture':<24} {'Predicted':<10} {'Confidence':<12} {'Top 3'}")
for name, params in test_cases.items():
    vec = make_real_mediapipe_hand(*params)
    pred, conf, top3 = forward(vec)
    t3 = ', '.join([f'{c}: {p*100:.1f}%' for c, p in top3])
    print(f"{name:<24} {pred:<10} {conf*100:<11.1f}% {t3}")
