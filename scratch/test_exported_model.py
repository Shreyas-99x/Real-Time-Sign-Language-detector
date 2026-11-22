import sys; sys.path.append('scratch')
import json, numpy as np
from test_clean_run import make_hand

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

print(f"{'Gesture':<25} {'Pred':<6} {'Conf':<8} {'Top 3'}")
for name, params in test_cases.items():
    lm = make_hand(*params)
    wrist = lm[0]; mid_mcp = lm[9]
    dx = mid_mcp[0] - wrist[0]; dy = mid_mcp[1] - wrist[1]
    rot_angle = -np.pi / 2.0 - np.arctan2(dy, dx)
    cos_t = np.cos(rot_angle); sin_t = np.sin(rot_angle)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    rel = lm - wrist
    rot = rel.copy()
    rot[:, :2] = np.dot(rel[:, :2], R.T)
    max_d = np.max(np.linalg.norm(rot, axis=1))
    vec = (rot / max_d).flatten()
    
    pred, conf, top3 = forward(vec)
    t3 = ', '.join([f'{c}: {p*100:.1f}%' for c, p in top3])
    print(f"{name:<25} {pred:<6} {conf*100:<7.1f}% {t3}")
