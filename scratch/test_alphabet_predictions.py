import json
import pickle
import numpy as np

with open('web/model/model.json', 'r') as f:
    m = json.load(f)

weights = [np.array(w, dtype=np.float32) for w in m['weights']]
biases = [np.array(b, dtype=np.float32) for b in m['biases']]
classes = m['classes']

with open('cnn_dataset.pickle', 'rb') as f:
    d = pickle.load(f)

feats = d['features'] # (16425, 2, 21, 3)
labels = d['labels']
label_map = d['label_map']

def js_normalize_landmarks(lm_21):
    # lm_21 is array of 21 (x, y, z)
    wrist = lm_21[0]
    rel = []
    max_dist = 0
    for p in lm_21:
        rx = p[0] - wrist[0]
        ry = p[1] - wrist[1]
        rz = p[2] - wrist[2]
        d = (rx*rx + ry*ry + rz*rz)**0.5
        if d > max_dist: max_dist = d
        rel.append([rx, ry, rz])
    if max_dist < 1e-6: max_dist = 1.0
    vec = []
    for r in rel:
        vec.extend([r[0]/max_dist, r[1]/max_dist, r[2]/max_dist])
    return np.array(vec, dtype=np.float32)

def js_forward(vector):
    curr = vector
    for l in range(len(weights)):
        out = np.dot(curr, weights[l]) + biases[l]
        curr = np.maximum(0, out) if l < len(weights)-1 else out
    e = np.exp(curr - np.max(curr))
    probs = e / np.sum(e)
    idx = np.argmax(probs)
    return classes[idx], probs[idx], probs

print("Testing 1 sample from each class A-Z using JS normalization & JS forward pass:")
all_correct = 0
for letter in classes:
    idx = np.where(labels == label_map[letter])[0][0]
    raw_lm = feats[idx, 0, :, :] # (21, 3)
    vec = js_normalize_landmarks(raw_lm)
    pred_cls, conf, probs = js_forward(vec)
    is_match = (pred_cls == letter)
    if is_match: all_correct += 1
    print(f"  Truth: {letter} -> Predicted: {pred_cls} ({conf*100:.1f}%) [{'PASS' if is_match else 'FAIL'}]")

print(f"\nResult: {all_correct}/26 match truth.")
