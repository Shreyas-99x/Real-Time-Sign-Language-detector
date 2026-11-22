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

with open('web/model/landmarks.json') as f:
    knn = json.load(f)

print(f"{'Letter':<8} {'Perfect':<12} {'+3% noise':<15} {'+6% noise':<15}")
for let in classes:
    c = np.array(knn['centroids'][let], dtype=np.float32)
    p0, c0, _ = forward(c)
    
    np.random.seed(ord(let))
    p3, c3, _ = forward(c + np.random.normal(0, 0.03, 63).astype(np.float32))
    p6, c6, _ = forward(c + np.random.normal(0, 0.06, 63).astype(np.float32))
    print(f"{let:<8} {p0} ({c0*100:.0f}%)      {p3} ({c3*100:.0f}%)        {p6} ({c6*100:.0f}%)")
