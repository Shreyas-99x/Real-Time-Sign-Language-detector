import json
import numpy as np

with open('web/model/model.json', 'r') as f:
    m = json.load(f)

weights = [np.array(w) for w in m['weights']]
biases = [np.array(b) for b in m['biases']]
classes = m['classes']

def predict(vec):
    curr = vec
    for i in range(len(weights)):
        out = np.dot(curr, weights[i]) + biases[i]
        curr = np.maximum(0, out) if i < len(weights)-1 else out
    e = np.exp(curr - np.max(curr))
    probs = e / np.sum(e)
    return classes[np.argmax(probs)], np.max(probs)

def normalize(lm):
    wrist = lm[0]
    rel = lm - wrist
    max_d = np.max(np.linalg.norm(rel, axis=1))
    if max_d < 1e-6: max_d = 1.0
    return (rel / max_d).flatten()

# Let's inspect class Z centroid vs class A, B, C centroids in landmarks.json
with open('web/model/landmarks.json') as f:
    knn = json.load(f)

centroids = knn['centroids']
print("Landmarks classes:", list(centroids.keys()))

# Check centroid of Z
c_Z = np.array(centroids['Z'])
c_A = np.array(centroids['A'])
c_B = np.array(centroids['B'])
c_C = np.array(centroids['C'])

print("Norm of c_Z:", np.linalg.norm(c_Z))
print("Norm of c_A:", np.linalg.norm(c_A))
print("Norm of c_B:", np.linalg.norm(c_B))
print("Norm of c_C:", np.linalg.norm(c_C))

# Test predict on centroids
for letter in ['A', 'B', 'C', 'D', 'E', 'L', 'V', 'Y', 'Z']:
    p, c = predict(np.array(centroids[letter]))
    print(f"Centroid of '{letter}' -> predicts '{p}' ({c*100:.1f}%)")
