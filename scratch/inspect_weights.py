import json
import numpy as np

with open('web/model/model.json', 'r') as f:
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

# Let's inspect the weights and biases of the model!
# Specifically, what does the output layer look like?
W_last = weights[-1] # shape (64, 26)
b_last = biases[-1]  # shape (26,)

print("b_last stats:")
for i, c in enumerate(classes):
    print(f"  {c}: bias = {b_last[i]:.4f}, W_norm = {np.linalg.norm(W_last[:, i]):.4f}")
