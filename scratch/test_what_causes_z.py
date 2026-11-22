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

# Let's test 1000 random vectors with different variances:
for std in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
    preds = {}
    for _ in range(200):
        v = np.random.normal(0, std, 63)
        p, c = predict(v)
        preds[p] = preds.get(p, 0) + 1
    top_p = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"std={std:<5}: top preds = {top_p}")
