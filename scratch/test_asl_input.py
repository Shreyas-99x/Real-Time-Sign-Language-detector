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
    return classes[np.argmax(probs)], np.max(probs), {classes[i]: probs[i] for i in range(len(classes))}

# Construct a realistic ASL "L" shape:
# 21 landmarks:
# 0: wrist at (0.5, 0.8, 0)
# 1-4: thumb extending to the left (x decreases)
# 5-8: index extending up (y decreases)
# 9-12: middle curled down (tip near palm)
# 13-16: ring curled down
# 17-20: pinky curled down
asl_L = np.zeros((21, 3))
asl_L[0] = [0.5, 0.8, 0.0] # wrist
# Thumb extending out
asl_L[1] = [0.45, 0.75, 0.0]
asl_L[2] = [0.40, 0.70, 0.0]
asl_L[3] = [0.35, 0.68, 0.0]
asl_L[4] = [0.30, 0.68, 0.0] # thumb tip
# Index extending straight up
asl_L[5] = [0.48, 0.65, 0.0]
asl_L[6] = [0.48, 0.55, 0.0]
asl_L[7] = [0.48, 0.45, 0.0]
asl_L[8] = [0.48, 0.35, 0.0] # index tip
# Middle curled
asl_L[9] = [0.51, 0.65, 0.0]
asl_L[10] = [0.51, 0.58, 0.0]
asl_L[11] = [0.51, 0.63, 0.0]
asl_L[12] = [0.51, 0.68, 0.0] # middle tip curled
# Ring curled
asl_L[13] = [0.54, 0.66, 0.0]
asl_L[14] = [0.54, 0.60, 0.0]
asl_L[15] = [0.54, 0.65, 0.0]
asl_L[16] = [0.54, 0.70, 0.0] # ring tip curled
# Pinky curled
asl_L[17] = [0.57, 0.68, 0.0]
asl_L[18] = [0.57, 0.63, 0.0]
asl_L[19] = [0.57, 0.68, 0.0]
asl_L[20] = [0.57, 0.72, 0.0] # pinky tip curled

# Normalize
wrist = asl_L[0]
rel = asl_L - wrist
max_d = np.max(np.linalg.norm(rel, axis=1))
norm_vec = (rel / max_d).flatten()

pred, conf, all_probs = predict(norm_vec)
print(f"ASL 'L' predicted: {pred} ({conf*100:.1f}%)")
# Top 5
top5 = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 for ASL 'L':", top5)
