import pickle
import numpy as np

with open('cnn_dataset.pickle', 'rb') as f:
    d = pickle.load(f)

feats = d['features'][:, 0, :, :]
labels = d['labels']
label_map = d['label_map']

ALPHABETS = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
print(f"{'Letter':<8} {'dy (tip-wrist)':<18} {'dx (tip-wrist)':<18} {'angle (deg)':<12}")
for let in ALPHABETS:
    idx = label_map[let]
    s = feats[labels == idx]
    dy = s[:, 12, 1] - s[:, 0, 1]
    dx = s[:, 12, 0] - s[:, 0, 0]
    angles = np.degrees(np.arctan2(dy, dx))
    print(f"{let:<8} {dy.mean():<18.3f} {dx.mean():<18.3f} {angles.mean():<12.1f}")
