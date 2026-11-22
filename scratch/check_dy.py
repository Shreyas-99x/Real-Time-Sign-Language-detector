import pickle
import numpy as np

with open('cnn_dataset.pickle', 'rb') as f:
    d = pickle.load(f)

feats = d['features'][:, 0, :, :] # (16425, 21, 3)
labels = d['labels']
label_map = d['label_map']

print(f"{'Letter':<8} {'Wrist Y':<10} {'IndexTip Y':<12} {'dy (Index-Wrist)':<18} {'dy (Middle-Wrist)':<18}")
print("-" * 68)

for letter in [chr(c) for c in range(ord('A'), ord('Z')+1)]:
    idx = np.where(labels == label_map[letter])[0]
    sample_feats = feats[idx, :, :]
    wrist_y = sample_feats[:, 0, 1].mean()
    index_y = sample_feats[:, 8, 1].mean()
    middle_y = sample_feats[:, 12, 1].mean()
    dy_index = (sample_feats[:, 8, 1] - sample_feats[:, 0, 1]).mean()
    dy_middle = (sample_feats[:, 12, 1] - sample_feats[:, 0, 1]).mean()
    print(f"{letter:<8} {wrist_y:<10.3f} {index_y:<12.3f} {dy_index:<18.3f} {dy_middle:<18.3f}")
