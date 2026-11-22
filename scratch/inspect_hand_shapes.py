import pickle
import numpy as np

with open('cnn_dataset.pickle', 'rb') as f:
    d = pickle.load(f)

feats = d['features'] # (16425, 2, 21, 3)
labels = d['labels']
label_map = d['label_map']
inv_map = {v: k for k, v in label_map.items()}

# For each letter A-Z, let's look at average distance between wrist and fingertips:
# Index tip = 8, Middle tip = 12, Ring tip = 16, Pinky tip = 20, Thumb tip = 4, Wrist = 0
print(f"{'Class':<6} {'Wrist->Index':<14} {'Wrist->Pinky':<14} {'Wrist->Thumb':<14} {'Index->Thumb':<14}")
print("-" * 62)

for letter in [chr(c) for c in range(ord('A'), ord('Z')+1)]:
    indices = np.where(labels == label_map[letter])[0]
    sample_feats = feats[indices, 0, :, :] # (N, 21, 3)
    
    # Distances
    d_index = np.linalg.norm(sample_feats[:, 8, :] - sample_feats[:, 0, :], axis=1).mean()
    d_pinky = np.linalg.norm(sample_feats[:, 20, :] - sample_feats[:, 0, :], axis=1).mean()
    d_thumb = np.linalg.norm(sample_feats[:, 4, :] - sample_feats[:, 0, :], axis=1).mean()
    d_idx_th = np.linalg.norm(sample_feats[:, 8, :] - sample_feats[:, 4, :], axis=1).mean()
    
    print(f"{letter:<6} {d_index:<14.3f} {d_pinky:<14.3f} {d_thumb:<14.3f} {d_idx_th:<14.3f}")
