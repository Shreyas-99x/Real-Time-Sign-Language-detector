import pickle
import numpy as np

with open('cnn_dataset.pickle', 'rb') as f:
    d = pickle.load(f)

f = d['features'][:, 0, :, :] # (16425, 21, 3)
labels = d['labels']
label_map = d['label_map']
inv_map = {v: k for k, v in label_map.items()}

idx_z = np.where(labels == label_map['Z'])[0]
z_samples = f[idx_z] # (400, 21, 3)

wrist_z = z_samples[:, 0, :]
rel_z = z_samples - z_samples[:, 0:1, :]
dists_z = np.linalg.norm(rel_z, axis=2)
maxd_z = np.max(dists_z, axis=1)[:, None, None]
norm_z = rel_z / maxd_z

print("Class Z normalized landmarks (mean of 400 samples):")
for i in range(21):
    m = norm_z[:, i, :].mean(axis=0)
    s = norm_z[:, i, :].std(axis=0)
    print(f"  LM {i:2d}: x={m[0]:.3f} (±{s[0]:.2f}), y={m[1]:.3f} (±{s[1]:.2f}), z={m[2]:.3f} (±{s[2]:.2f})")
