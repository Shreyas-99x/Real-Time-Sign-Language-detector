import pickle
import numpy as np

with open('cnn_dataset.pickle', 'rb') as f:
    d = pickle.load(f)

feats = d['features'] # (16425, 2, 21, 3)
labels = d['labels']
label_map = d['label_map']
inv_map = {v: k for k, v in label_map.items()}

# Let's inspect class Z
idx_z = np.where(labels == label_map['Z'])[0]
z_samples = feats[idx_z, 0, :, :] # (400, 21, 3)

print("Class Z mean landmark positions (first 5 landmarks):")
for i in range(5):
    print(f"  LM {i}: {z_samples[:, i, :].mean(axis=0)}")

print("\nClass A mean landmark positions (first 5 landmarks):")
idx_a = np.where(labels == label_map['A'])[0]
a_samples = feats[idx_a, 0, :, :]
for i in range(5):
    print(f"  LM {i}: {a_samples[:, i, :].mean(axis=0)}")
