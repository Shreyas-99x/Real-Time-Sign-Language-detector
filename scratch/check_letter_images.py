import os
import cv2
import pickle

with open('cnn_dataset.pickle', 'rb') as f:
    d = pickle.load(f)

label_map = d['label_map']
print("Classes in dataset:", sorted(label_map.keys()))

# Check images of A, B, C, Z
# Are they binary images of hands? What do they look like?
import numpy as np

for letter in ['A', 'B', 'C', 'D', 'Z']:
    idx = np.where(d['labels'] == label_map[letter])[0][0]
    img = d['images'][idx]
    print(f"Letter {letter}: shape {img.shape}, non-zero count: {np.count_nonzero(img)}, min={img.min()}, max={img.max()}")
