import pickle
import numpy as np

with open('cnn_dataset.pickle', 'rb') as f:
    d = pickle.load(f)

f = d['features'][:, 0, :, :]
labels = d['labels']
label_map = d['label_map']
inv_map = {v: k for k, v in label_map.items()}

rel = f - f[:, 0:1, :]
maxd = np.max(np.linalg.norm(rel, axis=2), axis=1)[:, None, None]
norm_f = rel / maxd

ALPHABETS = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
print(f"{'Letter':<8} {'Tip12 X':<10} {'Tip12 Y':<10} {'Tip8 X':<10} {'Tip8 Y':<10} {'Pose description'}")
for let in ALPHABETS:
    idx = label_map[let]
    samples = norm_f[labels == idx]
    t12_x = samples[:, 12, 0].mean()
    t12_y = samples[:, 12, 1].mean()
    t8_x = samples[:, 8, 0].mean()
    t8_y = samples[:, 8, 1].mean()
    
    # describe
    desc = []
    if t12_y < -0.7: desc.append("Upright")
    elif t12_y > 0.3: desc.append("Pointing Down")
    elif abs(t12_x) > 0.6: desc.append("Horizontal")
    else: desc.append("Curled / Tilted")
    
    print(f"{let:<8} {t12_x:<10.2f} {t12_y:<10.2f} {t8_x:<10.2f} {t8_y:<10.2f} {' '.join(desc)}")
