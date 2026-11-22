import pickle
import cv2

with open('cnn_dataset.pickle', 'rb') as f:
    d = pickle.load(f)

label_map = d['label_map']
for letter in ['A', 'B', 'L', 'V', 'Z']:
    idx = [i for i, lbl in enumerate(d['labels']) if lbl == label_map[letter]][0]
    img = d['images'][idx]
    # Let's check non-zero pixels center of mass vs wrist
    cv2.imwrite(f'sample_{letter}.png', img)
    print(f"Saved sample_{letter}.png")
