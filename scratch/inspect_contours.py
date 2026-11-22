import cv2
import numpy as np

for name in ['sample_A.png', 'sample_B.png', 'sample_L.png', 'sample_V.png', 'sample_Z.png']:
    img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    # Find contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"{name}: found {len(contours)} contours")
    for idx, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 50:
            x, y, w, h = cv2.boundingRect(c)
            print(f"  Contour {idx}: area={area:.0f}, bbox=({x},{y},{w},{h})")
