import json
import numpy as np

with open('web/model/landmarks.json') as f:
    knn = json.load(f)

centroids = knn['centroids']

# Let's test what centroid is closest to random hand shapes:
# Test a typical upright open palm:
open_hand = np.zeros(63)
# wrist is at 0, 0, 0
# fingertips pointing up: y is negative (-0.7 to -1.0)
for tip in [4, 8, 12, 16, 20]:
    open_hand[tip*3 + 1] = -0.9

dists = {}
for cls, c in centroids.items():
    d = np.sum((open_hand - np.array(c))**2)
    dists[cls] = d

closest = sorted(dists.items(), key=lambda x: x[1])[:5]
print("Closest centroid to upright open palm:", closest)

# Test a typical fist:
fist = np.zeros(63)
for tip in [4, 8, 12, 16, 20]:
    fist[tip*3 + 1] = -0.3

dists = {}
for cls, c in centroids.items():
    d = np.sum((fist - np.array(c))**2)
    dists[cls] = d

closest = sorted(dists.items(), key=lambda x: x[1])[:5]
print("Closest centroid to fist:", closest)

# Test a flat vector:
dists = {}
for cls, c in centroids.items():
    d = np.sum((np.zeros(63) - np.array(c))**2)
    dists[cls] = d

closest = sorted(dists.items(), key=lambda x: x[1])[:5]
print("Closest centroid to zero vector:", closest)
