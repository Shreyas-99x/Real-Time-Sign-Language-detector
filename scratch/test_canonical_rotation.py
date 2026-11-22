import pickle
import numpy as np

def rotate_to_canonical_upright(landmarks_21x3):
    """
    Rotates 21x3 landmarks in the XY plane so that the vector from
    Landmark 0 (wrist) to Landmark 9 (middle MCP) points vertically UP
    (i.e. angle = -90 degrees in image coordinates: dx = 0, dy = -1).
    """
    wrist = landmarks_21x3[0]
    rel = landmarks_21x3 - wrist
    
    # Vector from wrist to middle MCP
    mid_mcp = rel[9]
    angle = np.arctan2(mid_mcp[1], mid_mcp[0]) # current angle in radians
    
    # Target angle is -pi/2 (-90 deg: straight up)
    rot_angle = -np.pi / 2 - angle
    
    cos_theta = np.cos(rot_angle)
    sin_theta = np.sin(rot_angle)
    R = np.array([
        [cos_theta, -sin_theta],
        [sin_theta,  cos_theta]
    ])
    
    rotated = rel.copy()
    rotated[:, :2] = np.dot(rel[:, :2], R.T)
    
    # Scale normalization
    max_d = np.max(np.linalg.norm(rotated, axis=1))
    if max_d < 1e-6:
        max_d = 1.0
    return rotated / max_d

# Test on a horizontal hand (like Letter Z in dataset) vs an upright hand:
# Horizontal hand: wrist at (0.5, 0.5), mid MCP at (0.6, 0.5) (dx=0.1, dy=0)
lm_horiz = np.zeros((21, 3))
lm_horiz[0] = [0.5, 0.5, 0]
lm_horiz[9] = [0.6, 0.5, 0]
lm_horiz[12] = [0.7, 0.5, 0] # tip extended along same line

# Upright hand: wrist at (0.5, 0.8), mid MCP at (0.5, 0.6) (dx=0, dy=-0.2)
lm_upright = np.zeros((21, 3))
lm_upright[0] = [0.5, 0.8, 0]
lm_upright[9] = [0.5, 0.6, 0]
lm_upright[12] = [0.5, 0.4, 0]

norm_h = rotate_to_canonical_upright(lm_horiz)
norm_u = rotate_to_canonical_upright(lm_upright)

print("Horizontal hand normalized mid MCP:", norm_h[9, :2], "tip:", norm_h[12, :2])
print("Upright hand normalized mid MCP:   ", norm_u[9, :2], "tip:", norm_u[12, :2])
print("Difference between horizontal and upright after canonical rotation:", np.max(np.abs(norm_h - norm_u)))
