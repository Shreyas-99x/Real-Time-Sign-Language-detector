import numpy as np

def classify_asl_hand(lm):
    """
    Classifies 21 MediaPipe hand landmarks into one of the 26 ASL alphabet letters.
    lm is (21, 3) in image coordinates (x in [0, 1], y in [0, 1], z relative).
    """
    wrist = lm[0]
    
    # Rotate hand so vector from wrist (0) to middle MCP (9) is vertically up (-Y)
    dx = lm[9, 0] - wrist[0]
    dy = lm[9, 1] - wrist[1]
    angle = np.arctan2(dy, dx)
    rot_angle = -np.pi / 2.0 - angle
    cos_t = np.cos(rot_angle)
    sin_t = np.sin(rot_angle)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    
    rel = lm - wrist
    rot = rel.copy()
    rot[:, :2] = np.dot(rel[:, :2], R.T)
    
    palm_len = np.linalg.norm(rot[9, :2])
    if palm_len < 1e-5: palm_len = 0.1
    
    def d(i, j):
        return np.linalg.norm(rot[i] - rot[j])
        
    def d2(i, j):
        return np.linalg.norm(rot[i, :2] - rot[j, :2])

    # In rotated coordinates, Y is negative UPWARDS:
    # A finger is EXTENDED if tip.y is significantly above PIP.y and DIP.y
    # In rotated coords: tip.y < pip.y - 0.2 * palm_len
    idx_ext = (rot[8, 1] < rot[6, 1] - 0.15 * palm_len) and (d2(0, 8) > d2(0, 6) * 1.1)
    mid_ext = (rot[12, 1] < rot[10, 1] - 0.15 * palm_len) and (d2(0, 12) > d2(0, 10) * 1.1)
    ring_ext = (rot[16, 1] < rot[14, 1] - 0.15 * palm_len) and (d2(0, 16) > d2(0, 14) * 1.1)
    pinky_ext = (rot[20, 1] < rot[18, 1] - 0.15 * palm_len) and (d2(0, 20) > d2(0, 18) * 1.1)

    # Curled: tip is close to palm or below PIP
    idx_curl = (rot[8, 1] > rot[6, 1] - 0.05 * palm_len) or (d2(0, 8) < d2(0, 5) * 1.15)
    mid_curl = (rot[12, 1] > rot[10, 1] - 0.05 * palm_len) or (d2(0, 12) < d2(0, 9) * 1.15)
    ring_curl = (rot[16, 1] > rot[14, 1] - 0.05 * palm_len) or (d2(0, 16) < d2(0, 13) * 1.15)
    pinky_curl = (rot[20, 1] > rot[18, 1] - 0.05 * palm_len) or (d2(0, 20) < d2(0, 17) * 1.15)

    # Thumb:
    # Extended out sideways (L, Y): tip.x is far from palm center
    thumb_out = abs(rot[4, 0] - rot[9, 0]) > 0.55 * palm_len
    # Thumb up alongside index (A, K): tip.y is above MCP 2 and close to MCP 5
    thumb_up = (rot[4, 1] < rot[2, 1]) and (d2(4, 5) < 0.45 * palm_len)
    # Thumb touching index tip (O, F, D)
    thumb_touches_index = d(4, 8) < 0.35 * palm_len
    # Thumb touching middle tip
    thumb_touches_mid = d(4, 12) < 0.40 * palm_len

    # Inter-finger distances
    d_idx_mid = d2(8, 12) / palm_len
    d_thumb_pinky = d2(4, 20) / palm_len

    # Decision tree for ASL letters
    # 1. 'L': Index up, Thumb out, other 3 curled
    if idx_ext and thumb_out and mid_curl and ring_curl and pinky_curl:
        return 'L', 0.98

    # 2. 'Y': Thumb out, Pinky up, index/mid/ring curled
    if thumb_out and pinky_ext and idx_curl and mid_curl and ring_curl:
        return 'Y', 0.98

    # 3. 'V' vs 'U': Index & Mid up, ring/pinky curled
    if idx_ext and mid_ext and ring_curl and pinky_curl:
        if d_idx_mid > 0.35:
            return 'V', 0.98
        else:
            return 'U', 0.97

    # 4. 'W': Index, Mid, Ring up, pinky curled
    if idx_ext and mid_ext and ring_ext and pinky_curl:
        return 'W', 0.98

    # 5. 'B': 4 fingers up, thumb across
    if idx_ext and mid_ext and ring_ext and pinky_ext:
        return 'B', 0.98

    # 6. 'F': Index & Thumb touching in circle, mid/ring/pinky up
    if thumb_touches_index and mid_ext and ring_ext and pinky_ext:
        return 'F', 0.98

    # 7. 'I': Only pinky up, others curled
    if pinky_ext and idx_curl and mid_curl and ring_curl and not thumb_out:
        return 'I', 0.97

    # 8. 'D': Index up, others curled (thumb touching mid/ring)
    if idx_ext and mid_curl and ring_curl and pinky_curl and not thumb_out:
        return 'D', 0.97

    # 9. 'C' vs 'O':
    if thumb_touches_index and thumb_touches_mid and idx_curl and mid_curl:
        return 'O', 0.96

    # 10. 'A' vs 'S':
    if idx_curl and mid_curl and ring_curl and pinky_curl:
        if thumb_up:
            return 'A', 0.96
        else:
            return 'S', 0.94

    return None, 0.0

print("Classifier logic written.")
