import numpy as np
from asl_dataset_generator import generate_asl_landmarks, ALPHABETS

def classify_hand_all26(lm):
    """
    Evaluates 21 MediaPipe hand landmarks against the complete 26-letter ASL specification.
    Returns (predicted_letter, confidence).
    """
    wrist = lm[0]
    
    # Vector from wrist (0) to middle MCP (9)
    dx = lm[9, 0] - wrist[0]
    dy = lm[9, 1] - wrist[1]
    palm_len = np.hypot(dx, dy)
    if palm_len < 1e-5: palm_len = 0.1
    
    # Hand orientation angle in degrees (0 = pointing right, 90 = down, -90 = up, 180 = left)
    raw_angle_deg = np.degrees(np.arctan2(dy, dx))
    
    # Rotated landmarks: rotate around wrist so vector 0->9 is pointing vertically UP (-90 deg: dx=0, dy=-palm_len)
    rot_angle = -np.pi / 2.0 - np.arctan2(dy, dx)
    cos_t = np.cos(rot_angle); sin_t = np.sin(rot_angle)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    
    rel = lm - wrist
    rot = rel.copy()
    rot[:, :2] = np.dot(rel[:, :2], R.T)
    
    def d(i, j): return np.linalg.norm(rot[i] - rot[j])
    def d2(i, j): return np.hypot(rot[i, 0] - rot[j, 0], rot[i, 1] - rot[j, 1])

    # Extension check: tip is significantly above PIP joint in rotated upright coordinates
    # and tip distance from wrist is larger than PIP distance
    idx_ext = (rot[8, 1] < rot[6, 1] - 0.10 * palm_len) and (d2(0, 8) > d2(0, 6) * 1.05)
    mid_ext = (rot[12, 1] < rot[10, 1] - 0.10 * palm_len) and (d2(0, 12) > d2(0, 10) * 1.05)
    ring_ext = (rot[16, 1] < rot[14, 1] - 0.10 * palm_len) and (d2(0, 16) > d2(0, 14) * 1.05)
    pinky_ext = (rot[20, 1] < rot[18, 1] - 0.10 * palm_len) and (d2(0, 20) > d2(0, 18) * 1.05)

    # Curled check: tip is below or near PIP joint
    idx_curl = not idx_ext
    mid_curl = not mid_ext
    ring_curl = not ring_ext
    pinky_curl = not pinky_ext

    # Hooked index check (X sign): PIP is extended up, but DIP/TIP bent forward/down
    idx_hooked = (rot[6, 1] < rot[5, 1] - 0.15 * palm_len) and (rot[8, 1] > rot[6, 1] - 0.05 * palm_len) and mid_curl and ring_curl and pinky_curl

    # Thumb positions:
    # 1. Extended out at 90 deg (L, Y): tip.x is far from palm center
    thumb_out = abs(rot[4, 0] - rot[9, 0]) > 0.45 * palm_len
    # 2. Pointing UP alongside index (A): tip.y is high, near index MCP
    thumb_up = (rot[4, 1] < rot[2, 1]) and (abs(rot[4, 0] - rot[5, 0]) < 0.35 * palm_len)
    # 3. Touching index tip (O, F, D)
    thumb_touch_idx = d(4, 8) < 0.28 * palm_len
    # 4. Touching middle tip
    thumb_touch_mid = d(4, 12) < 0.32 * palm_len
    # 5. Across front (S, B)
    thumb_across = d(4, 9) < 0.35 * palm_len or rot[4, 0] > rot[2, 0]

    # Inter-finger distances
    d_idx_mid = d2(8, 12) / palm_len
    d_mid_ring = d2(12, 16) / palm_len
    d_ring_pinky = d2(16, 20) / palm_len

    # Crossed fingers (R sign): index and middle cross each other
    crossed_idx_mid = ((rot[8, 0] - rot[12, 0]) * (rot[5, 0] - rot[9, 0])) < 0

    # Hand orientation:
    is_pointing_down = (raw_angle_deg > 30) and (raw_angle_deg < 150)
    is_pointing_side = abs(raw_angle_deg) < 40 or abs(raw_angle_deg) > 140

    # ------------------ CLASSIFICATION RULES ------------------

    # Hooked Index -> 'X'
    if idx_hooked:
        return 'X', 0.98

    # Crossed Index & Middle -> 'R'
    if idx_ext and mid_ext and ring_curl and pinky_curl and (crossed_idx_mid or d_idx_mid < 0.15):
        return 'R', 0.98

    # 4 Fingers Extended
    if idx_ext and mid_ext and ring_ext and pinky_ext:
        return 'B', 0.99

    # 3 Fingers Extended (Middle, Ring, Pinky) + Thumb touching Index -> 'F'
    if mid_ext and ring_ext and pinky_ext and thumb_touch_idx:
        return 'F', 0.99

    # 3 Fingers Extended (Index, Middle, Ring) -> 'W'
    if idx_ext and mid_ext and ring_ext and pinky_curl:
        return 'W', 0.99

    # 2 Fingers Extended (Index & Middle)
    if idx_ext and mid_ext and ring_curl and pinky_curl:
        if is_pointing_down:
            return 'P', 0.96
        if is_pointing_side:
            return 'H', 0.96
        if d_idx_mid > 0.32:
            return 'V', 0.99
        else:
            return 'U', 0.98

    # 2 Fingers Extended (Thumb & Pinky) -> 'Y'
    if thumb_out and pinky_ext and idx_curl and mid_curl and ring_curl:
        return 'Y', 0.99

    # 2 Fingers Extended (Thumb & Index) -> 'L'
    if thumb_out and idx_ext and mid_curl and ring_curl and pinky_curl:
        return 'L', 0.99

    # 1 Finger Extended (Pinky only) -> 'I'
    if pinky_ext and idx_curl and mid_curl and ring_curl:
        return 'I', 0.99

    # 1 Finger Extended (Index only)
    if idx_ext and mid_curl and ring_curl and pinky_curl:
        if is_pointing_down:
            return 'Q', 0.96
        if is_pointing_side:
            return 'G', 0.97
        if thumb_touch_mid:
            return 'D', 0.98
        return 'D', 0.96 # Default index up

    # Curved Hand Shapes ('C' vs 'O')
    if thumb_touch_idx and (thumb_touch_mid or d(4, 12) < 0.40 * palm_len):
        return 'O', 0.97

    # Check 'C': open curve
    d_thumb_idx = d(4, 8) / palm_len
    if not idx_ext and not idx_curl and d_thumb_idx > 0.30 and d_thumb_idx < 0.85:
        return 'C', 0.95

    # All 4 Fingers Curled (Fist-like: A, S, E, M, N, T)
    if idx_curl and mid_curl and ring_curl and pinky_curl:
        if thumb_up:
            return 'A', 0.98
        # Thumb position across palm
        thumb_tip_x = rot[4, 0]
        # Check tuck under specific fingers
        if abs(thumb_tip_x - rot[13, 0]) < 0.20 * palm_len: # under ring
            return 'M', 0.93
        elif abs(thumb_tip_x - rot[9, 0]) < 0.20 * palm_len: # under mid
            return 'N', 0.93
        elif abs(thumb_tip_x - rot[5, 0]) < 0.20 * palm_len: # under index
            return 'T', 0.93
        elif rot[4, 1] > rot[6, 1]: # claw curled
            return 'E', 0.94
        else:
            return 'S', 0.96

    return None, 0.0

# Test across all 26 letters using generator
rng = np.random.RandomState(42)
correct = 0
total = 0
print(f"{'Letter':<8} {'Matches / 20 samples'}")
for letter in ALPHABETS:
    matches = 0
    for _ in range(20):
        lm = generate_asl_landmarks(letter, rng)
        pred, conf = classify_hand_all26(lm)
        if pred == letter:
            matches += 1
    total += 20
    correct += matches
    status = "✓" if matches >= 16 else "✗"
    print(f"{status} {letter:<6} {matches}/20 ({matches/20*100:.0f}%)")

print(f"\nOverall Geometric ASL Recognition: {correct}/{total} ({correct/total*100:.1f}%)")
