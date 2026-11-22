import sys; sys.path.append('scratch')
import numpy as np
from asl_dataset_generator import generate_asl_landmarks, ALPHABETS

def classify_hand_all26_v2(lm):
    wrist = lm[0]
    
    # Vector from wrist (0) to middle MCP (9)
    dx = lm[9, 0] - wrist[0]
    dy = lm[9, 1] - wrist[1]
    palm_len = np.hypot(dx, dy)
    if palm_len < 1e-5: palm_len = 0.1
    
    raw_angle_deg = np.degrees(np.arctan2(dy, dx))
    
    rot_angle = -np.pi / 2.0 - np.arctan2(dy, dx)
    cos_t = np.cos(rot_angle); sin_t = np.sin(rot_angle)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    
    rel = lm - wrist
    rot = rel.copy()
    rot[:, :2] = np.dot(rel[:, :2], R.T)
    
    def d(i, j): return np.linalg.norm(rot[i] - rot[j])
    def d2(i, j): return np.hypot(rot[i, 0] - rot[j, 0], rot[i, 1] - rot[j, 1])

    # Extension check: tip is far above PIP joint in rotated upright coordinates
    idx_ext = (rot[8, 1] < rot[6, 1] - 0.12 * palm_len) and (d2(0, 8) > d2(0, 6) * 1.1)
    mid_ext = (rot[12, 1] < rot[10, 1] - 0.12 * palm_len) and (d2(0, 12) > d2(0, 10) * 1.1)
    ring_ext = (rot[16, 1] < rot[14, 1] - 0.12 * palm_len) and (d2(0, 16) > d2(0, 14) * 1.1)
    pinky_ext = (rot[20, 1] < rot[18, 1] - 0.12 * palm_len) and (d2(0, 20) > d2(0, 18) * 1.1)

    idx_curl = not idx_ext
    mid_curl = not mid_ext
    ring_curl = not ring_ext
    pinky_curl = not pinky_ext

    # Hooked index (X): PIP is extended far up, but tip curls down
    idx_pip_up = rot[6, 1] < rot[5, 1] - 0.30 * palm_len
    idx_tip_down = rot[8, 1] > rot[6, 1] + 0.05 * palm_len
    idx_hooked = idx_pip_up and idx_tip_down and mid_curl and ring_curl and pinky_curl

    # Thumb states:
    thumb_out = abs(rot[4, 0] - rot[9, 0]) > 0.45 * palm_len
    thumb_up = (rot[4, 1] < rot[2, 1] - 0.10 * palm_len) and (rot[4, 0] < rot[5, 0] + 0.05 * palm_len)
    thumb_touch_idx = d(4, 8) < 0.28 * palm_len
    thumb_touch_mid = d(4, 12) < 0.32 * palm_len

    # Inter-finger distances
    d_idx_mid = d2(8, 12) / palm_len
    crossed_idx_mid = ((rot[8, 0] - rot[12, 0]) * (rot[5, 0] - rot[9, 0])) < 0

    is_pointing_down = (raw_angle_deg > 30) and (raw_angle_deg < 150)
    is_pointing_side = abs(raw_angle_deg) < 40 or abs(raw_angle_deg) > 140

    # ------------------ RULES ------------------

    # Hooked Index -> 'X'
    if idx_hooked:
        return 'X', 0.98

    # Crossed Index & Middle -> 'R'
    if idx_ext and mid_ext and ring_curl and pinky_curl and (crossed_idx_mid or d_idx_mid < 0.15):
        return 'R', 0.98

    # 4 Fingers Extended -> 'B'
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
        if is_pointing_down: return 'P', 0.96
        if is_pointing_side: return 'H', 0.96
        if d_idx_mid > 0.30: return 'V', 0.99
        else: return 'U', 0.98

    # 2 Fingers Extended (Thumb & Pinky) -> 'Y'
    if thumb_out and pinky_ext and idx_curl and mid_curl and ring_curl:
        return 'Y', 0.99

    # 2 Fingers Extended (Thumb & Index) -> 'L'
    if thumb_out and idx_ext and mid_curl and ring_curl and pinky_curl:
        return 'L', 0.99

    # 1 Finger Extended (Pinky only) -> 'I' / 'J'
    if pinky_ext and idx_curl and mid_curl and ring_curl:
        return 'I', 0.99

    # 1 Finger Extended (Index only)
    if idx_ext and mid_curl and ring_curl and pinky_curl:
        if is_pointing_down: return 'Q', 0.96
        if is_pointing_side: return 'G', 0.97
        if thumb_touch_mid: return 'D', 0.98
        return 'D', 0.96

    # Closed Circle 'O'
    if thumb_touch_idx and thumb_touch_mid and idx_curl and mid_curl:
        return 'O', 0.97

    # Curved Hand 'C'
    d_thumb_idx = d(4, 8) / palm_len
    if not idx_ext and d_thumb_idx > 0.35 and d_thumb_idx < 0.80 and rot[4, 1] < rot[0, 1] - 0.20 * palm_len:
        return 'C', 0.95

    # All 4 Fingers Curled
    if idx_curl and mid_curl and ring_curl and pinky_curl:
        if thumb_up:
            return 'A', 0.98
        # Thumb position
        thumb_tip_x = rot[4, 0]
        if abs(thumb_tip_x - rot[13, 0]) < 0.20 * palm_len:
            return 'M', 0.93
        elif abs(thumb_tip_x - rot[9, 0]) < 0.20 * palm_len:
            return 'N', 0.93
        elif abs(thumb_tip_x - rot[5, 0]) < 0.20 * palm_len:
            return 'T', 0.93
        elif rot[4, 1] > rot[6, 1]:
            return 'E', 0.94
        else:
            return 'S', 0.96

    return None, 0.0

rng = np.random.RandomState(42)
correct = 0
total = 0
for letter in ALPHABETS:
    matches = 0
    for _ in range(20):
        lm = generate_asl_landmarks(letter, rng)
        pred, conf = classify_hand_all26_v2(lm)
        if pred == letter:
            matches += 1
    total += 20
    correct += matches
    status = "✓" if matches >= 15 else "✗"
    print(f"{status} {letter:<6} {matches}/20 ({matches/20*100:.0f}%)")

print(f"\nOverall Geometric ASL Recognition: {correct}/{total} ({correct/total*100:.1f}%)")
