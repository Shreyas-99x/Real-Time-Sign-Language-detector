import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

ALPHABETS = [chr(c) for c in range(ord('A'), ord('Z') + 1)]

def generate_asl_landmarks(letter, rng):
    """
    Generates a realistic 21x3 MediaPipe hand landmark array for any ASL letter
    with natural human hand anatomical variations.
    """
    lm = np.zeros((21, 3), dtype=np.float32)
    # Hand base dimensions
    palm_w = rng.uniform(0.18, 0.24)
    palm_h = rng.uniform(0.28, 0.36)
    
    # 0: Wrist
    lm[0] = [0.0, 0.0, 0.0]
    
    # Base MCPs
    lm[1] = [-palm_w * 0.6, -palm_h * 0.25, rng.uniform(-0.02, 0.0)]
    lm[2] = [-palm_w * 0.9, -palm_h * 0.50, rng.uniform(-0.03, 0.0)]
    lm[5] = [-palm_w * 0.45, -palm_h * 0.90, rng.uniform(-0.02, 0.01)]
    lm[9] = [0.0,            -palm_h * 0.95, rng.uniform(-0.01, 0.02)]
    lm[13] = [palm_w * 0.45, -palm_h * 0.90, rng.uniform(-0.02, 0.01)]
    lm[17] = [palm_w * 0.85, -palm_h * 0.80, rng.uniform(-0.03, 0.0)]

    # Helper: extended finger
    def ext_finger(base_mcp, x_dir, y_len, spread_x=0.0):
        mcp_pt = lm[base_mcp]
        seg1 = y_len * 0.38
        seg2 = y_len * 0.32
        seg3 = y_len * 0.30
        
        lm[base_mcp + 1] = [mcp_pt[0] + spread_x * 0.3, mcp_pt[1] - seg1, rng.uniform(-0.02, 0.01)]
        lm[base_mcp + 2] = [mcp_pt[0] + spread_x * 0.65, mcp_pt[1] - (seg1 + seg2), rng.uniform(-0.03, 0.01)]
        lm[base_mcp + 3] = [mcp_pt[0] + spread_x, mcp_pt[1] - (seg1 + seg2 + seg3), rng.uniform(-0.04, 0.0)]

    # Helper: curled finger into palm
    def curl_finger(base_mcp, tuck_x=0.0):
        mcp_pt = lm[base_mcp]
        lm[base_mcp + 1] = [mcp_pt[0] + tuck_x * 0.3, mcp_pt[1] - palm_h * 0.18, rng.uniform(0.04, 0.08)]
        lm[base_mcp + 2] = [mcp_pt[0] + tuck_x * 0.6, mcp_pt[1] + palm_h * 0.05, rng.uniform(0.08, 0.14)]
        lm[base_mcp + 3] = [mcp_pt[0] + tuck_x,       mcp_pt[1] + palm_h * 0.18, rng.uniform(0.05, 0.10)]

    # Helper: curved finger (C or O)
    def curve_finger(base_mcp, curve_x=0.0):
        mcp_pt = lm[base_mcp]
        lm[base_mcp + 1] = [mcp_pt[0] + curve_x * 0.4, mcp_pt[1] - palm_h * 0.22, rng.uniform(0.02, 0.06)]
        lm[base_mcp + 2] = [mcp_pt[0] + curve_x * 0.8, mcp_pt[1] - palm_h * 0.15, rng.uniform(0.08, 0.15)]
        lm[base_mcp + 3] = [mcp_pt[0] + curve_x,       mcp_pt[1] - palm_h * 0.02, rng.uniform(0.12, 0.20)]

    # Helper: hooked finger (X)
    def hook_finger(base_mcp):
        mcp_pt = lm[base_mcp]
        lm[base_mcp + 1] = [mcp_pt[0], mcp_pt[1] - palm_h * 0.25, rng.uniform(0.01, 0.04)]
        lm[base_mcp + 2] = [mcp_pt[0], mcp_pt[1] - palm_h * 0.12, rng.uniform(0.08, 0.14)]
        lm[base_mcp + 3] = [mcp_pt[0], mcp_pt[1] + palm_h * 0.02, rng.uniform(0.06, 0.10)]

    # Thumb helpers
    def thumb_up_alongside():
        lm[3] = [-palm_w * 0.8, -palm_h * 0.72, rng.uniform(0.0, 0.04)]
        lm[4] = [-palm_w * 0.7, -palm_h * 0.95, rng.uniform(0.02, 0.06)]

    def thumb_out_90():
        lm[3] = [-palm_w * 1.5, -palm_h * 0.45, rng.uniform(-0.02, 0.02)]
        lm[4] = [-palm_w * 2.1, -palm_h * 0.42, rng.uniform(-0.02, 0.02)]

    def thumb_across_front():
        lm[3] = [0.0, palm_h * 0.05, rng.uniform(0.10, 0.15)]
        lm[4] = [palm_w * 0.4, palm_h * 0.08, rng.uniform(0.12, 0.18)]

    def thumb_tuck_under(target_x):
        lm[3] = [target_x * 0.7, -palm_h * 0.40, rng.uniform(0.04, 0.08)]
        lm[4] = [target_x,       -palm_h * 0.45, rng.uniform(0.02, 0.06)]

    def thumb_touch(target_pt):
        lm[3] = [(lm[2][0] + target_pt[0]) * 0.5, (lm[2][1] + target_pt[1]) * 0.5, rng.uniform(0.05, 0.10)]
        lm[4] = [target_pt[0] + rng.uniform(-0.02, 0.02), target_pt[1] + rng.uniform(-0.02, 0.02), target_pt[2] + rng.uniform(0.01, 0.03)]

    # 26 Letters definitions
    f_len = palm_h * rng.uniform(0.70, 0.85)

    if letter == 'A':
        curl_finger(5); curl_finger(9); curl_finger(13); curl_finger(17)
        thumb_up_alongside()
    elif letter == 'B':
        ext_finger(5, 0, f_len, spread_x=0.0)
        ext_finger(9, 0, f_len * 1.05, spread_x=0.0)
        ext_finger(13, 0, f_len * 0.95, spread_x=0.0)
        ext_finger(17, 0, f_len * 0.80, spread_x=0.0)
        thumb_across_front()
    elif letter == 'C':
        curve_finger(5, 0.02); curve_finger(9, 0.0); curve_finger(13, -0.02); curve_finger(17, -0.04)
        lm[3] = [-palm_w * 0.9, -palm_h * 0.40, 0.08]
        lm[4] = [-palm_w * 0.6, -palm_h * 0.20, 0.14]
    elif letter == 'D':
        ext_finger(5, 0, f_len)
        curve_finger(9); curve_finger(13); curve_finger(17)
        thumb_touch(lm[12])
    elif letter == 'E':
        for b in [5, 9, 13, 17]:
            mcp = lm[b]
            lm[b+1] = [mcp[0], mcp[1] - palm_h*0.15, 0.06]
            lm[b+2] = [mcp[0], mcp[1] - palm_h*0.05, 0.12]
            lm[b+3] = [mcp[0], mcp[1] + palm_h*0.05, 0.10]
        thumb_across_front()
    elif letter == 'F':
        curve_finger(5)
        ext_finger(9, 0, f_len * 1.05, spread_x=0.02)
        ext_finger(13, 0, f_len * 0.95, spread_x=0.04)
        ext_finger(17, 0, f_len * 0.80, spread_x=0.06)
        thumb_touch(lm[8])
    elif letter == 'G':
        # Pointing sideways
        lm[6] = [palm_w * 0.5, lm[5][1], 0.0]
        lm[7] = [palm_w * 1.0, lm[5][1], 0.0]
        lm[8] = [palm_w * 1.5, lm[5][1], 0.0]
        curl_finger(9); curl_finger(13); curl_finger(17)
        lm[3] = [palm_w * 0.3, lm[5][1] + palm_h * 0.12, 0.0]
        lm[4] = [palm_w * 0.8, lm[5][1] + palm_h * 0.12, 0.0]
    elif letter == 'H':
        # Index & mid pointing sideways
        lm[6] = [palm_w * 0.5, lm[5][1], 0.0]; lm[7] = [palm_w * 1.0, lm[5][1], 0.0]; lm[8] = [palm_w * 1.5, lm[5][1], 0.0]
        lm[10] = [palm_w * 0.5, lm[5][1] + palm_h * 0.08, 0.0]; lm[11] = [palm_w * 1.0, lm[5][1] + palm_h * 0.08, 0.0]; lm[12] = [palm_w * 1.5, lm[5][1] + palm_h * 0.08, 0.0]
        curl_finger(13); curl_finger(17)
        thumb_across_front()
    elif letter == 'I':
        curl_finger(5); curl_finger(9); curl_finger(13)
        ext_finger(17, 0, f_len * 0.80)
        thumb_across_front()
    elif letter == 'J':
        curl_finger(5); curl_finger(9); curl_finger(13)
        ext_finger(17, 0, f_len * 0.80, spread_x=palm_w * 0.3)
        thumb_across_front()
    elif letter == 'K':
        ext_finger(5, 0, f_len)
        lm[10] = [lm[9][0], lm[9][1] - f_len * 0.35, 0.10]
        lm[11] = [lm[9][0], lm[9][1] - f_len * 0.70, 0.20]
        lm[12] = [lm[9][0], lm[9][1] - f_len * 1.00, 0.30]
        curl_finger(13); curl_finger(17)
        thumb_up_alongside()
    elif letter == 'L':
        ext_finger(5, 0, f_len)
        curl_finger(9); curl_finger(13); curl_finger(17)
        thumb_out_90()
    elif letter == 'M':
        for b in [5, 9, 13]:
            mcp = lm[b]
            lm[b+1] = [mcp[0], mcp[1] - palm_h*0.20, 0.06]
            lm[b+2] = [mcp[0], mcp[1] + palm_h*0.02, 0.12]
            lm[b+3] = [mcp[0], mcp[1] + palm_h*0.16, 0.08]
        curl_finger(17)
        thumb_tuck_under(lm[13][0])
    elif letter == 'N':
        for b in [5, 9]:
            mcp = lm[b]
            lm[b+1] = [mcp[0], mcp[1] - palm_h*0.20, 0.06]
            lm[b+2] = [mcp[0], mcp[1] + palm_h*0.02, 0.12]
            lm[b+3] = [mcp[0], mcp[1] + palm_h*0.16, 0.08]
        curl_finger(13); curl_finger(17)
        thumb_tuck_under(lm[9][0])
    elif letter == 'O':
        for b in [5, 9, 13, 17]:
            curve_finger(b)
        thumb_touch(lm[8])
    elif letter == 'P':
        # Like K pointing down
        mcp5 = lm[5]; mcp9 = lm[9]
        lm[6] = [mcp5[0], mcp5[1] + f_len * 0.35, 0.0]; lm[7] = [mcp5[0], mcp5[1] + f_len * 0.70, 0.0]; lm[8] = [mcp5[0], mcp5[1] + f_len * 1.0, 0.0]
        lm[10] = [mcp9[0] + palm_w*0.3, mcp9[1] + f_len * 0.3, 0.08]; lm[11] = [mcp9[0] + palm_w*0.6, mcp9[1] + f_len * 0.6, 0.15]; lm[12] = [mcp9[0] + palm_w*0.9, mcp9[1] + f_len * 0.9, 0.22]
        curl_finger(13); curl_finger(17)
        thumb_out_90()
    elif letter == 'Q':
        # G pointing down
        mcp5 = lm[5]
        lm[6] = [mcp5[0], mcp5[1] + f_len * 0.35, 0.0]; lm[7] = [mcp5[0], mcp5[1] + f_len * 0.70, 0.0]; lm[8] = [mcp5[0], mcp5[1] + f_len * 1.0, 0.0]
        curl_finger(9); curl_finger(13); curl_finger(17)
        lm[3] = [lm[2][0], lm[2][1] + f_len * 0.3, 0.0]; lm[4] = [lm[2][0], lm[2][1] + f_len * 0.7, 0.0]
    elif letter == 'R':
        # Crossed index & middle
        mcp5 = lm[5]; mcp9 = lm[9]
        lm[6] = [mcp5[0] + palm_w*0.1, mcp5[1] - f_len*0.35, -0.02]
        lm[7] = [mcp5[0] + palm_w*0.25, mcp5[1] - f_len*0.70, -0.02]
        lm[8] = [mcp5[0] + palm_w*0.4, mcp5[1] - f_len*1.05, -0.02]
        
        lm[10] = [mcp9[0] - palm_w*0.1, mcp9[1] - f_len*0.35, 0.02]
        lm[11] = [mcp9[0] - palm_w*0.25, mcp9[1] - f_len*0.70, 0.02]
        lm[12] = [mcp9[0] - palm_w*0.4, mcp9[1] - f_len*1.05, 0.02]
        curl_finger(13); curl_finger(17)
        thumb_across_front()
    elif letter == 'S':
        curl_finger(5); curl_finger(9); curl_finger(13); curl_finger(17)
        thumb_across_front()
    elif letter == 'T':
        curl_finger(5); curl_finger(9); curl_finger(13); curl_finger(17)
        thumb_tuck_under(lm[5][0])
    elif letter == 'U':
        # Index & mid touching tight together
        ext_finger(5, 0, f_len, spread_x=palm_w * 0.05)
        ext_finger(9, 0, f_len, spread_x=-palm_w * 0.05)
        curl_finger(13); curl_finger(17)
        thumb_across_front()
    elif letter == 'V':
        # Index & mid spread wide
        ext_finger(5, 0, f_len, spread_x=-palm_w * 0.35)
        ext_finger(9, 0, f_len, spread_x=palm_w * 0.35)
        curl_finger(13); curl_finger(17)
        thumb_across_front()
    elif letter == 'W':
        ext_finger(5, 0, f_len, spread_x=-palm_w * 0.40)
        ext_finger(9, 0, f_len * 1.05, spread_x=0.0)
        ext_finger(13, 0, f_len, spread_x=palm_w * 0.40)
        curl_finger(17)
        thumb_touch(lm[17])
    elif letter == 'X':
        hook_finger(5)
        curl_finger(9); curl_finger(13); curl_finger(17)
        thumb_across_front()
    elif letter == 'Y':
        curl_finger(5); curl_finger(9); curl_finger(13)
        ext_finger(17, 0, f_len * 0.85, spread_x=palm_w * 0.5)
        thumb_out_90()
    elif letter == 'Z':
        ext_finger(5, 0, f_len, spread_x=0.0)
        curl_finger(9); curl_finger(13); curl_finger(17)
        thumb_across_front()

    # Canonical rotation
    wrist = lm[0]
    mid_mcp = lm[9]
    dx = mid_mcp[0] - wrist[0]
    dy = mid_mcp[1] - wrist[1]
    angle = np.arctan2(dy, dx)
    rot_angle = -np.pi / 2.0 - angle
    cos_t = np.cos(rot_angle)
    sin_t = np.sin(rot_angle)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    
    rel = lm - wrist
    rot = rel.copy()
    rot[:, :2] = np.dot(rel[:, :2], R.T)
    max_d = np.max(np.linalg.norm(rot, axis=1))
    if max_d < 1e-6: max_d = 1.0
    return rot / max_d

print("ASL generator ready.")
