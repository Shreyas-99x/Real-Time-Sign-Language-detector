import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

ALPHABETS = [chr(c) for c in range(ord('A'), ord('Z') + 1)]

def canonical_normalize(landmarks):
    wrist = landmarks[0]
    mid_mcp = landmarks[9]
    dx = mid_mcp[0] - wrist[0]
    dy = mid_mcp[1] - wrist[1]
    angle = np.arctan2(dy, dx)
    rot_angle = -np.pi / 2.0 - angle
    cos_t = np.cos(rot_angle)
    sin_t = np.sin(rot_angle)
    
    rel = landmarks - wrist
    rot_x = rel[:, 0] * cos_t - rel[:, 1] * sin_t
    rot_y = rel[:, 0] * sin_t + rel[:, 1] * cos_t
    rot_z = rel[:, 2]
    
    rot = np.column_stack([rot_x, rot_y, rot_z])
    max_d = np.max(np.linalg.norm(rot, axis=1))
    if max_d < 1e-6:
        max_d = 1.0
    return (rot / max_d).flatten()

def generate_hand(letter, is_left, rng):
    """
    Builds a high-fidelity 21-joint 3D MediaPipe hand model for all 26 ASL alphabet signs.
    """
    lm = np.zeros((21, 3), dtype=np.float32)
    palm_w = rng.uniform(0.18, 0.23)
    palm_h = rng.uniform(0.28, 0.35)
    f_len = palm_h * rng.uniform(0.72, 0.85)
    
    # Sign of X: for right hand, thumb is on -X side. For left hand, thumb is on +X side.
    h_sign = -1.0 if not is_left else 1.0
    
    # 0: Wrist
    lm[0] = [0.0, 0.0, 0.0]
    
    # Base MCPs (1-4: Thumb, 5-8: Index, 9-12: Mid, 13-16: Ring, 17-20: Pinky)
    lm[1] = [h_sign * palm_w * 0.55, -palm_h * 0.25, rng.uniform(-0.02, 0.01)]
    lm[2] = [h_sign * palm_w * 0.85, -palm_h * 0.48, rng.uniform(-0.03, 0.01)]
    
    lm[5]  = [h_sign * palm_w * 0.45, -palm_h * 0.90, rng.uniform(-0.01, 0.01)]
    lm[9]  = [0.0,                    -palm_h * 0.95, rng.uniform(-0.01, 0.01)]
    lm[13] = [-h_sign * palm_w * 0.45, -palm_h * 0.90, rng.uniform(-0.01, 0.01)]
    lm[17] = [-h_sign * palm_w * 0.85, -palm_h * 0.80, rng.uniform(-0.02, 0.01)]
    
    def ext_finger(base, spread_x=0.0, length_mult=1.0):
        mcp = lm[base]
        l = f_len * length_mult
        lm[base + 1] = [mcp[0] + spread_x * 0.3, mcp[1] - l * 0.38, rng.uniform(-0.02, 0.01)]
        lm[base + 2] = [mcp[0] + spread_x * 0.65, mcp[1] - l * 0.70, rng.uniform(-0.03, 0.01)]
        lm[base + 3] = [mcp[0] + spread_x,        mcp[1] - l * 1.00, rng.uniform(-0.04, 0.01)]
        
    def curl_finger(base, tuck_x=0.0):
        mcp = lm[base]
        lm[base + 1] = [mcp[0] + tuck_x * 0.3, mcp[1] - palm_h * 0.18, rng.uniform(0.04, 0.08)]
        lm[base + 2] = [mcp[0] + tuck_x * 0.6, mcp[1] + palm_h * 0.05, rng.uniform(0.08, 0.14)]
        lm[base + 3] = [mcp[0] + tuck_x,       mcp[1] + palm_h * 0.18, rng.uniform(0.05, 0.10)]
        
    def curve_finger(base, curve_x=0.0):
        mcp = lm[base]
        lm[base + 1] = [mcp[0] + curve_x * 0.4, mcp[1] - palm_h * 0.22, rng.uniform(0.02, 0.06)]
        lm[base + 2] = [mcp[0] + curve_x * 0.8, mcp[1] - palm_h * 0.15, rng.uniform(0.08, 0.14)]
        lm[base + 3] = [mcp[0] + curve_x,       mcp[1] - palm_h * 0.02, rng.uniform(0.12, 0.18)]
        
    def hook_finger(base):
        mcp = lm[base]
        lm[base + 1] = [mcp[0], mcp[1] - palm_h * 0.25, rng.uniform(0.01, 0.04)]
        lm[base + 2] = [mcp[0], mcp[1] - palm_h * 0.12, rng.uniform(0.08, 0.14)]
        lm[base + 3] = [mcp[0], mcp[1] + palm_h * 0.02, rng.uniform(0.06, 0.10)]
        
    def thumb_up_alongside():
        lm[3] = [h_sign * palm_w * 0.75, -palm_h * 0.70, rng.uniform(0.0, 0.04)]
        lm[4] = [h_sign * palm_w * 0.65, -palm_h * 0.95, rng.uniform(0.02, 0.06)]
        
    def thumb_out_90():
        lm[3] = [h_sign * palm_w * 1.5, -palm_h * 0.45, rng.uniform(-0.02, 0.02)]
        lm[4] = [h_sign * palm_w * 2.1, -palm_h * 0.42, rng.uniform(-0.02, 0.02)]
        
    def thumb_across_front():
        lm[3] = [0.0, -palm_h * 0.35, rng.uniform(0.08, 0.13)]
        lm[4] = [-h_sign * palm_w * 0.35, -palm_h * 0.42, rng.uniform(0.10, 0.15)]
        
    def thumb_tuck_under(target_x, depth_y=-0.40):
        lm[3] = [target_x * 0.7, -palm_h * (depth_y - 0.05), rng.uniform(0.04, 0.08)]
        lm[4] = [target_x,       -palm_h * depth_y,          rng.uniform(0.02, 0.06)]
        
    def thumb_touch(target_pt):
        lm[3] = [(lm[2][0] + target_pt[0]) * 0.5, (lm[2][1] + target_pt[1]) * 0.5, rng.uniform(0.05, 0.10)]
        lm[4] = [target_pt[0] + rng.uniform(-0.02, 0.02), target_pt[1] + rng.uniform(-0.02, 0.02), target_pt[2] + rng.uniform(0.01, 0.03)]

    # 26 Pure ASL Sign Formations
    if letter == 'A':
        curl_finger(5); curl_finger(9); curl_finger(13); curl_finger(17)
        thumb_up_alongside()
    elif letter == 'B':
        ext_finger(5, 0, 1.0)
        ext_finger(9, 0, 1.05)
        ext_finger(13, 0, 0.95)
        ext_finger(17, 0, 0.80)
        lm[3] = [h_sign * palm_w * 0.35, -palm_h * 0.35, 0.08]
        lm[4] = [0.0, -palm_h * 0.38, 0.10]
    elif letter == 'C':
        curve_finger(5, 0.02); curve_finger(9, 0.0); curve_finger(13, -0.02); curve_finger(17, -0.04)
        lm[3] = [h_sign * palm_w * 0.85, -palm_h * 0.40, 0.08]
        lm[4] = [h_sign * palm_w * 0.55, -palm_h * 0.20, 0.14]
    elif letter == 'D':
        ext_finger(5, 0, 1.0)
        curve_finger(9); curve_finger(13); curve_finger(17)
        thumb_touch(lm[12])
    elif letter == 'E':
        for b in [5, 9, 13, 17]:
            mcp = lm[b]
            lm[b+1] = [mcp[0], mcp[1] - palm_h*0.20, 0.05]
            lm[b+2] = [mcp[0], mcp[1] - palm_h*0.10, 0.10]
            lm[b+3] = [mcp[0], mcp[1] + palm_h*0.02, 0.08]
        lm[3] = [0.0, -palm_h * 0.25, 0.06]
        lm[4] = [h_sign * palm_w * 0.15, -palm_h * 0.28, 0.04]
    elif letter == 'F':
        curve_finger(5)
        ext_finger(9, 0, 1.05)
        ext_finger(13, 0, 0.95)
        ext_finger(17, 0, 0.80)
        thumb_touch(lm[8])
    elif letter == 'G':
        ext_finger(5, spread_x=0.0, length_mult=1.0)
        lm[3] = [h_sign * palm_w * 0.85, -palm_h * 0.65, 0.02]
        lm[4] = [h_sign * palm_w * 0.80, -palm_h * 0.90, 0.03]
        curl_finger(9); curl_finger(13); curl_finger(17)
    elif letter == 'H':
        lm[6] = [lm[5][0] + h_sign * palm_w * 0.4, lm[5][1], 0.0]; lm[7] = [lm[5][0] + h_sign * palm_w * 0.8, lm[5][1], 0.0]; lm[8] = [lm[5][0] + h_sign * palm_w * 1.3, lm[5][1], 0.0]
        lm[10] = [lm[9][0] + h_sign * palm_w * 0.4, lm[9][1], 0.0]; lm[11] = [lm[9][0] + h_sign * palm_w * 0.8, lm[9][1], 0.0]; lm[12] = [lm[9][0] + h_sign * palm_w * 1.3, lm[9][1], 0.0]
        curl_finger(13); curl_finger(17)
        thumb_across_front()
    elif letter == 'I':
        curl_finger(5); curl_finger(9); curl_finger(13)
        ext_finger(17, 0, 0.80)
        thumb_across_front()
    elif letter == 'J':
        curl_finger(5); curl_finger(9); curl_finger(13)
        ext_finger(17, -h_sign * palm_w * 0.30, 0.80)
        thumb_across_front()
    elif letter == 'K':
        ext_finger(5, 0, 1.0)
        lm[10] = [lm[9][0], lm[9][1] - f_len * 0.35, 0.08]
        lm[11] = [lm[9][0], lm[9][1] - f_len * 0.70, 0.16]
        lm[12] = [lm[9][0], lm[9][1] - f_len * 0.95, 0.24]
        curl_finger(13); curl_finger(17)
        lm[3] = [lm[5][0], -palm_h * 0.60, 0.08]
        lm[4] = [lm[9][0], -palm_h * 0.70, 0.14]
    elif letter == 'L':
        ext_finger(5, 0, 1.0)
        curl_finger(9); curl_finger(13); curl_finger(17)
        thumb_out_90()
    elif letter == 'M':
        for b in [5, 9, 13]:
            mcp = lm[b]
            lm[b+1] = [mcp[0], mcp[1] - palm_h*0.18, 0.06]
            lm[b+2] = [mcp[0], mcp[1] + palm_h*0.02, 0.12]
            lm[b+3] = [mcp[0], mcp[1] + palm_h*0.16, 0.08]
        curl_finger(17)
        thumb_tuck_under(-h_sign * palm_w * 0.55, depth_y=-0.38)
    elif letter == 'N':
        for b in [5, 9]:
            mcp = lm[b]
            lm[b+1] = [mcp[0], mcp[1] - palm_h*0.18, 0.06]
            lm[b+2] = [mcp[0], mcp[1] + palm_h*0.02, 0.12]
            lm[b+3] = [mcp[0], mcp[1] + palm_h*0.16, 0.08]
        curl_finger(13); curl_finger(17)
        thumb_tuck_under(-h_sign * palm_w * 0.15, depth_y=-0.38)
    elif letter == 'O':
        for b in [5, 9, 13, 17]:
            curve_finger(b)
        thumb_touch(lm[8])
    elif letter == 'P':
        mcp5 = lm[5]; mcp9 = lm[9]
        lm[6] = [mcp5[0], mcp5[1] + f_len * 0.35, 0.0]; lm[7] = [mcp5[0], mcp5[1] + f_len * 0.70, 0.0]; lm[8] = [mcp5[0], mcp5[1] + f_len * 1.0, 0.0]
        lm[10] = [mcp9[0], mcp9[1] + f_len * 0.35, 0.10]; lm[11] = [mcp9[0], mcp9[1] + f_len * 0.70, 0.18]; lm[12] = [mcp9[0], mcp9[1] + f_len * 0.95, 0.25]
        curl_finger(13); curl_finger(17)
        thumb_out_90()
    elif letter == 'Q':
        mcp5 = lm[5]
        lm[6] = [mcp5[0], mcp5[1] + f_len * 0.35, 0.0]; lm[7] = [mcp5[0], mcp5[1] + f_len * 0.70, 0.0]; lm[8] = [mcp5[0], mcp5[1] + f_len * 1.0, 0.0]
        curl_finger(9); curl_finger(13); curl_finger(17)
        lm[3] = [lm[2][0], lm[2][1] + f_len * 0.3, 0.0]; lm[4] = [lm[2][0], lm[2][1] + f_len * 0.7, 0.0]
    elif letter == 'R':
        mcp5 = lm[5]; mcp9 = lm[9]
        lm[6] = [mcp5[0] - h_sign*palm_w*0.1, mcp5[1] - f_len*0.35, -0.02]
        lm[7] = [mcp5[0] - h_sign*palm_w*0.25, mcp5[1] - f_len*0.70, -0.02]
        lm[8] = [mcp5[0] - h_sign*palm_w*0.4, mcp5[1] - f_len*1.05, -0.02]
        lm[10] = [mcp9[0] + h_sign*palm_w*0.1, mcp9[1] - f_len*0.35, 0.02]
        lm[11] = [mcp9[0] + h_sign*palm_w*0.25, mcp9[1] - f_len*0.70, 0.02]
        lm[12] = [mcp9[0] + h_sign*palm_w*0.4, mcp9[1] - f_len*1.05, 0.02]
        curl_finger(13); curl_finger(17)
        thumb_across_front()
    elif letter == 'S':
        curl_finger(5); curl_finger(9); curl_finger(13); curl_finger(17)
        lm[3] = [0.0, -palm_h * 0.35, 0.12]
        lm[4] = [-h_sign * palm_w * 0.25, -palm_h * 0.38, 0.14]
    elif letter == 'T':
        curl_finger(5); curl_finger(9); curl_finger(13); curl_finger(17)
        thumb_tuck_under(h_sign * palm_w * 0.35, depth_y=-0.42)
    elif letter == 'U':
        ext_finger(5, spread_x=-h_sign * palm_w * 0.04, length_mult=1.0)
        ext_finger(9, spread_x=h_sign * palm_w * 0.04, length_mult=1.05)
        curl_finger(13); curl_finger(17)
        thumb_across_front()
    elif letter == 'V':
        ext_finger(5, spread_x=h_sign * palm_w * 0.35, length_mult=1.0)
        ext_finger(9, spread_x=-h_sign * palm_w * 0.35, length_mult=1.05)
        curl_finger(13); curl_finger(17)
        thumb_across_front()
    elif letter == 'W':
        ext_finger(5, spread_x=h_sign * palm_w * 0.38, length_mult=1.0)
        ext_finger(9, spread_x=0.0, length_mult=1.05)
        ext_finger(13, spread_x=-h_sign * palm_w * 0.38, length_mult=0.95)
        curl_finger(17)
        thumb_touch(lm[17])
    elif letter == 'X':
        hook_finger(5)
        curl_finger(9); curl_finger(13); curl_finger(17)
        thumb_across_front()
    elif letter == 'Y':
        curl_finger(5); curl_finger(9); curl_finger(13)
        ext_finger(17, spread_x=-h_sign * palm_w * 0.45, length_mult=0.85)
        thumb_out_90()
    elif letter == 'Z':
        ext_finger(5, 0, 1.0)
        curl_finger(9); curl_finger(13); curl_finger(17)
        thumb_across_front()

    # 3D Camera Rotation Jitter
    roll = rng.uniform(-0.52, 0.52) # +/- 30 deg
    c_r = np.cos(roll); s_r = np.sin(roll)
    R_z = np.array([[c_r, -s_r, 0], [s_r, c_r, 0], [0, 0, 1]], dtype=np.float32)
    
    pitch = rng.uniform(-0.30, 0.30) # +/- 17 deg
    c_p = np.cos(pitch); s_p = np.sin(pitch)
    R_x = np.array([[1, 0, 0], [0, c_p, -s_p], [0, s_p, c_p]], dtype=np.float32)
    
    yaw = rng.uniform(-0.30, 0.30) # +/- 17 deg
    c_y = np.cos(yaw); s_y = np.sin(yaw)
    R_y = np.array([[c_y, 0, s_y], [0, 1, 0], [-s_y, 0, c_y]], dtype=np.float32)
    
    R_comb = np.dot(R_z, np.dot(R_x, R_y))
    lm_rotated = np.dot(lm, R_comb.T)
    
    # MediaPipe detection noise
    noise = rng.normal(0, 0.010, lm.shape).astype(np.float32)
    lm_noisy = lm_rotated + noise
    
    return canonical_normalize(lm_noisy)

if __name__ == '__main__':
    print('Generating high-precision ASL dataset across 26 letters (1,500 samples/class = 39,000 total)...')
    rng = np.random.RandomState(42)
    X = []
    y = []
    
    samples_per_class = 1500
    for cid, letter in enumerate(ALPHABETS):
        for i in range(samples_per_class):
            is_left = (i % 2 == 1)
            vec = generate_hand(letter, is_left, rng)
            X.append(vec)
            y.append(cid)
            
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f'Dataset shape: X={X.shape}, y={y.shape}')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        max_iter=180,
        batch_size=128,
        alpha=0.0001,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    print('Training optimized MLP classifier...')
    mlp.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, mlp.predict(X_test))
    print(f'Test Accuracy: {test_acc * 100:.2f}%')
    
    # Compute KNN centroids
    centroids = {}
    for cid, letter in enumerate(ALPHABETS):
        class_samples = X[y == cid]
        centroids[letter] = np.mean(class_samples, axis=0).tolist()
        
    weights = [w.tolist() for w in mlp.coefs_]
    biases = [b.tolist() for b in mlp.intercepts_]
    
    model_export = {
        'input_dim': 63,
        'layer_sizes': [256, 128, 64, 26],
        'weights': weights,
        'biases': biases,
        'classes': ALPHABETS,
        'accuracy': float(test_acc)
    }
    
    with open('web/model/model.json', 'w') as f:
        json.dump(model_export, f)
        
    with open('web/model/labels.json', 'w') as f:
        json.dump(ALPHABETS, f, indent=2)
        
    with open('web/model/landmarks.json', 'w') as f:
        json.dump({'classes': ALPHABETS, 'centroids': centroids}, f)
        
    print('Successfully exported updated model artifacts to web/model/!')
