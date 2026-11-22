import numpy as np
from test_clean_classifier import classify_asl_hand

# Generate realistic MediaPipe landmark arrays
def make_hand(thumb_state, index_state, mid_state, ring_state, pinky_state):
    # state: True = extended, False = curled
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.5, 0.75, 0.0] # wrist
    
    # MCPs
    lm[1] = [0.46, 0.70, -0.01]
    lm[2] = [0.43, 0.65, -0.02]
    lm[5] = [0.46, 0.55, -0.03]
    lm[9] = [0.50, 0.53, -0.03]
    lm[13] = [0.54, 0.55, -0.03]
    lm[17] = [0.58, 0.58, -0.03]
    
    # Thumb
    if thumb_state == 'out': # L, Y
        lm[3] = [0.38, 0.64, -0.02]
        lm[4] = [0.32, 0.63, -0.02]
    elif thumb_state == 'up': # A
        lm[3] = [0.45, 0.58, -0.01]
        lm[4] = [0.45, 0.52, 0.0]
    elif thumb_state == 'touch_index': # F, O
        lm[3] = [0.44, 0.60, 0.02]
        lm[4] = [0.46, 0.55, 0.02]
    else: # across (fist, B, S)
        lm[3] = [0.48, 0.60, 0.02]
        lm[4] = [0.50, 0.58, 0.03]
        
    fingers = [(5, index_state, 0.46), (9, mid_state, 0.50), (13, ring_state, 0.54), (17, pinky_state, 0.58)]
    for mcp, is_ext, x in fingers:
        mcp_y = lm[mcp][1]
        if is_ext:
            lm[mcp + 1] = [x, mcp_y - 0.07, -0.03]
            lm[mcp + 2] = [x, mcp_y - 0.13, -0.04]
            lm[mcp + 3] = [x, mcp_y - 0.19, -0.05]
        else:
            lm[mcp + 1] = [x, mcp_y - 0.03, 0.02]
            lm[mcp + 2] = [x, mcp_y + 0.02, 0.04]
            lm[mcp + 3] = [x, mcp_y + 0.05, 0.03]
            
    return lm

cases = {
    'A (Fist, thumb up)': ('up', False, False, False, False),
    'B (4 fingers up)': ('across', True, True, True, True),
    'D (Index up, thumb in)': ('across', True, False, False, False),
    'I (Pinky up)': ('across', False, False, False, True),
    'L (Index & Thumb)': ('out', True, False, False, False),
    'V (Index & Mid)': ('across', True, True, False, False),
    'W (Index, Mid, Ring)': ('across', True, True, True, False),
    'Y (Thumb & Pinky)': ('out', False, False, False, True),
    'S (Fist, thumb across)': ('across', False, False, False, False),
}

print(f"{'Gesture':<25} {'Pred':<6} {'Conf':<6}")
for name, params in cases.items():
    lm = make_hand(*params)
    pred, conf = classify_asl_hand(lm)
    print(f"{name:<25} {str(pred):<6} {conf*100:.0f}%")
