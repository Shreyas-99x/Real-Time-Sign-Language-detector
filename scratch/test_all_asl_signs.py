import json
import numpy as np

with open('web/model/model.json', 'r') as f:
    m = json.load(f)

weights = [np.array(w, dtype=np.float32) for w in m['weights']]
biases = [np.array(b, dtype=np.float32) for b in m['biases']]
classes = m['classes']

def forward(vec):
    curr = np.array(vec, dtype=np.float32)
    for i in range(len(weights)):
        out = np.dot(curr, weights[i]) + biases[i]
        curr = np.maximum(0, out) if i < len(weights)-1 else out
    e = np.exp(curr - np.max(curr))
    probs = e / np.sum(e)
    top_indices = np.argsort(probs)[::-1]
    return classes[top_indices[0]], probs[top_indices[0]], [(classes[i], probs[i]) for i in top_indices[:3]]

# Let's define the 21 landmarks of standard ASL letters.
# Wrist is at (0.5, 0.75, 0.0)
# Hands are held upright in front of webcam:
# Fingers extend upwards (y decreases)
def create_asl_landmarks(thumb_state, index_state, mid_state, ring_state, pinky_state, thumb_across=False):
    # state: 1.0 = fully extended up, 0.0 = curled into fist, 0.5 = curved (C/O)
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.5, 0.75, 0.0] # wrist
    
    # Base MCPs
    lm[1] = [0.46, 0.70, -0.01] # thumb CMC
    lm[2] = [0.43, 0.65, -0.02] # thumb MCP
    lm[5] = [0.47, 0.55, -0.03] # index MCP
    lm[9] = [0.50, 0.54, -0.03] # mid MCP
    lm[13] = [0.53, 0.55, -0.03] # ring MCP
    lm[17] = [0.56, 0.58, -0.03] # pinky MCP

    # Thumb
    if thumb_state == 'up':
        lm[3] = [0.41, 0.58, -0.02]
        lm[4] = [0.40, 0.50, -0.02]
    elif thumb_state == 'out': # L shape
        lm[3] = [0.36, 0.65, -0.01]
        lm[4] = [0.30, 0.65, 0.0]
    elif thumb_state == 'across': # A or S shape
        lm[3] = [0.46, 0.60, 0.02]
        lm[4] = [0.48, 0.58, 0.03]
    elif thumb_state == 'tucked':
        lm[3] = [0.48, 0.62, 0.02]
        lm[4] = [0.50, 0.60, 0.03]
    else: # curled
        lm[3] = [0.45, 0.65, 0.01]
        lm[4] = [0.47, 0.65, 0.02]

    # Helper for 4 fingers
    fingers = [
        (5, index_state, 0.47),
        (9, mid_state, 0.50),
        (13, ring_state, 0.53),
        (17, pinky_state, 0.56)
    ]
    
    for mcp_idx, state, x_pos in fingers:
        mcp_y = lm[mcp_idx][1]
        if state == 1.0: # fully extended
            lm[mcp_idx + 1] = [x_pos, mcp_y - 0.08, -0.03] # PIP
            lm[mcp_idx + 2] = [x_pos, mcp_y - 0.15, -0.04] # DIP
            lm[mcp_idx + 3] = [x_pos, mcp_y - 0.22, -0.05] # TIP
        elif state == 0.0: # curled into palm
            lm[mcp_idx + 1] = [x_pos, mcp_y - 0.04, 0.01]
            lm[mcp_idx + 2] = [x_pos, mcp_y + 0.02, 0.03]
            lm[mcp_idx + 3] = [x_pos, mcp_y + 0.06, 0.04] # tip touches palm
        elif state == 0.5: # curved (C/O)
            lm[mcp_idx + 1] = [x_pos, mcp_y - 0.06, 0.01]
            lm[mcp_idx + 2] = [x_pos + 0.04, mcp_y - 0.04, 0.03]
            lm[mcp_idx + 3] = [x_pos + 0.06, mcp_y + 0.01, 0.04]
        elif state == -1.0: # pointing down/hooked (X)
            lm[mcp_idx + 1] = [x_pos, mcp_y - 0.06, -0.01]
            lm[mcp_idx + 2] = [x_pos, mcp_y - 0.02, 0.02]
            lm[mcp_idx + 3] = [x_pos, mcp_y + 0.02, 0.03]

    wrist = lm[0]
    rel = lm - wrist
    maxd = np.max(np.linalg.norm(rel, axis=1))
    return (rel / maxd).flatten()

asl_tests = {
    'A (Fist, thumb beside)': ('across', 0.0, 0.0, 0.0, 0.0),
    'B (4 fingers up, thumb in)': ('curled', 1.0, 1.0, 1.0, 1.0),
    'C (Curved fingers)': ('curled', 0.5, 0.5, 0.5, 0.5),
    'D (Index up, others O)': ('curled', 1.0, 0.0, 0.0, 0.0),
    'I (Pinky up)': ('across', 0.0, 0.0, 0.0, 1.0),
    'L (Index & thumb out)': ('out', 1.0, 0.0, 0.0, 0.0),
    'V (Index & mid up)': ('across', 1.0, 1.0, 0.0, 0.0),
    'W (Index, mid, ring up)': ('across', 1.0, 1.0, 1.0, 0.0),
    'Y (Thumb & pinky out)': ('out', 0.0, 0.0, 0.0, 1.0),
}

print(f"{'Gesture':<28} {'Pred':<6} {'Conf':<8} {'Top 3'}")
for name, params in asl_tests.items():
    vec = create_asl_landmarks(*params)
    pred, conf, top3 = forward(vec)
    top3_str = ', '.join([f'{c}: {p*100:.1f}%' for c, p in top3])
    print(f"{name:<28} {pred:<6} {conf*100:<7.1f}% {top3_str}")
