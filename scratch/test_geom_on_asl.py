import numpy as np
from test_all_asl_signs import create_asl_landmarks
from test_hand_geometry import analyze_hand_geometry

asl_tests = {
    'A (Fist, thumb beside)': ('across', 0.0, 0.0, 0.0, 0.0),
    'B (4 fingers up, thumb in)': ('curled', 1.0, 1.0, 1.0, 1.0),
    'D (Index up, others O)': ('curled', 1.0, 0.0, 0.0, 0.0),
    'I (Pinky up)': ('across', 0.0, 0.0, 0.0, 1.0),
    'L (Index & thumb out)': ('out', 1.0, 0.0, 0.0, 0.0),
    'V (Index & mid up, spread)': ('across', 1.0, 1.0, 0.0, 0.0),
    'W (Index, mid, ring up)': ('across', 1.0, 1.0, 1.0, 0.0),
    'Y (Thumb & pinky out)': ('out', 0.0, 0.0, 0.0, 1.0),
}

for name, params in asl_tests.items():
    # reconstruct unnormalized 21 landmarks
    # from create_asl_landmarks
    lm_vec = create_asl_landmarks(*params) # 63
    lm = lm_vec.reshape(21, 3)
    geom = analyze_hand_geometry(lm)
    ext_pattern = f"T:{int(geom['thumb_ext'])} I:{int(geom['index_ext'])} M:{int(geom['mid_ext'])} R:{int(geom['ring_ext'])} P:{int(geom['pinky_ext'])}"
    print(f"{name:<28} -> {ext_pattern} | d_idx_mid={geom['d_index_mid']:.2f}")
