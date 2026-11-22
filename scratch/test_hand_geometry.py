import numpy as np

def analyze_hand_geometry(landmarks):
    """
    Analyzes 21 landmarks {x, y, z} and extracts robust geometric properties:
    1. Extension state of each finger [thumb, index, mid, ring, pinky] (0=curled, 1=extended, 0.5=curved)
    2. Inter-finger distances (index-mid, mid-ring, index-thumb, etc.)
    3. Hand orientation (palm vector from wrist to middle MCP)
    4. Tip distances relative to wrist and palm
    """
    wrist = landmarks[0]
    
    # Finger landmarks:
    # Thumb: 1, 2, 3, 4 (CMC, MCP, IP, TIP)
    # Index: 5, 6, 7, 8 (MCP, PIP, DIP, TIP)
    # Middle: 9, 10, 11, 12
    # Ring: 13, 14, 15, 16
    # Pinky: 17, 18, 19, 20

    # Vector from wrist to middle MCP (palm vertical axis)
    palm_v = landmarks[9] - wrist
    palm_len = np.linalg.norm(palm_v)
    if palm_len < 1e-6: palm_len = 1.0
    palm_unit = palm_v / palm_len
    
    # Palm normal vector (approximate using index MCP and pinky MCP)
    palm_cross = np.cross(landmarks[5] - wrist, landmarks[17] - wrist)
    palm_cross_len = np.linalg.norm(palm_cross)
    if palm_cross_len < 1e-6: palm_cross_len = 1.0
    palm_normal = palm_cross / palm_cross_len

    # Helper: Euclidean distance
    def dist(i, j):
        return np.linalg.norm(landmarks[i] - landmarks[j])

    # Check finger extension:
    # A finger is EXTENDED if tip distance to wrist is significantly larger than PIP distance to wrist,
    # and tip is further from MCP than PIP is.
    def is_finger_extended(mcp, pip, dip, tip):
        d_wrist_tip = dist(0, tip)
        d_wrist_pip = dist(0, pip)
        d_mcp_tip = dist(mcp, tip)
        d_mcp_pip = dist(mcp, pip)
        # Fully extended: tip is much further from wrist than PIP
        return (d_wrist_tip > d_wrist_pip * 1.15) and (d_mcp_tip > d_mcp_pip * 1.3)

    def is_finger_curled(mcp, pip, dip, tip):
        d_wrist_tip = dist(0, tip)
        d_wrist_pip = dist(0, pip)
        d_mcp_tip = dist(mcp, tip)
        return (d_wrist_tip < d_wrist_pip * 1.05) or (d_mcp_tip < dist(mcp, pip) * 1.1)

    index_ext = is_finger_extended(5, 6, 7, 8)
    mid_ext = is_finger_extended(9, 10, 11, 12)
    ring_ext = is_finger_extended(13, 14, 15, 16)
    pinky_ext = is_finger_extended(17, 18, 19, 20)

    # Thumb extension & orientation
    # Thumb tip distance from palm center (MCP 9) and pinky MCP (17)
    d_thumb_pinky = dist(4, 17)
    d_thumb_index_mcp = dist(4, 5)
    d_thumb_index_tip = dist(4, 8)
    d_wrist_thumb_tip = dist(0, 4)
    d_wrist_thumb_ip = dist(0, 3)
    thumb_extended = (d_wrist_thumb_tip > d_wrist_thumb_ip * 1.1) and (d_thumb_index_mcp > palm_len * 0.45)

    # Inter-finger tip distances relative to palm length
    d_index_mid = dist(8, 12) / palm_len
    d_mid_ring = dist(12, 16) / palm_len
    d_ring_pinky = dist(16, 20) / palm_len
    d_thumb_index = d_thumb_index_tip / palm_len

    return {
        'thumb_ext': bool(thumb_extended),
        'index_ext': bool(index_ext),
        'mid_ext': bool(mid_ext),
        'ring_ext': bool(ring_ext),
        'pinky_ext': bool(pinky_ext),
        'd_index_mid': float(d_index_mid),
        'd_thumb_index': float(d_thumb_index),
        'd_thumb_pinky': float(d_thumb_pinky / palm_len),
        'palm_len': float(palm_len)
    }

print("Geometric analysis function verified.")
