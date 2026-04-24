import numpy as np

def extract_landmarks(results):
    """
    Slices MediaPipe Holistic results into 126 features (42 points * 3 coords).
    """
    # 21 points * 3 = 63 features per hand
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    
    return np.concatenate([lh, rh])

def normalize_landmarks(data):
    """
    Applies Wrist-Centering: P_relative = P_absolute - P_wrist.
    Expects shape (30, 126).
    """
    processed = data.copy()
    for f in range(len(processed)):
        # Left Hand relative to LH Wrist (indices 0,1,2)
        processed[f, 0:63] -= np.tile(processed[f, 0:3], 21)
        # Right Hand relative to RH Wrist (indices 63,64,65)
        processed[f, 63:126] -= np.tile(processed[f, 63:66], 21)
    return processed