import pandas as pd
import numpy as np

def normalize_landmarks(landmarks):
    df = pd.DataFrame([landmarks], columns=[f"{axis}{i}" for i in range(1, 22) for axis in ['x', 'y', 'z']])
    df.iloc[:, 0:-1:3] = df.iloc[:, 0:-1:3].sub(df['x1'], axis=0)
    df.iloc[:, 1:-1:3] = df.iloc[:, 1:-1:3].sub(df['y1'], axis=0)

    wrist_x = df['x1'].values
    wrist_y = df['y1'].values
    fingertip_x = df['x10'].values
    fingertip_y = df['y10'].values

    scale_factor = np.sqrt((fingertip_x - wrist_x) ** 2 + (fingertip_y - wrist_y) ** 2).reshape(-1, 1)
    df.iloc[:, :-1:3] /= scale_factor
    df.iloc[:, 1::3] /= scale_factor

    return df

gesture_map = {
    0: 'call',
    1: 'dislike',
    2: 'fist',
    3: 'four',
    4: 'like',
    5: 'mute',
    6: 'ok',
    7: 'one',
    8: 'palm',
    9: 'peace',
    10: 'peace_inverted',
    11: 'rock',
    12: 'stop',
    13: 'stop_inverted',
    14: 'three',
    15: 'three2',
    16: 'two_up',
    17: 'two_up_inverted'
}