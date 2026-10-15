from scipy.ndimage import center_of_mass
import numpy as np
from plot_sequence import load_one_sequence
from normalization import load_normalization_stats

X, y_reflect, y_lightning = load_one_sequence("train", 0)
CHANNEL_MEAN, CHANNEL_STD = load_normalization_stats()
radar_in = X[:, 0] * CHANNEL_STD[0] + CHANNEL_MEAN[0]

for i, frame in enumerate(radar_in):
    cy, cx = center_of_mass(np.clip(frame, 0, None))
    print(f"input  t-{(6 - i) * 10:>3}m  centroid=({cy:.1f}, {cx:.1f})")
for i, frame in enumerate(y_reflect):
    cy, cx = center_of_mass(np.clip(frame, 0, None))
    print(f"fcst   t+{(i + 1) * 10:>3}m  centroid=({cy:.1f}, {cx:.1f})")
