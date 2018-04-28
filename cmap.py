
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def get_cmap():
    colors = [(0, 0, 255),
              (51, 0, 255),
              (76, 0, 255),
              (102, 0, 255),
              (127, 0, 255),
              (153, 0, 255),
              (179, 0, 255),
              (204, 0, 255),
              (229, 0, 255),
              (255, 0, 255),
              (255, 0, 0),
              (255, 51, 0),
              (255, 76, 0),
              (255, 102, 0),
              (255, 127, 0),
              (255, 153, 0),
              (255, 179, 0),
              (255, 204, 0),
              (255, 229, 0),
              (255, 255, 0),
              (255, 255, 51),
              (255, 255, 102),
              (255, 255, 153),
              (255, 255, 204),
              (255, 255, 255)]
    colors = np.array(colors, dtype=np.float64)/255.
    cmap = LinearSegmentedColormap.from_list('jet', colors, N=2048)
    return cmap
