import math
import numpy as np

def distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def time_filter(img: np.ndarray, prev_filtered_img, alpha=0.5, delta=20):
    if not isinstance(prev_filtered_img, np.ndarray):
        return img
    
    dtype = img.dtype
    img = img.astype(np.float32)
    prev_filtered_img = prev_filtered_img.astype(np.float32)

    diff = np.abs(img - prev_filtered_img)
    mask = diff > delta
    a = np.where(mask, 1.0, alpha)
    img = a * img + (1 - a) * prev_filtered_img

    return img.astype(np.uint16)