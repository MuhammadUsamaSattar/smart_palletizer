import math

import numpy as np


def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


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

    return img.astype(dtype)


def subsample(img, factor=2, method="auto"):
    """
    Sub-sample an image (depth map or RGB) intelligently using non-zero median or mean.

    Parameters:
        img    : np.ndarray, 2D (depth) or 3D (H,W,C) image
                 - np.uint16 for depth
                 - np.uint8 for RGB
        factor : int, sub-sampling factor (2, 3, 4, ...)
        method : 'median', 'mean', or 'auto'
                 'auto' chooses median for small factor (<4) and mean for larger factors

    Returns:
        subsampled_img : sub-sampled image of the same dtype as input
    """
    assert img.ndim in (2, 3), "Input must be 2D (depth) or 3D (RGB)"
    dtype = img.dtype
    if method == "auto":
        method = "median" if factor < 4 else "mean"

    if img.ndim == 2:
        H, W = img.shape
        C = 1
        img_reshaped = img[:, :, np.newaxis]
    else:
        H, W, C = img.shape
        img_reshaped = img

    new_H = (H + factor - 1) // factor
    new_W = (W + factor - 1) // factor

    # Pad to make divisible by factor
    pad_H = new_H * factor - H
    pad_W = new_W * factor - W
    padded = np.pad(
        img_reshaped,
        ((0, pad_H), (0, pad_W), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    # Reshape into blocks
    blocks = padded.reshape(new_H, factor, new_W, factor, C)
    blocks = blocks.transpose(0, 2, 1, 3, 4)
    blocks_flat = blocks.reshape(new_H, new_W, factor * factor, C)

    valid = blocks_flat > 0

    if method == "median":
        # Convert to float32 for median computation
        blocks_float = blocks_flat.astype(np.float32)
        blocks_masked = np.where(valid, blocks_float, np.nan)
        subsampled = np.nanmedian(blocks_masked, axis=2)
        subsampled = np.nan_to_num(subsampled, nan=0).astype(
            dtype
        )  # convert back to original dtype
    elif method == "mean":
        sums = np.sum(blocks_flat * valid, axis=2, dtype=np.float32)
        counts = np.sum(valid, axis=2, dtype=np.float32)
        subsampled = np.divide(
            sums, counts, out=np.zeros_like(sums, dtype=np.float32), where=counts > 0
        )
        subsampled = np.nan_to_num(subsampled, nan=0).astype(dtype)
    else:
        raise ValueError("Method must be 'median', 'mean', or 'auto'")

    if img.ndim == 2:
        subsampled = subsampled[:, :, 0]

    return subsampled


def get_mask(type, dy, dx):
    """Return boolean mask of selected neighborhood offsets for the given type."""
    if type == "U-1D":
        return dy < 0
    elif type == "D-1D":
        return dy > 0
    elif type == "L-1D":
        return dx < 0
    elif type == "R-1D":
        return dx > 0
    elif type == "U-2D-Excl":
        return (dy < 0) & (np.abs(dx) <= np.abs(dy))
    elif type == "D-2D-Excl":
        return (dy > 0) & (np.abs(dx) <= np.abs(dy))
    elif type == "L-2D-Excl":
        return (dx < 0) & (np.abs(dy) <= np.abs(dx))
    elif type == "R-2D-Excl":
        return (dx > 0) & (np.abs(dy) <= np.abs(dx))
    elif type == "U-2D-Incl":
        return dy < 0
    elif type == "D-2D-Incl":
        return dy > 0
    elif type == "L-2D-Incl":
        return dx < 0
    elif type == "R-2D-Incl":
        return dx > 0
    else:
        raise ValueError(f"Unknown type: {type}")


def holePatching(img, size, mode, type, max_iter=10):
    """
    Efficient iterative hole-filling for uint16 depth images using directional masks.
    Median, min, or max are applied only to non-zero neighbors.

    Parameters:
        img      : np.uint16 depth image
        size     : odd integer window size (e.g., 3, 5, 7, ...)
        mode     : 'median', 'min', 'max'
        type     : directional selection (e.g., 'U-1D', 'U-2D-Incl', 'R-2D-Excl', ...)
        max_iter : maximum number of iterations (default 10)

    Returns:
        patched_img : np.uint16 image
    """
    assert img.dtype == np.uint16, "Input must be np.uint16"
    assert size % 2 == 1, "Size must be odd"

    h, w = img.shape
    pad = size // 2
    patched = img.copy()

    # Prepare offset grid
    idx = np.arange(-pad, pad + 1)
    dy, dx = np.meshgrid(idx, idx, indexing="ij")

    # Compute only the needed mask
    mask = get_mask(type, dy, dx)
    dy_offsets = dy[mask]
    dx_offsets = dx[mask]

    for _ in range(max_iter):
        zeros = patched == 0
        if not np.any(zeros):
            break

        # Pad the current image
        padded = np.pad(patched, pad_width=pad, mode="edge")

        # Vectorized neighbor extraction
        neighbors = np.array(
            [
                padded[pad + dy_i : h + pad + dy_i, pad + dx_i : w + pad + dx_i]
                for dy_i, dx_i in zip(dy_offsets, dx_offsets)
            ],
            dtype=np.uint16,
        )

        valid = neighbors > 0  # Only consider non-zero neighbors

        if mode == "median":
            neighbors_masked = np.where(valid, neighbors, np.nan)
            newvals = np.nanmedian(neighbors_masked, axis=0)
            newvals = np.nan_to_num(newvals, nan=0).astype(np.uint16)
        elif mode == "min":
            neighbors_masked = np.where(valid, neighbors, np.iinfo(np.uint16).max)
            newvals = np.min(neighbors_masked, axis=0)
            newvals[np.all(~valid, axis=0)] = 0
        elif mode == "max":
            neighbors_masked = np.where(valid, neighbors, 0)
            newvals = np.max(neighbors_masked, axis=0)
        else:
            raise ValueError("Invalid mode. Use 'median', 'min', or 'max'.")

        # Fill only zero pixels with at least one valid neighbor
        fill_mask = zeros & (np.any(valid, axis=0))
        if not np.any(fill_mask):
            break

        patched[fill_mask] = newvals[fill_mask]

    return patched


def spatial_edge_preserving_filter(img, alpha=0.6, delta=8):
    """
    Edge-preserving spatial filter using recursive EMA with delta-based edge detection.
    Applies 2 passes in X and Y directions (forward and backward).

    Parameters:
        img   : np.ndarray, HxW (uint16) or HxWxC (uint8) image
        alpha : float, smoothing factor for EMA (0..1). 1=no smoothing, 0=infinite smoothing
        delta : int, threshold to temporarily disable smoothing (same units as img)

    Returns:
        filtered_img : np.ndarray, same shape and dtype as input
    """
    if img.ndim == 2:
        img_proc = img[:, :, np.newaxis]  # Convert to HxWx1 for uniform processing
    else:
        img_proc = img.copy()

    H, W, C = img_proc.shape
    filtered = img_proc.astype(np.float32)

    # Bi-directional passes
    for axis in [1, 0]:  # 1=X-axis (rows), 0=Y-axis (columns)
        for direction in [1, -1]:  # forward and backward
            if axis == 1:
                # Iterate over rows
                for y in range(H):
                    row = filtered[y, :, :]
                    if direction == -1:
                        row = row[::-1, :]
                    # Recursive EMA along row
                    for x in range(1, W):
                        diff = np.abs(row[x, :] - row[x - 1, :])
                        a = np.where(diff > delta, 1.0, alpha)
                        row[x, :] = a * row[x, :] + (1 - a) * row[x - 1, :]
                    if direction == -1:
                        row = row[::-1, :]
                    filtered[y, :, :] = row
            else:
                # Iterate over columns
                for x in range(W):
                    col = filtered[:, x, :]
                    if direction == -1:
                        col = col[::-1, :]
                    for y in range(1, H):
                        diff = np.abs(col[y, :] - col[y - 1, :])
                        a = np.where(diff > delta, 1.0, alpha)
                        col[y, :] = a * col[y, :] + (1 - a) * col[y - 1, :]
                    if direction == -1:
                        col = col[::-1, :]
                    filtered[:, x, :] = col

    # Convert back to original dtype
    filtered = np.clip(filtered, 0, np.iinfo(img.dtype).max).astype(img.dtype)
    if img.ndim == 2:
        filtered = filtered[:, :, 0]

    return filtered


def quaternion_from_euler(ai, aj, ak):
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    q = np.empty((4,))
    q[0] = cj * sc - sj * cs
    q[1] = cj * ss + sj * cc
    q[2] = cj * cs - sj * sc
    q[3] = cj * cc + sj * ss

    return q
