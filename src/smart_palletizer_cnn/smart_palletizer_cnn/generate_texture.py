import cv2
import numpy as np

def get_masked_rgb_img(rgb_img: np.ndarray, depth_img: np.ndarray) -> np.ndarray:
    """Get a masked RGB image using depth image.

    Args:
        rgb_img (np.ndarray): RGB image to mask.
        depth_img (np.ndarray): Depth image to use as mask.

    Returns:
        np.ndarray: Masked RGB image.
    """    
    rgb_img = cv2.bitwise_and(rgb_img, rgb_img, mask=cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY))
    cv2.imshow("win", rgb_img)
    cv2.waitKey(0)
    cv2.rot

def rotate_img():
    pass

if __name__ == "__main__":
    rgb_img = cv2.imread("data/medium_box/color_image.png")
    depth_img = cv2.imread("data/medium_box/medium_box_mask_0.png")

    get_masked_rgb_img(rgb_img, depth_img)
    
