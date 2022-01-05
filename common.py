import cv2
import numpy as np
from numpy.matrixlib import mat
import scipy.ndimage

def read_img(path):
    """Read image."""
    image = cv2.imread(path, cv2.COLOR_BGR2RGB)
    return image

def save_img(img, path):
    """Save image."""
    cv2.imwrite(path, img)

def match_filter(matches, kpts):
    out = []
    for match in matches:
        out.append(kpts[match.trainIdx])
    return tuple(out)

def gaussian_filter(image, sigma):
    
    H, W = image.shape
    # Ensure that the kernel size isn't too big and is odd
    kernel_size = int(2 * np.ceil(2 * sigma) + 1)
    kernel_size = min(kernel_size, min(H, W) // 2)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
        
    kernel_gaussian = np.array(
        [[1/(2*np.pi*sigma**2)*np.exp(-(x**2+y**2)/(2*sigma**2)) for x in range(-kernel_size//2, kernel_size//2+1)] for y in range(-kernel_size//2, kernel_size//2+1)])
    output = scipy.ndimage.convolve(image, kernel_gaussian, mode='reflect')
    return output

def xy_filter(kpts1, kpts2, matches):
    x_left = 600
    x_right = 800
    y_low = 0
    y_high = 1000
    out = []
    for match in matches:
        if x_left < kpts1[match.queryIdx].pt[0] < x_right and y_low < kpts1[match.queryIdx].pt[1] < y_high:
            out.append(match)
    return out


def rectangular_mask(img, x_left=460, x_right=840, y_low=100, y_high=1000):
    out = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if not (x_left < j < x_right and y_high < i < y_low):
                out[i, j] = 0
    return out

def match_keypoints(base_desc, img2):
    img2_filtered = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2_filtered, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(base_desc, descriptors_2)

    matches = sorted(matches, key = lambda x:x.distance)
    keypoints = match_filter(matches, keypoints_2)
    # out = cv2.drawMatches(img1, keypoints_1, img2,
    #                       keypoints_2, matches, img2, flags=2)
    return keypoints
