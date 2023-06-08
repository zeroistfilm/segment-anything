import cv2
import numpy as np

def extract_food_bins(fn):
    image = cv2.imread(fn)
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # lower_yellow = np.array([15, 100, 100])
    # upper_yellow = np.array([34, 255, 255])
    lower_yellow = np.array([15, 70, 70])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    return mask_yellow