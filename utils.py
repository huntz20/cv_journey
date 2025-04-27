import numpy as np
import cv2

def get_limits(color):
    c = np.uint8([[color]])
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    lower_limit  = np.array([hsv[0][0][0] - 10, 100, 100])
    upper_limit = np.array([hsv[0][0][0] + 10, 255, 255])
    return lower_limit, upper_limit