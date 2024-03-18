from math import pi
import numpy as np
import cv2

""" Utility Functions """

def point_to_line_distance(point: tuple, target: tuple, slope: float):
    """Finds the distance between poin and a line

    Args:
        point (tuple): reference point to make a line
        target (tuple): target point to calculate distance
        slope (float): slope of line

    Returns:
        np.array : returns the distance
    """
    denominator = 1+pow(slope,2)
    denominator = np.sqrt([denominator])
    x, y = target
    x1, y1 = point
    # if slope != float('inf'):
    return np.absolute([y - slope*x - y1 + slope*x1]) / denominator if slope != float('inf') else  np.abs([x-x1])



def load_image(img_path, shape=None):
    img = cv2.imread(img_path)
    if shape is not None:
        img = cv2.resize(img, shape)
    
    return img

def save_image(img_path, img):
    cv2.imwrite(img_path, img)

def get_rad(theta, phi, gamma):
    return (deg_to_rad(theta),
            deg_to_rad(phi),
            deg_to_rad(gamma))

def get_deg(rtheta, rphi, rgamma):
    return (rad_to_deg(rtheta),
            rad_to_deg(rphi),
            rad_to_deg(rgamma))

def deg_to_rad(deg):
    return deg * pi / 180.0

def rad_to_deg(rad):
    return rad * 180.0 / pi