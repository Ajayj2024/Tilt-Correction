from math import pi
import numpy as np
import cv2

""" Utility Functions """

def point_to_line_distance(point: tuple, slope: float, target: tuple):
    """Finds the distance between poin and a line

    Args:
        point (tuple): reference point to make a line (Q4)
        target (tuple): target point to calculate distance (Q4)
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


# Convert the point to Q4 quadrant to make it equalize to co-ordination system
def point_Q4(point: tuple):
    """Convert the point to Q4 quadrant used for the line equations

    Args:
        point (tuple): point (x,y)

    Returns:
        tuple: point convert to the Q4
    """
    return (point[0], -1*point[1])