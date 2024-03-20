from math import pi
import numpy as np
import cv2
import math

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
    # print(slope)
    denominator = 1+ math.pow(slope,2)
    
    denominator = np.sqrt([denominator])
    x, y = target
    x1, y1 = point
    # if slope != float('inf'):
    return np.absolute([y - slope*x - y1 + slope*x1]) / denominator if slope != float('inf') else  np.abs([x-x1])

def substitute(pt1, slope, target):
    x1, y1 = pt1
    x, y = target
    return y - y1 - slope*x + slope*x1


# Convert the point to Q4 quadrant to make it equalize to co-ordination system
def point_Q4(point: tuple):
    """Convert the point to Q4 quadrant used for the line equations

    Args:
        point (tuple): point (x,y)

    Returns:
        tuple: point convert to the Q4
    """
    return (point[0], -1*point[1])


def get_corner_indices(canny_img: np.array):
    
    non_zero_canny = np.nonzero(canny_img)
    corner_indices = [[non_zero_canny[1][0], non_zero_canny[0][0]], [non_zero_canny[1][-1], non_zero_canny[0][-1]]]
    
    non_zero_canny = np.nonzero(canny_img.T)
    corner_indices += [[non_zero_canny[0][0], non_zero_canny[1][0]], [non_zero_canny[0][-1], non_zero_canny[1][-1]]]
    
    return corner_indices


def draw_points(img: np.array, indices: list):
    
    radius, color, thickness = 15, (255,255,255), -1
    text = 'Point'  # Annotation text
    font = cv2.FONT_HERSHEY_SIMPLEX  # Font
    font_scale = 1.5  # Font scale
    text_color = (255, 255, 255)  # White color
    text_thickness = 1  # Text thickness

    for val in indices:
        print(val)
        cv2.circle(img,(int(val[0]),int(val[1])),radius,color,thickness)
        cv2.putText(img, str(val), (int(val[0]) + 10, int(val[1]) - 10), font, font_scale, text_color, text_thickness)
    
    return img   

def draw_line(img: np.array, lines):
    for indices in lines:
        cv2.line(img, indices[0], indices[1], (255,255,255), 2)
    return img
        
def save_img(img: np.array, img_path: str):
    cv2. imwrite(img_path, img)
    
    
def make_line(two_coordinates: list):
    line_lst = []
    for points in two_coordinates:
        (x1, y1), (x2, y2) = point_Q4(points[0]), point_Q4(points[1])
        slope = (y2 - y1) / (x2 - x1)
        line_lst.append([(x1,y1),slope])
        
    return line_lst

def pad_image(img, top=600, bottom= 600, left= 600, right= 600, border_color= [0,0,0]):
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_color)

def point_to_foot(line: list, target: list):
    x0, y0 = target
    (x1, y1), slope = line
    
    alpha = -1 * (slope*x0 - y0 + y1 - slope*x1) / (slope**2 + 1)
    
    return (alpha*slope + x0, y0 - alpha)
    
def solutio_of_2_lines(line: list):
    pass

def cyclic_ordering_indices(indices: list):
    A, C, B, D = indices
    return [A, B, C, D]