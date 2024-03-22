from math import pi
import numpy as np
import cv2
import math
from src.config import params 

# intializing config parametrs
param = params()
pad_params = param['pad_parameters']


""" Utility Functions """

def point_to_line_distance(point: tuple, slope: float, target: tuple):
    """Finds the distance between point and a line

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

def substitute(pt1, slope, target) -> bool:
    """Helps in finding whether the pt1 is same side of target and line

    Args:
        pt1 (list or tuple): point where the line passes through
        slope (float): slope of the line wich is passing through pt1
        target (list or tuple): target point

    Returns:
        bool : return whether the substitute is positive or not
    """
    x1, y1 = pt1
    x, y = target
    
    return True if y - y1 - slope*x + slope*x1 > 0 else False


# Convert the point to Q4 quadrant to make it equalize to co-ordination system
def point_Q4(point):
    """Convert the points to Q4 quadrant which helps for line equations

    Args:
        point (tuple): point (x,y)

    Returns:
        tuple: point convert to the Q4
    """
    # print(point)
    return [point[0], -1*point[1]] if point[1] > 0 else [point[0], point[1]]


def get_corner_indices(canny_img: np.array) -> list:
    """Gets the corner point of the tilted object

    Args:
        canny_img (np.array): image array

    Returns:
        list: gives the corner indices of the tilted object
    """
    non_zero_canny = np.nonzero(canny_img)
    corner_indices = [[non_zero_canny[1][0], non_zero_canny[0][0]], [non_zero_canny[1][-1], non_zero_canny[0][-1]]]
    
    non_zero_canny = np.nonzero(canny_img.T)
    corner_indices += [[non_zero_canny[0][0], non_zero_canny[1][0]], [non_zero_canny[0][-1], non_zero_canny[1][-1]]]
    
    return corner_indices


def draw_points(img: np.array, indices: list) -> np.array:
    """Plot the points to the image

    Args:
        img (np.array): image array
        indices (list): points to be plot in the image

    Returns:
        np.array: _description_
    """
    radius, color, thickness = 5, (255,255,255), -1
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


def draw_boundary(img: np.array, lines: list)->np.array:
    """Draw boundaries for forming quadileteral with given vertices

    Args:
        img (np.array): image array
        lines (list): list of lines wrt points

    Returns:
        np.array: Image with drawn lines
    """
    for i in range(len(lines)):
        j = i + 1
        if j == len(lines):
            j = 0
        cv2.line(img, point_Q4(lines[i]), point_Q4(lines[j]), (255,255,255), 2)
        
    return img
        

def save_img(img: np.array, img_path: str):
    """Saving the image to the path

    Args:
        img (np.array): image to be saved
        img_path (str): Path
    """
    cv2. imwrite(img_path, img)
    
    
def make_line(two_coordinates_indices: list) -> list:
    """Make lines with given two points

    Args:
        two_coordinates (list): list of two point coordinates for forming lines

    Returns:
        list: return a line with point and slope
    """
    line_lst = []
    # print(two_coordinates_indices)
    for points in two_coordinates_indices:
        (x1, y1), (x2, y2) = point_Q4(points[0]), point_Q4(points[1])
        slope = (y2 - y1) / (x2 - x1)
        line_lst.append([[x1,y1],slope])
        
    return line_lst


def pad_image(img, 
              top= pad_params['top'], 
              bottom= pad_params['bottom'], 
              left= pad_params['left'], 
              right= pad_params['right'], 
              border_color= pad_params['border_color']) -> np.array:
    """Padding the image

    Returns:
        np.array: returns a padded image 
    """
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_color)


def point_to_foot(line: list, target: list) -> tuple:
    """Foot of the perpendicular

    Args:
        line (list): Line containing information of slope and a point
        target (list): target point

    Returns:
        tuple: foot of the perpendicular
    """
    x0, y0 = target
    (x1, y1), slope = line
    
    alpha = -1 * (slope*x0 - y0 + y1 - slope*x1) / (slope**2 + 1)
    
    return (alpha*slope + x0, y0 - alpha)
    
def cyclic_ordering_indices(indices: list):
    """The indices we have are disorder to make them vertices of quadrilateral a cyclic order is formed

    Args:
        indices (list): corner indices of tilted object

    Returns:
        list: ordered list of corners
    """
    A, C, B, D = indices
    return [A, B, C, D]

def solutio_of_2_lines(line: list):
    pass


def distance_btw_pts(pt1, pt2):
    pt1, pt2 = point_Q4(pt1), point_Q4(pt2)
    return np.linalg.norm(np.array(pt1) - np.array(pt2))


def equalize_indices_of_padded_to_org_img(indices: list):
    for idx in indices:
        idx = [idx[0] - pad_params['top'], idx[1] - pad_params['left']]
        
    return idx
