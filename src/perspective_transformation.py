import numpy as np
import cv2

class PerspectiveTransformation:
    def __init__(self, img_path) -> None:
        self.img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
 