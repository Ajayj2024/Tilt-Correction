import cv2 as cv
import numpy as np
from config import params

class TiltCorrection:
    def __init__(self, tilted_obj) -> None:
        self.tilted_obj = tilted_obj
        
        # convert the object image to gray scale
        self.tilted_gray_obj = cv.cvtColor(self.tilted_obj, cv.COLOR_BGR2GRAY)
        self.params = params()
        
    @property
    def hough_lines(self) -> np.array:
        
        # Edge detection
        edge_detection = cv.Canny(self.tilted_gray_obj, 
                                  self.params['canny_parameters']['low_threshold'], 
                                  self.params['canny_parameters']['high_thrseshold'], 
                                  None if self.params['canny_parameters']['L2gradient'] == 'None' else self.params['canny_parameters']['L2gradient'],
                                  self.params['canny_parameters']['aperture_size']
                                  )
        
        # Finding the hough lines
        hough_lines = cv.HoughLinesP(edge_detection, self.params['hough_line_parameters']['rho'], 
                                     eval(self.params['hough_line_parameters']['theta']), 
                                     self.params['hough_line_parameters']['threshold'], 
                                     minLineLength=self.params['hough_line_parameters']['min_line_length'], 
                                     maxLineGap=self.params['hough_line_parameters']['max_line_gap']
                                     )
        
        return hough_lines
    
    def creating_boundaries(self, hough_lines):
        pass
        

