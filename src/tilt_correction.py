import cv2 as cv
import numpy as np
from src.config import params
from src.utils import point_to_line_distance
from src.image_transformation import ImageTransformer
import os
class TiltCorrection:
    def __init__(self, tilted_obj_path) -> None:
        self.tilted_obj_path = tilted_obj_path
        self.tilted_gray_obj = cv.imread(tilted_obj_path, cv.IMREAD_GRAYSCALE)
        self.params = params()
        
        self.results_path = self.params['result_dir']
        # convert the object image to gray scale
        # self.tilted_gray_obj = cv.cvtColor(self.tilted_obj, cv.COLOR_BGR2GRAY)
        self.height = self.tilted_gray_obj.shape[0]
        self.width = self.tilted_gray_obj.shape[1]
        
        # Edge detection
        self.edge_detection = cv.Canny(self.tilted_gray_obj, 
                                  self.params['canny_parameters']['low_threshold'], 
                                  self.params['canny_parameters']['high_threshold'], 
                                  None if self.params['canny_parameters']['L2gradient'] == 'None' else self.params['canny_parameters']['L2gradient'],
                                  self.params['canny_parameters']['aperture_size']
                                  )
        
        self.tilted_gray_obj_diagonal = np.sqrt([self.height**2 + self.width**2])
        
    @property
    def hough_lines(self) -> np.array:
        ''' Return hough lines'''
        
        
        try:
            # Finding the hough lines
            hough_lines = cv.HoughLinesP(self.edge_detection, self.params['hough_line_parameters']['rho'], 
                                        eval(self.params['hough_line_parameters']['theta']), 
                                        self.params['hough_line_parameters']['threshold'], 
                                        minLineLength=self.params['hough_line_parameters']['min_line_length'], 
                                        maxLineGap=self.params['hough_line_parameters']['max_line_gap']
                                        )
            
        except Exception as e:
            raise e
        
        return hough_lines
    
    def boundary_lines(self):
        # Hough lines
        hough_lines = self.hough_lines
        # print(hough_lines)
        boundary_lines = []
        
        # finding the corners of object
        non_zero_canny = np.nonzero(self.edge_detection)
        corner_indices = [[non_zero_canny[0][0], non_zero_canny[1][0]], [non_zero_canny[0][-1], non_zero_canny[1][-1]]]
        non_zero_canny = np.nonzero(self.edge_detection.T)
        corner_indices += [[non_zero_canny[1][0], non_zero_canny[0][0]], [non_zero_canny[1][-1], non_zero_canny[0][-1]]]
        # print(corner_indices)
        # Finding the hough line boundaries of an object by calculating the distances of corner point to the houghlines
        for indices in corner_indices:
            min_distance = float('inf')
            nearest_line = None
            for line in hough_lines:
                x1, y1, x2, y2 = line[0]
                slope = float('inf')
                if x2 != x1:
                    slope = (y2 - y1)/(x2 - x1)
                d = point_to_line_distance((x1,y1),indices,slope)
                if d<min_distance:
                    nearest_line = line[0]
                    min_distance = d
                    
            boundary_lines.append(nearest_line)
            
            
        return boundary_lines
    
    
    def rotate_object_and_save(self):
        
        try:
            # Acquiring the boundary lines
            boundary_lines = self.boundary_lines()
            # print(boundary_lines)
            height, width = self.height, self.width
            # resize = False
            # if self.height < self.tilted_gray_obj_diagonal:
            #     resize = True
            #     self.height = self.tilted_gray_obj_diagonal
                
            # if self.width < self.tilted_gray_obj_diagonal:
            #     resize = True
            #     self.width = self.tilted_gray_obj_diagonal
            angles = self.angles(boundary_lines= boundary_lines)
            # print(angles)
            center = (height//2, width//2)
            i = 1
            for angle in angles:
                if angle>0:
                    rotated_angle = angle if angle < 45 else angle - 90
                elif angle<0:
                    rotated_angle = angle if abs(angle) < 45 else  90 - abs(angle)
                
                # Padding the image
                top = 100
                bottom = 100
                left = 100
                right = 100
                border_color = [0, 0, 0]  # Border color in BGR format (black in this case)

                # # Add border to the image
                self.padded_tilted_obj = cv.copyMakeBorder(self.edge_detection, top, bottom, left, right, cv.BORDER_CONSTANT, value=border_color)
                # Perform the rotation
                M = cv.getRotationMatrix2D(center, angle, 1.0)
                rotated_image = cv.warpAffine(self.padded_tilted_obj, M, (width+200, height+200), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
                
                res_img_dir = (self.tilted_obj_path.split('/')[-1]).split('.')[0] 
                os.makedirs(self.results_path+res_img_dir, exist_ok=True)
                
                res_img_path = os.path.join(self.results_path,res_img_dir) + '/' + res_img_dir + f'_{i}' + '.jpg'
                i+=1
                # print(res_img_path)
                # saving image
                cv.imwrite(res_img_path,rotated_image)
            
            # Store original image
            cv.imwrite(os.path.join(self.results_path,res_img_dir) + '/' + 'original_image.jpg', self.tilted_gray_obj)  
        except Exception as e:
            raise e
            
    
        
    def angles(self, boundary_lines):
        angles = []
        for line in boundary_lines:
            x1, y1, x2, y2 = line
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            
            if angle not in angles:
                angles.append(angle)  
                
        return angles  
             
        
        

