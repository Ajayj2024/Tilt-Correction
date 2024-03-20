import cv2 as cv
import numpy as np
from src.config import params
from src.utils import *
from src.image_transformation import ImageTransformer
import os
from tqdm import tqdm
class TiltCorrection:
    def __init__(self, tilted_obj_path) -> None:
        self.tilted_obj_path = tilted_obj_path
        self.tilted_gray_obj = cv.imread(tilted_obj_path, cv.IMREAD_GRAYSCALE)
        self.params = params()
        
        self.results_path = self.params['result_dir']
        # convert the object image to gray scale
        # self.tilted_gray_obj = cv.cvtColor(self.tilted_obj, cv.COLOR_BGR2GRAY)
        
        # Edge detection
        self.edge_detection = cv.Canny(self.tilted_gray_obj, 
                                  self.params['canny_parameters']['low_threshold'], 
                                  self.params['canny_parameters']['high_threshold'], 
                                  None if self.params['canny_parameters']['L2gradient'] == 'None' else self.params['canny_parameters']['L2gradient'],
                                  self.params['canny_parameters']['aperture_size']
                                  )
        self.padded_edge_detection = pad_image(self.edge_detection)
        
        self.height = self.padded_edge_detection.shape[0]
        self.width = self.padded_edge_detection.shape[1]
        # image corner coordinates
        self.ref_coordinates = {
            'A': (0,0), 
            'B': (self.width, 0), 
            'C': (self.width, -self.height), 
            'D': (0, -self.height)
        }
        
        
    def perspective_transform(self):
        corner_indices = get_corner_indices(self.padded_edge_detection)
        img = draw_points(img= self.padded_edge_detection.copy(), indices= corner_indices)
        save_img(img, 'results/canny_corners/shoe.png')
        corner_indices = cyclic_ordering_indices(corner_indices)
        lines_lst = []
        for i in range(len(corner_indices)):
            pt1 = corner_indices[i]
            j = i+1
            if j == len(corner_indices):
                j = 0
                
            pt2 = corner_indices[j]
            lines_lst.append([pt1, pt2])
        # print(lines_lst)
        # img = draw_line(self.padded_edge_detection.copy(),lines_lst)
        # save_img(img, 'results/canny_corners/shoe.png')
        lines = make_line(lines_lst)
        # print(lines)
        line_map_to_target = {}
        for vertex in self.ref_coordinates.keys():
            print(vertex)
            target = self.ref_coordinates[vertex]
            min_d = float('inf')
            for line in lines:
                # print(line)
                d = point_to_line_distance(line[0],line[1], target)
                foot = point_to_foot(line, target)
                print(d,foot)
                print(self.padded_edge_detection.shape)
                if foot[0] >= 0 and foot[1] <= 0 and d < min_d and foot[0] < self.width and abs(foot[1]) < self.height:
                    line_map_to_target[vertex] = line
                    min_d = d
                    
        print(line_map_to_target)
        
        boundary_lines = {}
        non_zero_canny_padded = np.nonzero(self.padded_edge_detection)
        t = 0
        for vertex in tqdm(line_map_to_target.keys()):
            pt = line_map_to_target[vertex][0]
            slope = line_map_to_target[vertex][1]
            l1 = substitute(pt, slope, self.ref_coordinates[vertex])
            l1_sign = l1 > 0
            print(l1_sign)
            
            indices = []
            
            for j in range(len(non_zero_canny_padded[0])):
                l2 = substitute(pt, slope,[non_zero_canny_padded[1][j],-non_zero_canny_padded[0][j]])
                l2_sign = l2 > 0
                if l2_sign == l1_sign:
                    indices.append([non_zero_canny_padded[1][j],-non_zero_canny_padded[0][j]])
            # print(indices)
            boundary_lines[t] = [pt, slope]   
            
            for p in indices:
                l2 = substitute(boundary_lines[t][0], boundary_lines[t][1],p)
                l2_sign = l2 > 0
                if l1_sign == l2_sign:
                    boundary_lines[t] = [p,boundary_lines[t][1]]
                    temp = True
                        
                
            t += 1
            
        print(boundary_lines)
            
        sol_points = []
        for i in boundary_lines.keys():
            line1 = boundary_lines[i]
            # print(line1)
            a1, b1, c1 = -1*line1[1], 1, -1*line1[1]*line1[0][0] + (line1[0][1])
            j = i + 1
            if i == 3:
                j = 0

            line2 = boundary_lines[j]
            a2, b2, c2 = -1*line2[1], 1, -1*line2[1]*line2[0][0] + (line2[0][1])

            A = np.array([[a1, b1], [a2, b2]])

            # Create constant matrix
            B = np.array([c1, c2])

            solution = np.linalg.solve(A, B)
            sol_points.append(solution)
            
        for i in range(len(sol_points)):
            sol_points[i][0]= int(sol_points[i][0])
            sol_points[i][1]= int(-1*sol_points[i][1])
        print(sol_points)
        
        img = draw_points(self.padded_edge_detection.copy(), sol_points)
        save_img(img,'results/canny_corners/shoe1.png') 
        
        pts1 = np.float32(sol_points)
        # Size of the Transformed Image
        pts2 = np.float32([[0,0],[600,0],[600,300],[0,300]])
        # for val in pt1:
        #     cv2.circle(padded_image,(val[0],val[1]),5,(0,255,0),-1)
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(self.padded_edge_detection,M,(600,300))
        cv2.imwrite('results/canny_corners/shoe2.png',dst)
        
        
    # @property
    # def hough_lines(self) -> np.array:
    #     ''' Return hough lines'''
        
        
    #     try:
    #         # Finding the hough lines
    #         hough_lines = cv.HoughLinesP(self.edge_detection, self.params['hough_line_parameters']['rho'], 
    #                                     eval(self.params['hough_line_parameters']['theta']), 
    #                                     self.params['hough_line_parameters']['threshold'], 
    #                                     minLineLength=self.params['hough_line_parameters']['min_line_length'], 
    #                                     maxLineGap=self.params['hough_line_parameters']['max_line_gap']
    #                                     )
            
    #         return hough_lines
            
    #     except Exception as e:
    #         raise e
        
       
    
    # def boundary_lines(self):
    #     # Hough lines
    #     hough_lines = self.hough_lines
    #     # print(hough_lines)
    #     boundary_lines = []
        
    #     # finding the corners of object
    #     non_zero_canny = np.nonzero(self.edge_detection)
    #     corner_indices = [[non_zero_canny[0][0], non_zero_canny[1][0]], [non_zero_canny[0][-1], non_zero_canny[1][-1]]]
    #     non_zero_canny = np.nonzero(self.edge_detection.T)
    #     corner_indices += [[non_zero_canny[1][0], non_zero_canny[0][0]], [non_zero_canny[1][-1], non_zero_canny[0][-1]]]
    #     print(corner_indices)
    #     # Finding the hough line boundaries of an object by calculating the distances of corner point to the houghlines
    #     for indices in corner_indices:
    #         min_distance = float('inf')
    #         nearest_line = None
    #         for line in hough_lines:
    #             x1, y1, x2, y2 = line[0]
    #             slope = float('inf')
    #             if x2 != x1:
    #                 slope = (y2 - y1)/(x2 - x1)
    #             d = point_to_line_distance((x1,y1),indices,slope)
    #             if d<min_distance:
    #                 nearest_line = line[0]
    #                 min_distance = d
                    
    #         boundary_lines.append(nearest_line)
            
            
    #     return boundary_lines
    
    
    # def rotate_object_and_save(self):
        
    #     try:
    #         # Acquiring the boundary lines
    #         boundary_lines = self.boundary_lines()
    #         # print(boundary_lines)
    #         height, width = self.height, self.width
    #         # resize = False
    #         # if self.height < self.tilted_gray_obj_diagonal:
    #         #     resize = True
    #         #     self.height = self.tilted_gray_obj_diagonal
                
    #         # if self.width < self.tilted_gray_obj_diagonal:
    #         #     resize = True
    #         #     self.width = self.tilted_gray_obj_diagonal
    #         angles = self.angles(boundary_lines= boundary_lines)
    #         # print(angles)
    #         # center = (height//2, width//2)
    #         center = [899, 938]
    #         i = 1
    #         for angle in angles:
    #             if angle>0:
    #                 rotated_angle = angle if angle < 45 else angle - 90
    #             elif angle<0:
    #                 rotated_angle = angle if abs(angle) < 45 else  90 - abs(angle)
                
    #             # Padding the image
    #             top = 100
    #             bottom = 100
    #             left = 100
    #             right = 100
    #             border_color = [0, 0, 0]  # Border color in BGR format (black in this case)

    #             # # Add border to the image
    #             self.padded_tilted_obj = cv.copyMakeBorder(self.edge_detection, top, bottom, left, right, cv.BORDER_CONSTANT, value=border_color)
    #             # Perform the rotation
    #             M = cv.getRotationMatrix2D(center, angle, 1.0)
    #             rotated_image = cv.warpAffine(self.padded_tilted_obj, M, (width+200, height+200), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
                
    #             res_img_dir = (self.tilted_obj_path.split('/')[-1]).split('.')[0] 
    #             os.makedirs(self.results_path+res_img_dir, exist_ok=True)
                
    #             res_img_path = os.path.join(self.results_path,res_img_dir) + '/' + res_img_dir + f'_{i}' + '.jpg'
    #             i+=1
    #             # print(res_img_path)
    #             # saving image
    #             cv.imwrite(res_img_path,rotated_image)
            
    #         # Store original image
    #         cv.imwrite(os.path.join(self.results_path,res_img_dir) + '/' + 'original_image.jpg', self.tilted_gray_obj)  
    #     except Exception as e:
    #         raise e
            
    
        
    # def angles(self, boundary_lines):
    #     angles = []
    #     for line in boundary_lines:
    #         x1, y1, x2, y2 = line
    #         angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            
    #         if angle not in angles:
    #             angles.append(angle)  
                
    #     return angles  
             
        
        

