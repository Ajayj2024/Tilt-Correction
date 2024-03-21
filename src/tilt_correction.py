import cv2 as cv
import numpy as np
from src.config import params
from src.utils import *
from src.image_transformation import ImageTransformer
import os
from tqdm import tqdm
class TiltCorrection:
    def __init__(self, tilted_obj_path) -> None:
        self.img_name = tilted_obj_path.split('/')[-1]
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
        self.gray_padded_edge_detection = pad_image(self.tilted_gray_obj)
        
        self.height = self.padded_edge_detection.shape[0]
        self.width = self.padded_edge_detection.shape[1]
        # image corner coordinates
        self.ref_coordinates = {
            'A': (0,0), 
            'B': (self.width, 0), 
            'C': (self.width, -self.height), 
            'D': (0, -self.height)
        }
    
    # lines with 2 point coordinates are convert to line having one point and slope
    def make_line_with_point_slope(self):
        
        # Getting extreme corners of edge detection image
        corner_indices = get_corner_indices(self.padded_edge_detection)
        
        # Drawing the point and saving the image
        img = draw_points(img= self.padded_edge_detection.copy(), indices= corner_indices)
        save_img(img, self.params['corner_extreme_img_dir']+self.img_name)
        
        # Arranging the the extreme points in cyclic order like ABCD
        corner_indices = cyclic_ordering_indices(corner_indices)
        
        lines_wrt_points = []
        for i in range(len(corner_indices)):
            pt1 = corner_indices[i]
            j = i+1
            if j == len(corner_indices):
                j = 0
                
            pt2 = corner_indices[j]
            lines_wrt_points.append([pt1, pt2])
        
        lines_wrt_slope = make_line(lines_wrt_points)  
        
        return lines_wrt_slope
    
    # mapping the image corners to the lines which are nearer
    def mapping_lines_to_vertices(self, lines_wrt_point_slope):
        line_map_to_ref_points = {}
        for vertex in self.ref_coordinates.keys():
            target = self.ref_coordinates[vertex]
            min_d = float('inf')
            for line in lines_wrt_point_slope:
                d = point_to_line_distance(line[0],line[1], target)
                foot = point_to_foot(line, target)
                if foot[0] >= 0 and foot[1] <= 0 and d < min_d and foot[0] < self.width and abs(foot[1]) < self.height:
                    line_map_to_ref_points[vertex] = line
                    min_d = d
        return line_map_to_ref_points
    
    # Get extreme boundary lines for the tilted objects
    def get_extreme_boundary_lines(self, line_map_to_vertices):
        boundary_lines = {}
        non_zero_canny_padded = np.nonzero(self.padded_edge_detection)
        t = 0
        for vertex in tqdm(line_map_to_vertices.keys()):
            pt = line_map_to_vertices[vertex][0]
            slope = line_map_to_vertices[vertex][1]
            l1_sign = substitute(pt, slope, self.ref_coordinates[vertex])
            
            
            indices = []
            
            for j in range(len(non_zero_canny_padded[0])):
                l2_sign = substitute(pt, slope,[non_zero_canny_padded[1][j],-non_zero_canny_padded[0][j]])
                
                if l2_sign == l1_sign:
                    indices.append([non_zero_canny_padded[1][j],-non_zero_canny_padded[0][j]])
            boundary_lines[t] = [pt, slope]   
            
            for p in indices:
                l2_sign = substitute(boundary_lines[t][0], boundary_lines[t][1],p)
                if l1_sign == l2_sign:
                    boundary_lines[t] = [p,boundary_lines[t][1]]
                    temp = True
                    
            t += 1
            
            
        return boundary_lines
    
    # Solving for the lines equations
    def solution_of_lines(self,boundary_lines):
        sol_points = []
        for i in range(len(boundary_lines)):
            line1 = boundary_lines[i]
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
            
        return sol_points
    
    # rotating the lines wrt point 
    def line_rotating_wrt_point(self, solution_lines,points_d_map, dist, lines_wrt_slope, u,v):
        
        # consider the maximum opposite side
        a, b = points_d_map[max([dist[u], dist[v]])]
        
        # get the side which should be considered for rotation of opposite sides about fixed poit
        index = lines_wrt_slope.index(make_line([[a,b]])[0])
        if index == -1: 
            print("Index is -1")
            return
        i = 3 if index - 1 == -1 else index - 1
        j = 0 if index + 1 == 4 else index + 1
        
        slope1, slope2 = lines_wrt_slope[i][1], lines_wrt_slope[j][1]
        
        # Just rotate the image upto the inclination equals to average of inclination of opposite sides
        angle = (np.arctan([slope1]) + np.arctan([slope2])) / 2
        res_slope = np.tan(angle).tolist()[0]
        
        # Making the lines parallel with res_slope and passing through a, b respectively
        solution_lines[i] = [a, res_slope]
        solution_lines[j] = [b, res_slope]
    
    def find_rectangle_boundaries(self, solution):
        dist = []
        points_d_map = {}
        lines_wrt_slope = []
        solution_lines = {}
        
        # Finding the length of side 
        for i in range(len(solution)):
            pt1 = solution[i]
            j = i + 1
            if j == len(solution):
                j = 0
                
            pt2 = solution[j]
            d = distance(solution[i], solution[j])
            points_d_map[d] = [pt1,pt2]
            dist.append(d)
        
        # make sides having 2 points 
        lines_wrt_points = []
        for i in range(len(solution)):
            pt1 = solution[i]
            j = i + 1
            if j == len(solution):
                j = 0
                
            pt2 = solution[j]
            lines_wrt_points.append([pt1, pt2])
        
        # making sides having 1 point and slope  
        lines_wrt_slope = make_line(lines_wrt_points)   
        
        # Rotate the line wrt to the vertices which has largest side upto making the lines paralle
        self.line_rotating_wrt_point(solution_lines,points_d_map, dist, lines_wrt_slope,0,2)
        self.line_rotating_wrt_point(solution_lines,points_d_map, dist, lines_wrt_slope,1,3)

        # Solution for the rectangle line equations
        res_sol = self.solution_of_lines(solution_lines)
        
        # Converting y coordinates to positive
        for i in range(len(res_sol)):
            res_sol[i] = res_sol[i].tolist()
            res_sol[i][0]= int(res_sol[i][0])
            res_sol[i][1]= int(res_sol[i][1]) * -1
        
        
        return res_sol
        
     
    def perspective_transform(self):
        
        # Getting lines wrt points and slope
        lines_wrt_point_slope = self.make_line_with_point_slope()
        print(lines_wrt_point_slope)
        
        # mapping the vertices ABCD to the lines which are nearer 
        line_map_to_vertices = self.mapping_lines_to_vertices(lines_wrt_point_slope)
        print(line_map_to_vertices)
        
        # Getting extreme boundary lines of object in image
        boundary_lines = self.get_extreme_boundary_lines(line_map_to_vertices)
        print(boundary_lines)
        
        # Solving the solution for the extreme boundary lines
        solution = self.solution_of_lines(boundary_lines)
        
        # the point got in the list will in in numpy array we convert it to list
        for i in range(len(solution)):
            solution[i] = solution[i].tolist()
            solution[i][0]= solution[i][0]
            solution[i][1]= solution[i][1]
        
        # Finding vertices for rectangle boundary
        ## List of vertices will be in order BCDA
        res_sol = self.find_rectangle_boundaries(solution)
        print(res_sol)
        
        # Draw reecatngle Boundaries for the object
        sol_points_img = draw_boundary(self.padded_edge_detection.copy(),res_sol)
        save_img(sol_points_img, self.params['boundary_line_img_dir']+self.img_name)
        
        # Size if the object
        res_img_width = int(distance(res_sol[1],res_sol[2]))
        res_img_height = int(distance(res_sol[0],res_sol[1]))
        
        pts1 = np.float32(res_sol)
        
        # Horizontal perspective tranformation
        # Size of the Transformed Image => horizontal image
        horizontal_pts2 = np.float32([[res_img_width,0],[res_img_width,res_img_height],[0,res_img_height], [0,0]])
        
        M_horizontal = cv2.getPerspectiveTransform(pts1,horizontal_pts2)
        horizontal_transformed_image = cv2.warpPerspective(self.padded_edge_detection,M_horizontal,(res_img_width,res_img_height))
        save_img(horizontal_transformed_image, self.params['tilt_corrected_img_dir']+ 'horizontal_' +self.img_name)
        
        # vertical perspective tranformation
        # Size of the Transformed Image => vertical image
        vertical_pts2 = np.float32([[res_img_height,res_img_width],[0,res_img_width], [0,0], [res_img_height,0]])
        
        vertical = cv2.getPerspectiveTransform(pts1,vertical_pts2)
        vertical_transformed_image = cv2.warpPerspective(self.padded_edge_detection,vertical,(res_img_height,res_img_width))
        save_img(vertical_transformed_image, self.params['tilt_corrected_img_dir']+ 'vertical_' +self.img_name)
        
        
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
             
        
        

