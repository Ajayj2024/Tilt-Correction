from src.tilt_correction import TiltCorrection
# from src.image_transformation import ImageTransformer
import argparse, cv2, glob, os
from src.config import parameters
import numpy as np
from src.utils import *
import time
import shutil
param = parameters()

def main(img_path):
    tilt_correction = TiltCorrection(tilted_obj_path=img_path)
    tilt_correction.perspective_transform()
    print(f"{img_path} completed")

def func(img_path):
    a, b, c= 3, 1, 0
    img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    edge_detection = cv2.Canny(img_arr, 
                                  param['canny_parameters']['low_threshold'], 
                                  param['canny_parameters']['high_threshold'], 
                                  None if param['canny_parameters']['L2gradient'] == 'None' else param['canny_parameters']['L2gradient'],
                                  param['canny_parameters']['aperture_size']
                                  )
    padded_edge_detection = pad_image(edge_detection)
    non_zero_canny = np.nonzero(padded_edge_detection)
    print([padded_edge_detection.shape[0], padded_edge_detection.shape[1]])
    print([padded_edge_detection.shape[0], 0])
    ref_points = [[0,0], [0, padded_edge_detection.shape[1]], [padded_edge_detection.shape[0], padded_edge_detection.shape[1]], [padded_edge_detection.shape[0], 0]]
    corner_indices = []
    for p in ref_points:
        point_d = None
        min_d = float('inf')
        for i in range(len(non_zero_canny[0])):
            d = a*manhattan_distance_distance(p, [non_zero_canny[1][i], non_zero_canny[0][i]]) + b*euclidean_distance(p, [non_zero_canny[1][i], non_zero_canny[0][i]]) + c*minimum_value_distance(p, [non_zero_canny[1][i], non_zero_canny[0][i]])
            if d < min_d:
                min_d = d
                point_d = [non_zero_canny[1][i], non_zero_canny[0][i]]
        corner_indices.append(point_d)  
        print(min_d)
    print(corner_indices)
    img = draw_points(padded_edge_detection.copy(), corner_indices)
    return img
if __name__ == "__main__":
    
    start_time = time.time()
    
    # Intializing config.yaml
    param = parameters()
    i = 3
    # creating folders
    os.makedirs(param['result_dir'], exist_ok= True)
    os.makedirs(param['corner_extreme_img_dir'], exist_ok= True)
    os.makedirs(param['boundary_line_img_dir'], exist_ok= True)
    os.makedirs(param['tilt_corrected_img_dir'], exist_ok= True)
    os.makedirs('experiment', exist_ok= True)
    os.makedirs('experiment/experiment' + str(i), exist_ok= True)
    parser = argparse.ArgumentParser()
    parser.add_argument('img_file',help='img_file')
    args = parser.parse_args()
    

    print("Image Tilt Correction started")
    
    img = func(param['img_dir'] + args.img_file)
    save_img(img,'experiment/experiment'+ f'{i}/' +args.img_file)
    print("Image Tilt Correction ended")
    
    end_time = time.time()
    print(f"Execution Time: {end_time - start_time}")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#     # def main(tilted_obj_path: str):
#     img_path = "images/phone.png"
#     it = ImageTransformer(img_path)
#     # angles = {
#     #     'theta':[10,30,50,70],
#     #     'phi':[10,30,50,70],
#     #     'gamma':[10,30,50,70]
#     # }
#     # for a in angles.keys():
#     #     theta = phi = gamma = 0
#     #     for n in angles[a]:
#     #         if a == 'theta': theta = n
#     #         elif a == 'phi': phi = n
#     #         elif a == 'gamma': gamma = n
#     rotated_image = it.rotate_along_axis(gamma=50, theta=0, phi=-20)
#     save_image(f"rotated/phone_{-20}_{50}.png", rotated_image)
#     # tilt_correction = TiltCorrection(tilted_obj_path= tilted_obj_path)
#     # tilt_correction.rotate_object_and_save()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('img_file',help='img_file')
#     args = parser.parse_args()
#     # rotate()
#     params = params()

#     print("Image Tilt Correction started")
#     # print(params['img_dir'])
#     main(params['img_dir'] + args.img_file)
#     print("Image Tilt Correction ended")
