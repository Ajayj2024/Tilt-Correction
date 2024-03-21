from src.tilt_correction import TiltCorrection
from src.image_transformation import ImageTransformer
import argparse, cv2, glob, os
from src.config import params
import numpy as np
from src.utils import *
import time


def main(img_path):
    tilt_correction = TiltCorrection(tilted_obj_path=img_path)
    tilt_correction.perspective_transform()
    print(f"{img_path} completed")


if __name__ == "__main__":
    
    start_time = time.time()
    
    # Intializing config.yaml
    params = params()
    
    # creating folders
    os.makedirs(params['result_dir'], exist_ok= True)
    os.makedirs(params['corner_extreme_img_dir'], exist_ok= True)
    os.makedirs(params['boundary_line_img_dir'], exist_ok= True)
    os.makedirs(params['tilt_corrected_img_dir'], exist_ok= True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('img_file',help='img_file')
    args = parser.parse_args()
    

    print("Image Tilt Correction started")
    
    main(params['img_dir'] + args.img_file)
    
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
