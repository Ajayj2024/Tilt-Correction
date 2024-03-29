from src.tilt_correction import TiltCorrection
# from src.image_transformation import ImageTransformer
import argparse, cv2, glob, os
from src.config import parameters
import numpy as np
from src.utils import *
import time
import shutil
from tqdm import tqdm
param = parameters()

def main(img_path):
    tilt_correction = TiltCorrection(tilted_obj_path=img_path)
    tilt_correction.perspective_transform()
    print(f"{img_path} completed")


if __name__ == "__main__":
    
    start_time = time.time()
    
    # Intializing config.yaml
    param = parameters()
    i = 1
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
    img_dir = param['img_dir1']
    if args.img_file == '*':
        img_path = os.listdir(img_dir)
        for path in tqdm(img_path):
            main(img_dir + path)
    else:
        main(param['img_dir'] + args.img_file)
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
