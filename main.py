from src.tilt_correction import TiltCorrection
from src.image_transformation import ImageTransformer
import argparse,cv2,glob,os
from src.config import params
import numpy as np

def shear():
    img_path = 'rotated/rotated_phone.png'
    shear_factor = 1/2
    shear_matrix_x = np.float32([[1, 1, 0],
                                [0, shear_factor, 0]])
    
    # Apply the shear transformation
    image= cv2.imread('rotated/rotated_phone.png')
    # print(image)
    h, w = image.shape[:2]
    sheared_image = cv2.warpAffine(image, shear_matrix_x, (w+500, h+500))
    cv2.imwrite('rotated/shear.png',sheared_image)
    
def rotate():
    img_path = 'images/phone.png'
    it = ImageTransformer(img_path)
    it.rotate_along_axis(phi=10)
    
def main(img_dir):
    # img_path = 'rotated/rotated_phone.png'
    # it = ImageTransformer(img_path)
    # it.rotate_along_axis(theta=40)
    
    img_path = os.listdir(img_dir)
    for img in img_path:
        
        tilt_correction = TiltCorrection(tilted_obj_path= img_dir+img)
        tilt_correction.rotate_object_and_save()
        print(f'{img} completed')
        

if __name__ == "__main__":
    # it = ImageTransformer('images/phone.png',shape=(3000,3000))
    # it.rotate_along_axis(gamma=45)    
# def main(tilted_obj_path: str):
#     img_path = 'images/phone.png'
#     it = ImageTransformer(img_path)
#     it.rotate_along_axis(theta=10)
#     # tilt_correction = TiltCorrection(tilted_obj_path= tilted_obj_path)
#     # tilt_correction.rotate_object_and_save()

# if __name__ == "__main__":
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument('img_file',help='img_file')
#     # args = parser.parse_args()
#     # rotate()
    params = params()
    
    print("Image Tilt Correction started")
    # print(params['img_dir'])
    main(params['img_dir'])
    print("Image Tilt Correction ended")