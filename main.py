from src.tilt_correction import TiltCorrection

def main(tilted_obj_path: str):
    tilt_correction = TiltCorrection(tilted_obj_path= tilted_obj_path)
    tilt_correction.rotate_object_and_save()

if __name__ == "__main__":
    print("Image Tilt Correction started")
    main('images/shoe.png')
    print("Image Tilt Correction ended")