import cv2 as cv
import os 

'''
The class aimed to utilized to rename "Dev*_Image_w1920_h1200_fn*.jpg" to 00****.jpg. For those frames in between given first_frame_number to last_frame_number
placed in the path named path_read and output path is path_write. Arguments are passed through rename_ELTECar.py

first_frame_num - starting frame number (1 if it needs to be starting from first frame)
last_frame_num - last image frame number to be renamed 
path_to_read - image path which "Dev*_Image_w1920_h1200_fn*.jpg" exists
path_to_write - path to write renamed images
cameraName - specifiying which camera images are renaming with "DEV0" or "DEV1" or "DEV2" or "DEV3"

written by Baigalmaa 2024.05
'''
class rename():
    def __init__(self, first_frame_num, last_frame_num, path_to_read, path_to_write, cameraName):
        self.first_frame_number = first_frame_num
        self.last_frame_number = last_frame_num
        self.filename_to_read = cameraName + "_Image_w1920_h1200_fn"
        self.path_read = path_to_read
        self.path_write = path_to_write
        self.img_ext = ".jpg"
        self.rename_img()


    def rename_images(self):
        for i in range(self.first_frame_number, self.last_frame_number):
            if(i < 10):
                padding = "00000"
            elif(i < 100):
                padding = "0000"
            elif(i < 1000):
                padding = "000"
            elif(i < 10000):
                padding = "00"
            if(os.path.exists(self.path_read + self.filename_to_read + str(i) + self.img_ext)):
                image_dev0 = cv.imread(self.path_read + self.filename_to_read + str(i) + self.img_ext, cv.IMREAD_GRAYSCALE)
                cv.imwrite(self.path_write + padding + str(i) + self.img_ext, image_dev0)
            else:
                print("Frame number " + str(i) + " could not found.")
                continue