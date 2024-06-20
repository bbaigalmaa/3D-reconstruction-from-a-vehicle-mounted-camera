from rename import rename_images
import sys

'''
sys.argv[1] - starting frame number (1, if it needs to be starting from first frame)
sys.argv[2] - last image frame number to be renamed 
sys.argv[3] - image path which "Dev*_Image_w1920_h1200_fn*.jpg" exists
sys.argv[4] - path to write renamed images
sys.argv[5] - specifiying which camera images are renaming with "DEV0" or "DEV1" or "DEV2" or "DEV3"

written by Baigalmaa 2024.05
'''

img_ = rename_images(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
img_.rename_images()