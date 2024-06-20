from visual_odometry import visual_odometry
import numpy as np
import sys

if(sys.argv[1] == "kitti"):
    # Source: https://www.cvlibs.net/datasets/kitti/setup.php

    # KITTI camera Cam 0 = P0 = projection_left
    # KITTI camera Cam 1 = P1 = projection_right
    projection_left = np.array([7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00, 
                                0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00,
                                0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]).reshape(3,4)
    projection_right = np.array([7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, -3.861448000000e+02,
                                 0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00,
                                 0.000000000000e+00 ,0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]).reshape(3,4)
    
    
    distortion = np.array([0.0, 0.0, 0.0, 0.0])
    K = projection_left[:, :3]

    # Baseline given in setup 
    baseline = 0.54
    # Image extension
    img_ext = ".png"

    starting_seq = 0 # Starting frame number
    number_of_images = 1000 # Number of frames to track

else:
    # Source: https://www.hackademix.hu/wp-content/uploads/2023/06/calibration_params.txt
    # ELTECar camera DEV1 - projection_left   - DEV1 camera matrix (K) and translation at (0, 0, 0)
    # ELTECar camera DEV0 - projection_right  - DEV1 -> DEV0 parameter PoseCamera (R|t) matrix
    projection_left = np.array([1.296017692307357e+03, 0, 9.407268031732034e+02, -1.089354932150932e+02,
                                0.0,1.294832210476451e+03,5.837191315595016e+02, -0.983919606029976,
                                0.0,0.0,1.0, -14.523814320016147]).reshape(3,4)
    projection_right = np.array([1.282635220934342e+03,0,9.604166763029937e+02, 0.0,
                                 0,1.282748868123230e+03,6.369097917615544e+02, 0.0,
                                 0,0,1, 0.0]).reshape(3,4)
        
    K = projection_left[:, :3]

    # Baseline estimated by DEV0 and DEV1 camera's X coordinate values
    # Source: https://www.hackademix.hu/wp-content/uploads/2023/06/Sensor_pack_summary_2023.pdf - page 3 
    baseline = 0.7333
    distortion = np.array([0.0, 0.0, 0.0, 0.0])
    img_ext = ".jpg"
    number_of_images = 1000
    starting_seq = 92

vo = visual_odometry(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], projection_left, projection_right, distortion, baseline, img_ext, number_of_images, starting_seq)

# outlier removal - "1pts" or "2pnp" or ""
# pose estimation - "5pts" or "P3PRansac"
# vo.main("MIX", "1pts", "5pts")
vo.main("KLT", "2pnp", "P3PRansac", "output.ply")

