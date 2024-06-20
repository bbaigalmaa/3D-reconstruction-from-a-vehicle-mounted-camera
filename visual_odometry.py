import numpy as np
import glob
import cv2 as cv
import random
import math
import open3d as o3d
import os


class visual_odometry():
    def __init__(self, dataset, left_img_path, right_img_path, ground_truth_path, proj_l, proj_r, distortion, baseline, img_ext, number_of_images, start_seq):
        self.path_right_images = right_img_path
        self.path_left_images = left_img_path
        self.ground_truth_path = ground_truth_path
        self.projection_left = proj_l
        self.projection_right = proj_r
        self.K = self.projection_left[:, :3]
        self.baseline = baseline
        self.f = self.K[0][0]
        self.dataset_type = dataset
        self.images_extension = img_ext
        self.number_of_images = number_of_images
        self.img_left = []
        self.img_right = []
        self.gt_poses = []
        self.linear_LS_triangulation_C = -np.eye(2, 3)
        self.iterative_LS_triangulation_C = -np.eye(2, 3)
        self.output_dtype = float
        self.width = 0
        self.height = 0
        self.starting_sequence = start_seq
        self.distortion = distortion
        self.camera_position = []
        self.ground_truth_3d = []
        self.cam_pos = np.array([0.0 , 0.0, 0.0, 1.0]).reshape(4,1)
        self.inv_transform = np.hstack((np.eye(3), np.array([0, 0, 0]).reshape(3, 1)))
        self.traj = np.zeros((1000, 1000, 3))

    def read_imgs(self):
        '''
        Read images from initialized image paths.
        '''

        # Reading images from left and right image folder
        im_left  = glob.glob(self.path_left_images + "*" + self.images_extension)
        im_left = sorted(im_left)
        im_right = glob.glob(self.path_right_images + "*" + self.images_extension)
        im_right = sorted(im_right)

        for img in im_left[self.starting_sequence:self.number_of_images]:
            n= cv.imread(img, cv.IMREAD_GRAYSCALE)
            self.img_left.append(n)

        for img in im_right[self.starting_sequence:self.number_of_images]:
            n= cv.imread(img, cv.IMREAD_GRAYSCALE)
            self.img_right.append(n)

    def get_pose(self):
        '''
        Read ground truth trajectory from initialized ground truth path. 
        For ELTECar dataset arranged by starting point to (0, 0) and subtracting initial starting point from remaining.
        '''

        file = open(self.ground_truth_path, "r")
        for f in file:
            val = f.split(" ")
            if(self.dataset_type == "kitti"):    
                p = np.array([float(val[3]), float(val[7]), float(val[11])])
            else:
                p = np.array([float(val[0]), float(val[1])])
            self.gt_poses.append(p)

        file.close()
        
        # For ELTECar dataset to make the map visualization starting from (0, 0) 
        # Following has substracted initial starting point from all remaining trajectory coordinates
        if (self.dataset_type != "kitti"):
            gt_poses = np.array(self.gt_poses)
            g_x = gt_poses[:, 0] - gt_poses[0][0]
            g_x = g_x.reshape(g_x.shape[0], 1)
            g_y = gt_poses[:, 1] - gt_poses[0][1]
            g_y = g_y.reshape(g_y.shape[0], 1)
            gt = np.hstack((g_x, g_y))
            self.gt_poses = gt

    def distanceCalculate(self, p1, p2):
        """p1 and p2 in format (x1,y1) and (x2,y2) tuples"""
        dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        return dis

    # Linear Triangualation Source - https://www.morethantechnical.com/blog/2012/01/04/simple-triangulation-with-opencv-from-harley-zisserman-w-code/
    def linear_LS_triangulation(self, u1, P1, u2, P2):
        """
        Linear Least Squares based triangulation.
        Relative speed: 0.1
        
        (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
        (u2, P2) is the second pair.
        
        u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
        
        The status-vector will be True for all points.
        """
        A = np.zeros((4, 3))
        b = np.zeros((4, 1))
        
        # Create array of triangulated points
        x = np.zeros((3, len(u1)))
        
        # Initialize C matrices
        C1 = np.array(self.linear_LS_triangulation_C)
        C2 = np.array(self.linear_LS_triangulation_C)
        
        for i in range(len(u1)):
            # Derivation of matrices A and b:
            # for each camera following equations hold in case of perfect point matches:
            #     u.x * (P[2,:] * x)     =     P[0,:] * x
            #     u.y * (P[2,:] * x)     =     P[1,:] * x
            # and imposing the constraint:
            #     x = [x.x, x.y, x.z, 1]^T
            # yields:
            #     (u.x * P[2, 0:3] - P[0, 0:3]) * [x.x, x.y, x.z]^T     +     (u.x * P[2, 3] - P[0, 3]) * 1     =     0
            #     (u.y * P[2, 0:3] - P[1, 0:3]) * [x.x, x.y, x.z]^T     +     (u.y * P[2, 3] - P[1, 3]) * 1     =     0
            # and since we have to do this for 2 cameras, and since we imposed the constraint,
            # we have to solve 4 equations in 3 unknowns (in LS sense).

            # Build C matrices, to construct A and b in a concise way
            C1[:, 2] = u1[i, :]
            C2[:, 2] = u2[i, :]
            
            # Build A matrix:
            # [
            #     [ u1.x * P1[2,0] - P1[0,0],    u1.x * P1[2,1] - P1[0,1],    u1.x * P1[2,2] - P1[0,2] ],
            #     [ u1.y * P1[2,0] - P1[1,0],    u1.y * P1[2,1] - P1[1,1],    u1.y * P1[2,2] - P1[1,2] ],
            #     [ u2.x * P2[2,0] - P2[0,0],    u2.x * P2[2,1] - P2[0,1],    u2.x * P2[2,2] - P2[0,2] ],
            #     [ u2.y * P2[2,0] - P2[1,0],    u2.y * P2[2,1] - P2[1,1],    u2.y * P2[2,2] - P2[1,2] ]
            # ]
            A[0:2, :] = C1.dot(P1[0:3, 0:3])    # C1 * R1
            A[2:4, :] = C2.dot(P2[0:3, 0:3])    # C2 * R2
            
            # Build b vector:
            # [
            #     [ -(u1.x * P1[2,3] - P1[0,3]) ],
            #     [ -(u1.y * P1[2,3] - P1[1,3]) ],
            #     [ -(u2.x * P2[2,3] - P2[0,3]) ],
            #     [ -(u2.y * P2[2,3] - P2[1,3]) ]
            # ]
            b[0:2, :] = C1.dot(P1[0:3, 3:4])    # C1 * t1
            b[2:4, :] = C2.dot(P2[0:3, 3:4])    # C2 * t2
            b *= -1
            
            # Solve for x vector
            cv.solve(A, b, x[:, i:i+1], cv.DECOMP_SVD)
        
        return x.T.astype(self.output_dtype), np.ones(len(u1), dtype=bool)

    # Iterative Linear Triangualation Source - https://www.morethantechnical.com/blog/2012/01/04/simple-triangulation-with-opencv-from-harley-zisserman-w-code/
    def iterative_LS_triangulation(self, u1, P1, u2, P2, tolerance=3.e-5):
        """
        Iterative (Linear) Least Squares based triangulation.
        From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997.
        Relative speed: 0.025
        
        (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
        (u2, P2) is the second pair.
        "tolerance" is the depth convergence tolerance.
        
        Additionally returns a status-vector to indicate outliers:
            1: inlier, and in front of both cameras
            0: outlier, but in front of both cameras
            -1: only in front of second camera
            -2: only in front of first camera
            -3: not in front of any camera
        Outliers are selected based on non-convergence of depth, and on negativity of depths (=> behind camera(s)).
        
        u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
        """
        A = np.zeros((4, 3))
        b = np.zeros((4, 1))
        
        # Create array of triangulated points
        x = np.empty((4, len(u1))); x[3, :].fill(1)    # create empty array of homogenous 3D coordinates
        x_status = np.empty(len(u1), dtype=int)
        
        # Initialize C matrices
        C1 = np.array(self.iterative_LS_triangulation_C)
        C2 = np.array(self.iterative_LS_triangulation_C)
        
        for xi in range(len(u1)):
            # Build C matrices, to construct A and b in a concise way
            C1[:, 2] = u1[xi, :]
            C2[:, 2] = u2[xi, :]
            
            # Build A matrix
            A[0:2, :] = C1.dot(P1[0:3, 0:3])    # C1 * R1
            A[2:4, :] = C2.dot(P2[0:3, 0:3])    # C2 * R2
            
            # Build b vector
            b[0:2, :] = C1.dot(P1[0:3, 3:4])    # C1 * t1
            b[2:4, :] = C2.dot(P2[0:3, 3:4])    # C2 * t2
            b *= -1
            
            # Init depths
            d1 = d2 = 1.
            
            for i in range(10):    # Hartley suggests 10 iterations at most
                # Solve for x vector
                #x_old = np.array(x[0:3, xi])    # TODO: remove
                cv.solve(A, b, x[0:3, xi:xi+1], cv.DECOMP_SVD)
                
                # Calculate new depths
                d1_new = P1[2, :].dot(x[:, xi])
                d2_new = P2[2, :].dot(x[:, xi])
                
                # Convergence criterium
                #print i, d1_new - d1, d2_new - d2, (d1_new > 0 and d2_new > 0)    # TODO: remove
                #print i, (d1_new - d1) / d1, (d2_new - d2) / d2, (d1_new > 0 and d2_new > 0)    # TODO: remove
                #print i, np.sqrt(np.sum((x[0:3, xi] - x_old)**2)), (d1_new > 0 and d2_new > 0)    # TODO: remove
                ##print i, u1[xi, :] - P1[0:2, :].dot(x[:, xi]) / d1_new, u2[xi, :] - P2[0:2, :].dot(x[:, xi]) / d2_new    # TODO: remove
                #print bool(i) and ((d1_new - d1) / (d1 - d_old), (d2_new - d2) / (d2 - d1_old), (d1_new > 0 and d2_new > 0))    # TODO: remove
                ##if abs(d1_new - d1) <= tolerance and abs(d2_new - d2) <= tolerance: print "Orig cond met"    # TODO: remove
                if abs(d1_new - d1) <= tolerance and \
                        abs(d2_new - d2) <= tolerance:
                #if i and np.sum((x[0:3, xi] - x_old)**2) <= 0.0001**2:
                #if abs((d1_new - d1) / d1) <= 3.e-6 and \
                        #abs((d2_new - d2) / d2) <= 3.e-6: #and \
                        #abs(d1_new - d1) <= tolerance and \
                        #abs(d2_new - d2) <= tolerance:
                #if i and 1 - abs((d1_new - d1) / (d1 - d_old)) <= 1.e-2 and \    # TODO: remove
                        #1 - abs((d2_new - d2) / (d2 - d1_old)) <= 1.e-2 and \    # TODO: remove
                        #abs(d1_new - d1) <= tolerance and \    # TODO: remove
                        #abs(d2_new - d2) <= tolerance:    # TODO: remove
                    break
                
                # Re-weight A matrix and b vector with the new depths
                A[0:2, :] *= 1 / d1_new
                A[2:4, :] *= 1 / d2_new
                b[0:2, :] *= 1 / d1_new
                b[2:4, :] *= 1 / d2_new
                
                # Update depths
                #d_old = d1    # TODO: remove
                #d1_old = d2    # TODO: remove
                d1 = d1_new
                d2 = d2_new
            
            # Set status
            x_status[xi] = ( i < 10 and                       # points should have converged by now
                            (d1_new > 0 and d2_new > 0) )    # points should be in front of both cameras
            if d1_new <= 0: x_status[xi] -= 1
            if d2_new <= 0: x_status[xi] -= 2
        
        return x[0:3, :].T.astype(self.output_dtype), x_status

    # Polynomial Triangualation Source - https://www.morethantechnical.com/blog/2012/01/04/simple-triangulation-with-opencv-from-harley-zisserman-w-code/
    def polynomial_triangulation(self, u1, P1, u2, P2):
        """
        Polynomial (Optimal) triangulation.
        Uses Linear-Eigen for final triangulation.
        Relative speed: 0.1
        
        (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
        (u2, P2) is the second pair.
        
        u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
        
        The status-vector is based on the assumption that all 3D points have finite coordinates.
        """
        P1_full = np.eye(4); P1_full[0:3, :] = P1[0:3, :]    # convert to 4x4
        P2_full = np.eye(4); P2_full[0:3, :] = P2[0:3, :]    # convert to 4x4
        P_canon = P2_full.dot(cv.invert(P1_full)[1])    # find canonical P which satisfies P2 = P_canon * P1
        
        # "F = [t]_cross * R" [HZ 9.2.4]; transpose is needed for numpy
        F = np.cross(P_canon[0:3, 3], P_canon[0:3, 0:3], axisb=0).T
        
        # Other way of calculating "F" [HZ (9.2)]
        #op1 = (P2[0:3, 3:4] - P2[0:3, 0:3] .dot (cv2.invert(P1[0:3, 0:3])[1]) .dot (P1[0:3, 3:4]))
        #op2 = P2[0:3, 0:4] .dot (cv2.invert(P1_full)[1][0:4, 0:3])
        #F = np.cross(op1.reshape(-1), op2, axisb=0).T
        
        # Project 2D matches to closest pair of epipolar lines
        u1_new, u2_new = cv.correctMatches(F, u1.reshape(1, len(u1), 2), u2.reshape(1, len(u1), 2))
        
        
        # For a purely sideways trajectory of 2nd cam, correctMatches() returns NaN for all possible points!
        if np.isnan(u1_new).all() or np.isnan(u2_new).all():
            F = cv.findFundamentalMat(u1, u2, cv.FM_7POINT)[0]    # so use a noisy version of the fund mat
            u1_new, u2_new = cv.correctMatches(F, u1.reshape(1, len(u1), 2), u2.reshape(1, len(u1), 2))

        
        
        # Triangulate using the refined image points
        return self.linear_LS_triangulation(u1_new[0], P1, u2_new[0], P2)    

    def least_square_reprojection_error(self, first_pts, world_coords, proj):
        '''
        Least square error by projection world coordinates into 2D

        first_pts    -    set of 2D coordinates
        world_coords -    corresponding 3D coordinates
        proj         -    Projection matrix to convert 3D to 2D

        Returns: Errors between given first_pts and projected 2D points
        '''

        # Homogeneous coordinate conversion
        world_coords_hom = np.hstack((world_coords, np.ones((world_coords.shape[0], 1))))

        # 3D points projection
        pixel_coords = proj@np.transpose(world_coords_hom)
        pixel_coords = np.transpose(pixel_coords)

        # Homogenous division by last coordinate 
        pixel_coords[:, 0 ] = pixel_coords[:, 0]/pixel_coords[:, 2]
        pixel_coords[:, 1 ] = pixel_coords[:, 1]/pixel_coords[:, 2]
        pixel_coords[:, 2 ] = pixel_coords[:, 2]/pixel_coords[:, 2]

        # Error calculation by square root of (x2 - x1)^2 + (y2 - y1)^2
        x = pixel_coords[:, 0] - first_pts[:, 0]
        y = pixel_coords[:, 1] - first_pts[:, 1]
        error = np.sqrt(x*x + y*y)

        return error

    def best_triangulation(self, first_points, second_points):
        '''
        The method triangulates pair of points by 4 ways of calculation and returns results which has lowest mean reprojection error.
        Ray emitting from given 2 pairs of points forms triangular plane to estimate 3d points of the matching pair.

        Input:
            first_points  - one pair of the correspondent points set
            second_points - another pair

        Output:
            pts_l   -  first pair of points, lower than 1000, with the lowest mean reprojection error
            pts_r   -  second pair of points, lower than 1000, with the lowest mean reprojection error
            world   -  wolrd coordinates whose reprojection is close to given point pairs, first_points and second_points, respectively
            err_l   -  error information of first pairs of points
            err_r   -  error of second pairs of points
            inliers -  filtered pattern that eliminates points greater than 1000

        In addition, best_tr variable stores name of the method used for the estimation.

        ''' 

        # Four method for triangulation estimation including OpenCV triangulation function
        tr_methods  = ["linearLS", "iterativeLS", "polynomial", "opencv"] 
        threshold = np.inf
        best_tr = ""

        err_l = []

        for m in tr_methods:
            if(m == "linearLS"):
                world_coords = self.linear_LS_triangulation(first_points, self.projection_left, second_points, self.projection_right)[0]
            elif(m=="iterativeLS"):
                # Fastest estimation out of Linear, Iterative and Polynomials
                world_coords = self.iterative_LS_triangulation(first_points, self.projection_left, second_points, self.projection_right)[0]
            elif("polynomial"):
                world_coords = self.polynomial_triangulation(first_points, self.projection_left, second_points, self.projection_right)[0]
            else:
                # OpenCV Triangulation function
                world_coords = cv.triangulatePoints(self.projection_left, self.projection_right, np.transpose(first_points), np.transpose(second_points))
                world_coords = np.transpose(world_coords[:3, :])

            # Creating mask to filter points 
            mask = np.ones((world_coords.shape[0]), dtype=bool)
            mask[(world_coords[:, 2] <= 0) | (world_coords[:, 0] >= 1000) | (world_coords[:, 2] >= 1000) | (world_coords[:, 1] >= 1000)] = False
            
            # Filtering pair of point and triangulated world coordinates
            world_coords = world_coords[mask, :]
            first_points_masked = first_points[mask, :]
            second_points_masked = second_points[mask, :]

            # Reprojection error estimation for first pair and second pair
            err_first = self.least_square_reprojection_error(first_points_masked, world_coords, self.projection_left)
            err_second = self.least_square_reprojection_error(second_points_masked, world_coords, self.projection_right)


            # To find best triangulation, here filtering by lowest mean reprojection error
            if((np.mean(np.abs(err_first)) < threshold)):
                threshold = np.mean(np.abs(err_first))
                err_l = err_first
                err_r = err_second
                pts_l = first_points_masked
                pts_r = second_points_masked
                world = world_coords
                best_tr = m
                inliers = mask

        return pts_l, pts_r, world, err_l, err_r, inliers

    def feature_matching(self, img_left, img_right, type, matcher):
        '''
        The method has 2 different functions:
            1. Feature match - on 2 images (given img_left and img_right) - using SIFT or ORB matchers
            2. Corner Detection - on given img_left - using Shi-Tomasi corner detection

        Input: 
            img_left   -   left pair of stereo image or previous image of sequence
            img_right  -   right pair of stereo image or next image of sequence
            type       -   "SIFT" or "ORB" or "cornerDetection"
            matcher    -    "BF" or "FLANN" or ""(in case of corner detection)

        1. Output for Feature match:
            first_points     -   corresponding first pairs
            second_points    -   corresponding second pairs
            world_coords     -   triangulated world coordinates
            err_first        -   first pairs reprojection error
            err_second       -   second pairs reprojection error

        2. Output for Corner Detection:
            corners          -   detected corners on left pair of image or previous image of sequence            

        '''

        # SIFT feature matching 
        if(type == "SIFT"):
            sift = cv.SIFT_create()
            
            kp_l, desc_l = sift.detectAndCompute(img_left, None)
            kp_r, desc_r = sift.detectAndCompute(img_right, None)

            if(matcher == "BF"):
                matcher = cv.BFMatcher.create(cv.NORM_L1) 
            else:

                # FLANN parameters
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
                search_params = dict(checks=50) 
                
                matcher = cv.FlannBasedMatcher(index_params,search_params)
        
            # Feature matching from two descriptors
            match = matcher.knnMatch(desc_l,desc_r,k=2)

        # ORB feature matching  
        elif(type == "ORB"):
            # Initialize the ORB detector algorithm 
            orb = cv.ORB_create(500) 
            
            kp_l, desc_l = orb.detectAndCompute(img_left,None) 
            kp_r, desc_r = orb.detectAndCompute(img_right,None) 
            
            if(matcher == "BF"):
                matcher = cv.BFMatcher.create(cv.NORM_HAMMING) 
            else :
                FLANN_INDEX_LSH = 6
                index_params= dict(algorithm = FLANN_INDEX_LSH,
                                table_number = 6, # 12
                                key_size = 12, # 20
                                multi_probe_level = 1) #2
            
                matcher = cv.FlannBasedMatcher(index_params)

            # Feature matching from two descriptors
            match = matcher.knnMatch(desc_l,desc_r,k=2)
        
        # Shi-Tomasi corner detection
        elif(type == "cornerDetection"):
            corners = cv.goodFeaturesToTrack(img_left, 5000,0.01,10) #5000 - maxCorners, 0.01 - qualityLevel, 10 - minDistance
            corners = np.int32(corners)
            corners = corners.reshape(corners.shape[0], corners.shape[2])

            # Filtering corners in image frame
            mask = np.zeros((corners.shape[0]), dtype=bool)
            mask[(corners[:, 0] > 0) & (corners[:, 0] < self.width) & (corners[:, 1] > 0) & (corners[:, 1] < self.height)] = True
            corners = corners[mask,:]

            return corners

        
        matchFiltered = []
        ratio_thresh = 0.7
        # Filtering matched points by distance lower than 0.7(distance threshold) - to get more accurate correspondences between points
        for m in match:
            if(len(m) > 1):
                if (m[0].distance < (ratio_thresh*m[1].distance)):
                    matchFiltered.append((m[0], ))
                elif(len(m) == 1):
                    matchFiltered.append((m[0], ))
                else:
                    continue
                    
        best_fm = matchFiltered
        first_points = []
        second_points = []
        # Separating keypoints into source (first) and destination (second) points
        for p in best_fm:
            first_points.append(kp_l[p[0].queryIdx].pt)
            second_points.append(kp_r[p[0].trainIdx].pt)

        first_points = np.array(first_points, dtype=np.float32)
        second_points = np.array(second_points, dtype=np.float32)
        
        # Triangulating the correspondences to estimate 3D points
        first_points, second_points, world_coords, err_first, err_second, inliers = self.best_triangulation(first_points, second_points)


        return first_points, second_points, world_coords, err_first, err_second

    def get_world_coords_from_disparity(self, pts, disparity):
        '''
        Estimation of world coordinates by left image points and disparity of stereo pair. Using the the formula below:
            Z = b*f/d
            X = (x - p_x)*Z/f
            Y = (y - p_y)*Z/f,  where b is baseline, 
                                      f is focal length, 
                                      d is disparity at a point,
                                      p_x and p_y are principal points.

        Input:
            pts       -  stereo pair left image points
            disparity -  disparity calculated by stereo images

        Output:
            world_coords - 3D points 
        '''
        
        world_coords = []

        for p in pts:
            if(disparity[int(p[1])][int(p[0])] > 0):
                w_z = (self.baseline * self.f)/disparity[int(p[1])][int(p[0])]
                w_x = (p[0] - self.K[0][2])*w_z/self.f
                w_y = (p[1] - self.K[1][2])*w_z/self.f

                world_coords.append([w_x, w_y, w_z])

        world_coords = np.array(world_coords)

        return world_coords

    def feature_matching_corner_detection_with_disparity_map(self, il, ir):
        '''
        Detect corner features and world coordinates estimation by disparity map, utilized Semi-Global Block Matching(SGBM).

        Input:
            il   -   left pair of stereo image
            ir   -   right pair of stereo image

        Output:
            pts_l         -   features points of left image
            pts_r         -   Empty list - [] due to no feature points in right image 
            world_coords  -   3D points
            err_first     -   reprojection error of left image points and 3D points
            err_r         -   infinity - np.inf because there is not right image point correspondence
        '''

        # Shi-tomasi corner detection
        pts_left = self.feature_matching(il, ir, "cornerDetection", "") # 5000 features

        # Semi-Global Block matching (SGBM)
        window_size = 3
        min_disp = 16
        num_disp = 112-min_disp
        stereo = cv.StereoSGBM_create(minDisparity = min_disp,
                    numDisparities = num_disp,
                    blockSize = 16,
                    P1 = 8*3*window_size**2,
                    P2 = 32*3*window_size**2,
                    disp12MaxDiff = 1,
                    uniquenessRatio = 10,
                    speckleWindowSize = 100,
                    speckleRange = 32
                )
        disp = stereo.compute(il, ir).astype(np.float32) / 16.0

        # World coordinate
        world_coords_in_cam_coords = self.get_world_coords_from_disparity(pts_left, disp)

        # Filtering on world coordinate
        mask = np.ones((world_coords_in_cam_coords.shape[0]), dtype=bool)
        mask[(world_coords_in_cam_coords[:, 2] <= 0) | (world_coords_in_cam_coords[:, 0] >= 1000) | (world_coords_in_cam_coords[:, 2] >= 1000) | (world_coords_in_cam_coords[:, 1] >= 1000)] = False
        world_coords = world_coords_in_cam_coords[mask, :]

        # Same filter applied for left feature points
        pts_l = pts_left[mask, :]

        # Reprojection error
        err_first = self.least_square_reprojection_error(pts_l, world_coords, self.projection_left)
        # print("Error in KLT disparity world reprojected into left image: ", np.mean(np.abs(err_first)))

        # Set up right points and error to empty [] and infinity(np.inf), respectively
        pts_r = []
        err_r = np.inf
        
        return pts_l, pts_r, world_coords, err_first, err_r

    def feature_matching_final(self, il, ir, method):
        '''
        Function for corner detection or feature matchers 

        Input:
            il      -  left image of stereo pair or previous frame of consequent sequence
            ir      -  right image of stereo pair or next frame of consequent sequence
            method  -  "KLT" or "SIFT"
                        "KLT" - Kanade Lucas Tomasi - for corner detection
                        "SIFT" - SIFT with FLANN matcher 

        Output:
            pts_l   -  corresponding left image points 
            pts_r   -  if method equals "SIFT" then corresponding right image points
                       if method is "KLT" then empty list - []
            world   -  3D points
            err_l   -  Reprojection error of left image points and 3D points
            err_r   -  if method equals "SIFT" then reprojection error of right image points and 3D points
                       if method equals "KLT" then it is infinity (np.inf)
        '''
        if(method == "KLT"):
            pts_l, pts_r, world, err_l, err_r = self.feature_matching_corner_detection_with_disparity_map(il, ir)
        elif(method == "SIFT"):
            pts_l, pts_r, world, err_l, err_r = self.feature_matching(il, ir, "SIFT", "FLANN")

        return pts_l, pts_r, world, err_l, err_r
    
    def track(self, refImg, currImg, pts_left, world_coords):
        '''
        Optical flow feature points correspondences in consecutive frames. 
        OpenCV optical flow calculation is calcOpticalFlowPyrLK which takes previous frame, current frame and previous frame feature points as argument. 
        Then returns corresponding current frame image points. This leads to corresponding consecutive frame feature points.

        Input:
            refImg       -   reference image (previous frame/previous image)
            currImg      -   current image (current frame)
            pts_left     -   left image feature points
            world_coords -   3D points correspond to left image feature points

        Output:
            dst_pts      -   desctination points (current image feature points)
            world        -   3D points
            src_pts      -   source points (previous image feature points)
    
        '''
        pts_left = pts_left.astype(np.float32)

        # Calculating current image feature points correspond to previous image
        next_pts, status, err = cv.calcOpticalFlowPyrLK(refImg, currImg, pts_left, None)
        status = status.ravel().astype(bool)

        # Filtering points 
        dst_pts = next_pts[status]
        src_pts = pts_left[status]
        world = world_coords[status]

        # Create filter for current image feature points (in the image frame/ inside the image)
        mask = np.zeros((dst_pts.shape[0]), dtype=bool)
        mask[(dst_pts[:, 0] > 0)& (dst_pts[:, 0] < self.width)& (dst_pts[:, 1] < self.height) & (dst_pts[:, 1] > 0) ] = True

        # Application of filter
        dst_pts = dst_pts[mask, :]
        src_pts = src_pts[mask, :]
        world = world[mask, :]

        return dst_pts, world, src_pts

    def p3p_transformation(self, world_coords, points):
        '''
        RANSAC iteration for Perspective-N-Point problem on random 4 points. 

        Input: 
            world_coords  -  3D point
            points        -  Corresponding 2D point

        Output:
            best_r        -  Rotation matrix
            best_t        -  Translation matrix
            threshold     -  mean distance of 2D point sets
            inliers       -  Inliers mask
        '''

        threshold = np.inf
        best_r = np.zeros((3,3))
        best_t = np.zeros((3,1))

        # Random points to estimation pose with OpenCV Perspective-3-Point solution
        number_of_random_points = 4

        # RANSAC maximum iteration
        number_of_Ransac_Iteration = np.log(1 - 0.99) / np.log(1 - pow(1 - 0.75, number_of_random_points))
        # print("number_of_Ransac_Iteration: ", number_of_Ransac_Iteration)

        # Iteration start
        for i in range(int(number_of_Ransac_Iteration)):
            # Saving 4 random points indeces in between 0 to total number of points
            random_index = random.sample(range(0, points.shape[0]), number_of_random_points)

            # Filtering the indeces in 2D and 3D point sets
            random_pts = points[random_index]
            random_world = world_coords[random_index]

            # Estimating pose by Perspective-N-Points OpenCV solution, 
            # where returning value rvec - rotation vector and tvec - translation vector
            _, rvec, tvec = cv.solvePnP(random_world, random_pts, self.K, self.distortion, useExtrinsicGuess=False, flags=cv.SOLVEPNP_P3P)


            if((rvec is None) | (tvec is None)):
                continue

            # Converting rotation vector to rotation matrix
            rmat, jacobian = cv.Rodrigues(rvec)

            # Inverse rigid body transformation
            rmat1 = np.transpose(rmat) # Rotation matrix inverse = Rotation matrix transpose
            tvec1 = (-rmat1 @ tvec)    # Translation = - (Rotation matrix inverse * translation vector)
            tvec1 = tvec1.ravel()
            proj = np.hstack((rmat, tvec.reshape(3,1))) # Projection matrix combined with Rotation matarix inverse and Translation

            # Converting 3D points to homogenous coordinates by adding 1s as last column
            world_coords_hom = np.hstack((world_coords, np.ones((world_coords.shape[0], 1))))

            # Reprojecting 3D points into image plane
            world_projected = self.K@proj@np.transpose(world_coords_hom)
            world_projected = np.transpose(world_projected)
            
            # Homogenous division
            world_projected[:, 0] = world_projected[:, 0]/world_projected[:, 2]
            world_projected[:, 1] = world_projected[:, 1]/world_projected[:, 2]
            world_projected[:, 2] = world_projected[:, 2]/world_projected[:, 2]

            # Taking only first 2 columns to get pixel coordinates
            world_to_pixel = world_projected[:, :2]


            # Calculating distance between reprojected 2D points and given 2D points
            distance = 0
            d = []
            for p in range(points.shape[0]):
                p1 = (points[p][0], points[p][1])
                p2 = (world_to_pixel[p][0], world_to_pixel[p][1])
                di = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
                distance += di
                d.append(di)
            # Distance mean
            distance /= points.shape[0]

            # Converting list of distances to numpy array
            d = np.array(d).reshape(len(d), )

            # Creating inliers filters by distance
            mask = np.ones((d.shape), dtype=bool)
            mask[(d > np.abs(distance))] = False

            # Selecting best pose estimation 
            if(distance < threshold):
                threshold = distance
                inliers = mask 
                best_r = rmat
                best_t = tvec

        
        return best_r, best_t, threshold, inliers

    def convert_to_spherical_coords(self, pts1):
        '''
        Converting 2D coordinates to Spherical coordinates using camera matrix

        Input:
            pts1          -  2D points

        Output:
            pts1_sphere   -  Converted spherical coordinates

        '''

        # In case of more than 1 coordinates it is adding multiple 1s as last column
        if(pts1.shape[0] > 1):
            pts1_hom = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
            pts1_sphere = np.transpose(self.K@np.transpose(pts1_hom))    
            pts1_sphere[:, 0] = pts1_sphere[:, 0]/pts1_sphere[:,2]
            pts1_sphere[:, 1] = pts1_sphere[:, 1]/pts1_sphere[:,2]
            pts1_sphere[:, 2] = pts1_sphere[:, 2]/pts1_sphere[:,2]
        # If the given a 2D point, homogenous conversion as follows
        else:
            pts1_hom = np.array([pts1[0][0], pts1[0][1], 1])
            pts1_sphere = np.transpose(self.K@np.transpose(pts1_hom))    
            pts1_sphere /= pts1_sphere[2]

        return pts1_sphere

    def convert_to_unit_spherical_coords(self, pts1):
        '''
        Conversion of unit spherical coordinates where each points length equals 1.

        Input:
            pts1     -  2D points 

        Output:
            result   -  Spherical coordinates with unit length
        '''

        # In case of more than 1 coordinates it is adding multiple 1s as last column
        if(pts1.shape[0] > 1):
            pts1_hom = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
            pts1_sphere = np.transpose(self.K@np.transpose(pts1_hom))    
            pts1_sphere_len = np.sqrt(pts1_sphere[:, 0]**2 + pts1_sphere[:, 1]**2 + pts1_sphere[:, 2]**2)
            pts1_sphere_len = pts1_sphere_len.reshape(pts1_sphere_len.shape[0], 1)

        # If the given a 2D point, homogenous conversion as follows
        else:
            pts1_hom = np.array([pts1[0][0], pts1[0][1], 1])
            pts1_sphere = np.transpose(self.K@np.transpose(pts1_hom))    
            pts1_sphere_len = np.sqrt(pts1_sphere[0]**2 + pts1_sphere[1]**2 + pts1_sphere[2]**2)

        result = pts1_sphere/pts1_sphere_len

        return result


    def unit_vector_from_2d(self, pts1):
        '''
        Converting homogenous coordinates tto unit vector

        Input:
            pts1        -   2D points
        
        Output:
            pts1_unit   -   Homogenous unit vector
        
        '''

        # In case of more than 1 coordinates it is adding multiple 1s as last column
        if(pts1.shape[0] > 1):
            pts1_hom = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
            pts1_len = np.sqrt(pts1_hom[:, 0] * pts1_hom[:, 0] + pts1_hom[:, 1] * pts1_hom[:, 1] + pts1_hom[:, 2] * pts1_hom[:, 2])
            pts1_unit = pts1_hom/pts1_len.reshape(pts1_len.shape[0], 1)
        # If the given a 2D point, homogenous conversion as follows
        else:
            pts1_hom = np.array([pts1[0][0], pts1[0][1], 1])
            pts1_len = np.sqrt(pts1_hom[0] ** 2 + pts1_hom[1] ** 2 + pts1_hom[2] ** 2)
            pts1_unit = pts1_hom/pts1_len

        return pts1_unit

    def t_skew(self, t):

        '''
        Convert translation vector to skew symmetric matrix.

        Input:
            t       -  Translation vector
        
        Output:
            t_skew  -  Skew-symmetric matrix
        '''

        t_skew = np.array([0.0, -t[2], t[1], 
                            t[2], 0.0, -t[0], 
                            -t[1], t[0], 0.0]).reshape(3,3)
        
        return t_skew

    def motion_from_2_angles(self, theta, pi):
        '''
        Planar motion estimation by given angles. Rotation around y axis and translation z and x axis.

        Input:
            theta      -    Rotation angle
            pi         -    Translation angle

        Output:
            R          -    Rotation matirx
            t          -    Translation vector
        '''

        # Translation in x and z axis
        t = np.array([np.sin(pi), 0.0 ,np.cos(pi)]).reshape(3,)
        
        # Rotation around y axis
        R = np.array([np.cos(theta), 0.0, np.sin(theta), 
                        0.0,          1.0,     0.0, 
                        -np.sin(theta), 0.0, np.cos(theta)]).reshape(3,3)
        
        return R, t

    def sampson_approximation(self, first_pts, second_pts, E):
        '''
        Sampson approximation - geometric cost function between corresponding 2 point pairs 
        according to the book named "Multiple View Geometry" by Richard Hartley and Andrew Zisserman
        The function also returns corrected point pairs.

        Input:
            first_pts         -  first pair of corresponding points
            second_pts        -  second pair of corresponding points
            E                 -  Essential matrix

        Output:
            corrected_first    -  corrected first pair 
            corrected_second   -  corrected second pair
            err                -  error between corresponding points
        '''
        corrected_first = []
        corrected_second = []
        err = []


        # Converting given 2D points into homogenous unit vectors
        first_pts_unit = self.unit_vector_from_2d(first_pts)
        second_pts_unit = self.unit_vector_from_2d(second_pts)

        # Estimating error for each points
        for i in range(first_pts.shape[0]):
            x = first_pts[i][0]
            y = first_pts[i][1]
            x_ = second_pts[i][0]
            y_ = second_pts[i][1]
            vec1 = np.array([x, y, x_, y_]).reshape(4,1)

            # Implementation of the formula - "Multiple View Geometry" chapter 12, page 315
            # ---------------------- Start ------------------
            p1_hom = first_pts_unit[i].reshape(3,1)
            p2_hom = second_pts_unit[i].reshape(3,1)

            Ex = E@p1_hom
            E_x = np.transpose(E)@p2_hom

            Ex1 = Ex[0]
            Ex2 = Ex[1]

            E_x1 = E_x[0]
            E_x2 = E_x[1]

            x_Ex = np.transpose(p2_hom)@E@p1_hom
            denominator = Ex1*Ex1 + Ex2*Ex2 + E_x1*E_x1 + E_x2*E_x2

            vec2 = np.array([E_x1, E_x2, E_x1, E_x2]).reshape(4,1)

            sol = vec1 - (x_Ex/denominator)*vec2
            # ---------------------- End ------------------

            # Conbining each error and corresponding point to list
            err.append(x_Ex**2/denominator)
            corrected_first.append([sol[0], sol[1]])
            corrected_second.append([sol[2], sol[3]])

        # Converting the list to numpy array
        corrected_first = np.array(corrected_first).reshape(len(corrected_first), 2)
        corrected_second = np.array(corrected_second).reshape(len(corrected_second), 2)
        err = np.array(err).reshape(len(err))

        # Filtering first pair of the points which are inside the image frame
        mask = np.ones((corrected_first.shape[0]), dtype=bool)
        mask[(corrected_first[:, 0] < 0) & (corrected_first[:, 0] > self.width)] = False

        # Applying filter
        corrected_first = corrected_first[mask, :]
        corrected_second = corrected_second[mask, :]

        return corrected_first, corrected_second, err

    def directional_error(self, src_pts, dst_pts, E):

        '''
        Source: "Exact Two–Image Structure from Motion" by John Oliensis
        Directional error estimation from above source formula (13) explained in page 9.

        Input:
            src_pts     -   first pair of corresponding pairs
            dst_pts     -   second pair od corresponding pairs
            E           -   Essential matrix

        Output:
            total_err   -   Directional Error between 2 corresponding points

        '''

        # Converting points into homogenous unit vector
        src_pts_unit = self.unit_vector_from_2d(src_pts)
        dst_pts_unit = self.unit_vector_from_2d(dst_pts)

        # Directional error estimation by Essential matrix
        # --------------------- Start ---------------------
        total_err = []
        for i in range(src_pts_unit.shape[0]):
            p0 = src_pts_unit[i]
            p1 = dst_pts_unit[i]
            A_m = p0@np.transpose(E)@E@np.transpose(p0) + p1@E@np.transpose(E)@np.transpose(p1)
            B_m = p1@E@np.transpose(p0)**2
            err = A_m/2 - np.sqrt(np.abs((A_m**2)/4 - B_m))
            total_err.append([err])

        total_err = np.array(total_err).ravel()
        # --------------------- End ---------------------

        return total_err

    def find_theta(self, src_pts, dst_pts, number_of_points):
        '''
        Source: "1-Point-RANSAC Structure from Motion for Vehicle-Mounted Cameras by Exploiting Non-holonomic Constraints" by Davide Scaramuzza
        Rotational angle estimation under planar motion constraints. In this case, rotation around y axis and translation in x and z axis

        Input:
            src_pts             -   first pair of corresponding points
            dst_pts             -   second pair of corresponding points
            number_of_points    -   number of points

        Output:
            theta               -   Rotational angle
        '''

        # Converting given points to spherical unit coordinates
        first_pts_hom = self.convert_to_unit_spherical_coords(src_pts)
        second_pts_hom = self.convert_to_unit_spherical_coords(dst_pts)

        # In case of, rotation angle estimation on more than 1 points
        if(number_of_points > 1):
            # In our case, rotation is around y axis and translation is in x and z axis, which is also considered as planar motion.
            # Such that initial formula has changed accordingly.
            xy_ = first_pts_hom[:, 0] * second_pts_hom[:, 1]
            yx_ = first_pts_hom[:, 1] * second_pts_hom[:, 0]
            zy_ = first_pts_hom[:, 2] * second_pts_hom[:, 1]
            yz_ = first_pts_hom[:, 1] * second_pts_hom[:, 2]
            row = np.array([xy_-yx_, -yz_-zy_])
            row = np.transpose(row)
            row_len = np.sqrt(row[:, 0]**2 + row[:, 1]**2)
            row[:, 0] = row[:, 0]/row_len
            row[:, 1] = row[:, 1]/row_len

            M = row.reshape(first_pts_hom.shape[0],2)
            u, s, vt = np.linalg.svd(np.transpose(M)@M, full_matrices=True)

            sol = vt[np.argmin(s)]
            sol_len = np.sqrt(sol[0]**2 + sol[1]**2)
            e = sol/sol_len

            theta = 2*math.atan2(e[1], e[0])
        # Rotation angle estimation on only one pair of correspondence 
        else:
            x = (first_pts_hom[0]*second_pts_hom[1] - first_pts_hom[1]*second_pts_hom[0])
            y = (first_pts_hom[1]*second_pts_hom[2] + first_pts_hom[2] * second_pts_hom[1])
            theta = -2*(math.atan2(x, y))

        return theta
        
    def histogram_voting_1pts(self, src_pts, dst_pts):
        '''
        Source: "1-Point-RANSAC Structure from Motion for Vehicle-Mounted Cameras by Exploiting Non-holonomic Constraints" by Davide Scaramuzza
        Histogram voting outlier removal implementation from the source (page 79) by median.

        Input:
            src_pts        -    first pair of corresponding point pairs
            dst_pts        -    second pair of corresponding point pairs
        
        Output:
            inliers_src    -    filtered first pair (inlier points) by threshold
            inliers_dst    -    filtered second pair (inlier points) by threshold
            mask           -    applied filter
        
        '''

        number_of_points = 1

        # Rotation angle estimation for each points
        # ---------------- Start -----------------
        theta_list = []
        for i in range(src_pts.shape[0]):
            p1 = src_pts[i].reshape(1,2)
            p2 = dst_pts[i].reshape(1,2)
            the_combined = self.find_theta(p1, p2, number_of_points)
            theta_list.append(the_combined)
        theta_list = np.array(theta_list)
        theta = np.median(theta_list)
        # ---------------- End -----------------

        # Here translation angle can be estimated as half of rotation angle.
        # Converting rotation and translation angle to Rotation and translation matrix
        R, t = self.motion_from_2_angles(theta, theta/2)

        # Essential matrix (E) = translation matrix * rotation matrix
        E = self.t_skew(t)@R

        # Error estimation by both directional and sampson approximation
        err_dir = self.directional_error(src_pts, dst_pts, E)
        _,_, err_samp = self.sampson_approximation(src_pts, dst_pts, E)

        # Choosing the smallest one as estimated error 
        if((np.abs(np.mean(err_samp)) < np.abs((np.mean(err_dir))))):
            err = err_samp
        else:
            err = err_dir

        # Creating filter mask for the error. 
        # Here the paper was suggested to set it as greater than 1 pixel, but this was not suitable in our case.
        mask = np.ones((err.shape[0]), dtype=bool)
        mask[err > 0.01] = False # greater than 1 pixel 4.65 µm * 10^-6 = 0.00000465

        # Taking mean error
        mean_err = np.abs(np.mean(err))
        print("Mean error of motion with directional error: ", mean_err)

        # Applying filter to corresponding points
        inliers_src =  src_pts[mask, :]
        inliers_dst =  dst_pts[mask, :]

        return inliers_src, inliers_dst, mask

    def two_pts_pnp(self, world_coords, cam_coords): #only for 2 point correspondences
        '''
        Source: "A New 2-Point Absolute Pose Estimation Algorithm under Plannar Motion" by Sung-In Choi and Soon-Yong Park
        Implementation of absolute pose estimation using corrsponding 3D and 2D points. This implemenation is for two 3D to 2D point correspondences.

        Input:
            world_coords     -   3D points
            cam_coords       -   2D points corrsepond with 3D points

        If determinant equals 0      
            Output: [] 
        
        else:
            Output: 
                theta        -   Rotation angle
                t            -   translation vector
                proj         -   Projection matrix
        '''


        # Focal points
        fx = self.K[0][0]
        fy = self.K[1][1]
        # Principal points
        cx = self.K[0][2]
        cy = self.K[1][2]

        # First corresponding 3D coordinate and its separation with coordinates
        w1 = world_coords[0]
        w1_x = w1[0]
        w1_y = w1[1]
        w1_z = w1[2]

        # Second corresponding 3D coordinate and its separation with coordinates
        w2 = world_coords[1]
        w2_x = w2[0]
        w2_y = w2[1]
        w2_z = w2[2]

        # First corresponding 2D coordinate
        p1 = cam_coords[0]
        p1_x = p1[0]
        p1_y = p1[1]

        # Seconde corresponding 2D coordinate
        p2 = cam_coords[1]
        p2_x = p2[0]
        p2_y = p2[1]

        # A and B matrices by formula (4) given at page 644
        A = np.array([fx*w1_z + w1_x*p1_x - w1_x*cx,    fx*w1_x + w1_z*cx - w1_z*p1_x,    fx,    cx - p1_x, 
                    w1_x*p1_y - w1_x*cy,              w1_z*cy - w1_z*p1_y,              0.0,   cy - p1_y,
                    fx*w2_z + w2_x*p2_x - w2_x*cx,    fx*w2_x + w2_z*cx - w2_z*p2_x,    fx,    cx - p2_x,
                    w2_x*p2_y - w2_x*cy,              w2_z*cy - w2_z*p2_y,              0.0,   cy - p2_y]).reshape(4, 4)
        B = np.array([0.0, -fy*w1_y, 0.0, -fy*w2_y]).reshape(4,1)


        # A' and B' matrices by formula (12) given at page 645
        # --------------------------- Start -------------------------
        a0 = A[:, 0]
        a1 = A[:, 1]
        a2 = A[:, 2]
        a3 = A[:, 3]

        a11 = a0[0]
        a21 = a0[1]
        a31 = a0[2]
        a41 = a0[3]

        a12 = a1[0]
        a22 = a1[1]
        a32 = a1[2]
        a42 = a1[3]

        a13 = a2[0]
        a23 = a2[1]
        a33 = a2[2]
        a43 = a2[3]

        a14 = a3[0]
        a24 = a3[1]
        a34 = a3[2]
        a44 = a3[3]

        a_00 = np.sqrt(a11**2 + a12**2)
        a_01 = a13
        a_02 = a14

        a_10 = np.sqrt(a11**2 + a12**2) * a22* a31 - np.sqrt(a11**2 + a12**2) * a21* a32
        a_11 = a11*a22*a33 - a12*a21*a33
        a_12 = a11*a22*a34 - a11*a24*a32 - a12*a21*a34 + a12*a24*a31

        a_20 = np.sqrt(a11**2 + a12**2) * a22* a41 - np.sqrt(a11**2 + a12**2) * a21* a42
        a_21 = 0.0
        a_22 = a11*a22*a44 - a11*a24*a42 - a12*a21*a44 + a12*a24*a41

        a_30 = np.sqrt(a11**2 + a12**2) * a32* a41 - np.sqrt(a11**2 + a12**2) * a31* a42
        a_31 = -a11*a33*a42 + a12*a33*a41
        a_32 = a11*a32*a44 - a11*a34*a42 - a12*a31*a44 + a12*a34*a41

        A_ = np.array([a_00, a_01, a_02,
                    a_10, a_11, a_12,
                    a_20, a_21, a_22,
                    a_30, a_31, a_32]).reshape(4,3)
        
        b1 = B[0][0]
        b2 = B[1][0]
        b3 = B[2][0]
        b4 = B[3][0]

        b_0 = b1
        b_1 = a11*a22*b3 - a11*a32*b2 - a12*a21*b3 + a12*a31*b2
        b_2 = a11*a22*b4 - a11*a42*b2 - a12*a21*b4 + a12*a41*b2
        b_3 = a11*a32*b4 - a11*a42*b3 - a12*a31*b4 + a12*a41*b3

        B_ = np.array([b_0, b_1, b_2, b_3]).reshape(4,1)
        # -------------------------- End ---------------------------

        # Assumption of determinant is not equals 0
        A_A_t = A_@np.transpose(A_)
        det = np.linalg.det(A_A_t)
        if(det != 0.0):

            # A'w' = B' -> w' = B'A'_   (A'_ - A' pseudo inverse)
            pinv = np.linalg.pinv(A_)
            w_ = pinv@B_
            w_ = w_.reshape(3,)

            # Translation vector
            t = np.array([w_[1], 0.0, w_[2]])

            # Pi and theta angle by given formula (13)(14)
            # -------------------- Start ----------------
            temp = 0.0
            if(a11 < 0.0):
                temp = np.pi
            pi_angle = math.atan2(a12, a11) + temp
            theta = np.arcsin(w_[0]) - pi_angle
            if(-1 <= w_[0] <= 1):
                theta = math.asin(w_[0]) - pi_angle
                
            else:
                theta_plus_pi = w_[0]
                while((-1 > theta_plus_pi) | (theta_plus_pi > 1)):
                    if(theta_plus_pi > 0):
                        theta_plus_pi -= 1.0
                    else:
                        theta_plus_pi -= -1.0
                
                theta = math.asin(theta_plus_pi) - pi_angle
            # -------------------- End ----------------

            # Projection matrix M by formual (2)
            # --------------------- Start ----------------
            proj_00 = fx*np.cos(theta) - cx*np.sin(theta)
            proj_01 = 0.0
            proj_02 = fx*np.sin(theta) + cx*np.cos(theta)
            proj_03 = fx*t[0] + cx*t[2]

            proj_10 = -cy*np.sin(theta)
            proj_11 = fy
            proj_12 = cy*np.cos(theta)
            proj_13 = cy*t[2]

            proj_20 = -np.sin(theta)
            proj_21 = 0.0
            proj_22 = np.cos(theta)
            proj_23 = t[2]

            proj = np.array([proj_00, proj_01, proj_02, proj_03,
                                proj_10, proj_11, proj_12, proj_13,
                                proj_20, proj_21, proj_22, proj_23]).reshape(3, 4)

            
            # --------------------- End ----------------
            return theta, t, proj
            
        else:
            return []

    def ransac_two_points_pnp_with_inliers(self, world_coords, points, src, confidence, outlier_ratio):
        '''
        2 points pose estimation(the function above named two_pts_pnp) with RANSAC to find optimal solution from set of correspondences.

        Input: 
            world_coords        -   3D points
            points              -   second pair of correspoding point pairs
            src                 -   first pair of corresponding point pairs
            confidence          -   float number for confidence
            outlier_ratio       -   outlier ratio compared with inliers

        Output:
            best_proj           -   Projection matrix
            best_inliers_world  -   3D points (filtered by mean distance)
            best_inliers_point  -   2D points (filtered by mean distance)
            inliers             -   filtered mask 
            threshold           -   distance threshold
        '''

        # For each iteration random 2 points selected
        number_of_random_points = 2

        # Parameters initialization
        threshold = np.inf
        error = 10.0
        # Maximum number of iteration
        number_of_Ransac_Iteration = np.log(1 - confidence) / np.log(1 - pow(1 - outlier_ratio, number_of_random_points))
        # print("number_of_Ransac_Iteration: ", number_of_Ransac_Iteration)

        
        for i in range(int(number_of_Ransac_Iteration)):
            # 2 random indices between 0 to given number_of_random_points
            random_index = random.sample(range(0, points.shape[0]), number_of_random_points)

            # Selecting random indices points
            random_pts = points[random_index]
            random_world = world_coords[random_index]

            # Estimating pose with two 3D to 2D points pose estimation
            result = self.two_pts_pnp(random_world, random_pts)
            if(len(result) != 0):
                theta = result[0]
                t = result[1]
                proj = result[2]
                if(np.isnan(proj).any()):
                    print("nan value found")
                    continue
            else:
                print("result not found")
                continue
            

            # The resulting projection matrix is combination of camera matrix, rotation matrix and translation vector
            # So, deducting camera matrix from the projection matrix by multiplying with its inverse
            Rt = np.linalg.inv(self.K)@proj
            R = Rt[:, :3]
            t = Rt[:, 3]

            # Essential matrix
            E = self.t_skew(t)@R

            # 3D points reprojection to left image plane
            world_coords_hom = np.hstack((world_coords, np.ones((world_coords.shape[0], 1))))
            world_projected = proj@np.transpose(world_coords_hom)
            world_projected = np.transpose(world_projected)
            world_projected[:, 0] = world_projected[:, 0]/world_projected[:, 2]
            world_projected[:, 1] = world_projected[:, 1]/world_projected[:, 2]
            world_projected[:, 2] = world_projected[:, 2]/world_projected[:, 2]

            world_to_pixel = world_projected[:, :2]

            # Sampson error approximation with first corresponding pair with reprojected points
            corrected_src, corrected_world_pixel, err = self.sampson_approximation(src, world_to_pixel, E)

            # Calculating distance between reprojected 3D point and first corresponding pair
            distance = []
            for p in range(src.shape[0]):
                p1 = (src[p][0], src[p][1])
                p2 = (corrected_world_pixel[p][0], corrected_world_pixel[p][1])
                d = np.sqrt(((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2))
                distance.append(d)

            distance = np.array(distance).reshape(len(distance), )

            # Creating mask for calculated distance and sampson approximation error
            mask = np.ones((distance.shape), dtype=bool)
            mask[(distance > np.mean(np.abs(distance))) & (err > 0.01)] = False
            
            # Finding suitable pose estimation
            if((threshold > np.mean(np.abs(distance))) & (np.mean(np.abs(err)) < 0.1)):
                threshold = np.mean(np.abs(distance))
                inliers = mask
                best_inliers_world = world_coords[mask,:]
                best_inliers_point = points[mask, :]
                best_theta = theta
                best_t = t
                best_proj = proj
        
        # print("2 point pnp ransac number of inliers: ", inliers.shape)
        # print("2 point pnp ransac threshold - mean distance: ", threshold)

        # Projection matrix deducted thw left camera matrix
        best_proj = np.linalg.inv(self.K)@best_proj

        return best_proj, best_inliers_world, best_inliers_point, inliers, threshold

    def draw_trajectory(self, x, z, g_x, g_z, il_next):
        '''
        The function to draw trajectory on 1000 by 1000 black screen.

        Input:
            x           -   estimated pose x coordinate
            z           -   estimated pose z coordinate
            g_x         -   ground truth x coordinate
            g_z         -   ground truth z coordinate
            il_next     -   left image

        Output:
            Drawn trajectory
            Image
        '''

        # If KITTI Benchmark utilized
        if(self.dataset_type == "kitti"):
            # Placing coordinates in a position
            center = (int(x) * 1 + 400, int(z)*-1 + 300)
            t_center = (g_x + 400, g_z*-1 + 300)
                
            # Draw estimated and true values on trajectory plane
            cv.circle(self.traj, center ,1, (255,0,0), 2)
            cv.circle(self.traj, t_center, 1, (0,255,0), 2)

        # In case of ELTECar dataset
        else:
            # Placing coordinates in a position
            center = (int(x) * 1 + 600, int(z)*-1 + 900)
            t_center = (g_x*-1 + 600, g_z + 900)
                
            # Draw estimated and true values on trajectory plane
            cv.circle(self.traj, center ,1, (255,0,0), 2)
            cv.circle(self.traj, t_center, 1, (0,255,0), 2)

        # Displaying Image and trajectory
        cv.imshow("Camera view", il_next)
        cv.imshow("Trajectory", self.traj)
            
        cv.waitKey(1)

    def display_3d_points(self, filename):
        '''
        The function illustrate the trajectory in 3 dimentional environment using Open3D.

        Input:
            filename          -   file name to write 3D representation of result

        Output:
            3D representation stored in the file
            Visualization of the trajectory
        '''

        # Estimated Position into numpy array
        data = np.array(self.camera_position)

        # Reshape position data and corresponding color (1, 0, 0) to all the position points
        data = data.reshape(data.shape[0], 3)
        data_color = np.zeros((data.shape[0], 3))
        data_color[:, 0] = np.ones((data.shape[0]))
        data_color[:, 1] = np.zeros((data.shape[0]))
        data_color[:, 2] = np.zeros((data.shape[0]))


        # Set up ground value and its corresponding (0, 1, 0) color value
        true_val = np.array(self.ground_truth_3d)
        true_val = true_val.reshape(true_val.shape[0], 3)
        true_val_color = np.zeros((true_val.shape[0], 3))
        true_val_color[:, 0] = np.zeros((true_val.shape[0]))
        true_val_color[:, 1] = np.ones((true_val.shape[0]))
        true_val_color[:, 2] = np.zeros((true_val.shape[0]))

        # Combining ground truth and position point
        data = np.vstack((data, true_val))
        # Combining both color data
        color = np.vstack((data_color, true_val_color))

        # Creating PointCloud Object
        pt = o3d.geometry.PointCloud()
        # Assigning points and colors
        pt.points = o3d.utility.Vector3dVector(data)
        pt.colors = o3d.utility.Vector3dVector(color)

        # Write points to the file with given name
        o3d.io.write_point_cloud(filename, pt)

        # Displaying points
        o3d.visualization.draw_geometries([pt])

    def next_frame(self, il_next, ir_next, fm_method):
        '''
        Helper function to 
            1. assign next frame feature points and 3D points 
            2. transform 3D points by inverse transformation matrix 

        Input: 
            il_next     -   left image frame 
            ir_next     -   right image frame
            fm_method   -   feature matching method "KLT" (corner detection) or "SIFT" (feature matching)

        Output:
            il_prev     -   given left image frame (now it become previous image frame)
            pts_l       -   first pair of corresponding pairs or detected corners in KLT case
            world       -   3D points 
        '''

        # feature matching either corner or feature detection depends on the variable passed in fm_method
        pts_l, pts_r, world, err_l, err_r = self.feature_matching_final(il_next, ir_next, fm_method)

        # Take homogenous coordinates of 3D points and apply inverse transformation
        world_coords_hom = np.hstack((world, np.ones((world.shape[0], 1))))

        # Apply inverse transformation
        world_coords_transformed = self.inv_transform@np.transpose(world_coords_hom)
        world_coords_transformed = np.transpose(world_coords_transformed)

        # Create filter for 3D points 
        mask = np.ones((world_coords_transformed.shape[0]), dtype=bool)
        mask[(world_coords_transformed[:, 2] <= 0) | (world_coords_transformed[:, 0] >= 1000) | (world_coords_transformed[:, 2] >= 1000) | (world_coords_transformed[:, 1] >= 1000)] = False

        # Filter application
        world = world_coords_transformed[mask, :]
        pts_l = pts_l[mask, :]

        # Make current left image frame as previous left image frame
        il_prev = il_next

        return il_prev, pts_l, world


    def main(self, fm_method, outlier_removal_method, pose_estimation_method, filename):
        '''
        Visual Odomtery main function
        
        Input:
            fm_method                   -   Feature matching method "KLT" or "SIFT"
            outlier_removal_method      -   Method name to remove outlier. Here we have 3 options:
                                                                                                    1. "1pts"  - One Point Ransac           - function related - histogram_voting_1pts()
                                                                                                    2. "2pnp"  - 2 points pose estimation   - function related - ransac_two_points_pnp_with_inliers()
                                                                                                    3. ""      - No outlier removal applied
            pose_estimation_method      -   Method name for pose estimation. 2 options: 
                                                                                        1.  "5pts"        -   5 points pose estimation      - function related - OpenCV findEssentialMat()
                                                                                        2.  "P3PRansac"   -   Perspective-3-Points solution - function related - p3p_transformation()
        
            filename                    -   file to store trajectory
        Output:
            Store and visualize trajectory 
        
        '''

        # Reading and storing images
        self.read_imgs()
        
        # Saving image height and width
        self.height = self.img_left[0].shape[0]
        self.width = self.img_left[0].shape[1]

        # Ground truth pose 
        self.get_pose()

        # Ground truth starting from given starting sequence
        if(self.dataset_type != "kitti"):
            self.gt_poses = self.gt_poses[self.starting_sequence:, :]


        # Initialization of inverse transformation
        self.inv_transform[:, 3]  = np.array([self.gt_poses[0][0], 0.0, self.gt_poses[0][1]])

        # First image pairs
        il = self.img_left[0]
        ir = self.img_right[0]

        # Feature matching or corner points extraction depending on given method (fm_method)
        pts_l, pts_r, world, err_l, err_r = self.feature_matching_final(il, ir, fm_method)

        # Setting previous image frame
        il_prev = il

        # Iterating over all the images 
        for i in range(1, len(self.img_left)):
            print("img: ", i)
            # Reading next image frames
            il_next = self.img_left[i]
            ir_next = self.img_right[i]


            # condition - in case there is not enough points to continue
            if((pts_l.shape[0] < 5) ):
                il_prev, pts_l, world = self.next_frame(il_next, ir_next, fm_method)
                continue
            
            # Corresponding tracked points on consecutive image pairs
            dst_pts, world, src_pts = self.track(il_prev, il_next, pts_l, world)

            # condition - in case there is not enough points to continue
            if((world.shape[0] < 5) ):
                il_prev, pts_l, world = self.next_frame(il_next, ir_next, fm_method)
                continue

            # Outlier removal 
            # -------------------- Start -----------------------
            if(outlier_removal_method == "1pts"):
                inliers_src, inliers_point, mask = self.histogram_voting_1pts(src_pts, dst_pts)
                inliers_world = world[mask, :]
                inlier_ratio = inliers_point.shape[0]/dst_pts.shape[0]
                # print("Inliers 1pts: ", inliers_point.shape)
                # print("inliers ratio 1pts: ", inlier_ratio)
            elif(outlier_removal_method == "2pnp"):
                _, inliers_world, inliers_point, mask, threshold = self.ransac_two_points_pnp_with_inliers(world, dst_pts, src_pts, 0.99, 0.7)
                inliers_src = src_pts[mask, :]
                inlier_ratio = inliers_point.shape[0]/dst_pts.shape[0]
                # print("Inliers 2pnp: ", inliers_point.shape)
                # print("inliers ratio 2pnp: ", inlier_ratio)
            elif(outlier_removal_method == ""):
                inliers_world = world
                inliers_point = dst_pts
                inliers_src = src_pts
                inlier_ratio = 1.0
            # -------------------- End -----------------------
        

            # condition - in case there is not enough points to continue
            if((inliers_world.shape[0] < 5) ):
                il_prev, pts_l, world = self.next_frame(il_next, ir_next, fm_method)
                continue

            # Pose estimation
            # ---------------------- Start -----------------------
            if(pose_estimation_method == "P3PRansac"):  
                # Pose estimation on given set of 3D and 2D points
                rmat, tvec, threshold, _ = self.p3p_transformation(inliers_world, inliers_point)

                # Inverse Transformation 
                rmat1 = np.transpose(rmat)
                tvec1 = -np.transpose(rmat)@tvec
                tvec1 = tvec1.ravel()

                # Projection matrix
                proj = np.hstack((rmat1, tvec1.reshape(3,1)))

                # Transformed projection matrix
                cam_location = proj@self.cam_pos

                x = cam_location[0]
                y = cam_location[1]
                z = cam_location[2]

                # Making sure that the points are not too far
                if((np.abs(x)<1000) & (np.abs(y)<1000) & (z>0) & (z<1000)):  
                    self.inv_transform[:, :3] = np.transpose(rmat)
                    self.inv_transform[:, 3] = -np.transpose(rmat)@tvec.reshape(3,)

            elif(pose_estimation_method == "5pts"):
                # When points are equals 5 the more than one solution finds
                if(inliers_point.shape[0] == 5):
                    # Essential matrix estimation on 5 points
                    E, inlier_mask = cv.findEssentialMat(inliers_src, inliers_point, self.K)
                    num = 0
                    # More than 1 solution has found
                    if(E.shape[0] > 3):
                        # Iterating over all the solution 
                        for j in np.arange(0, E.shape[0], 3):
                            E_ = E[j:j+3, :]
                            # Recovering rotation matrix and translation vector
                            inlier_num, rmat, tvec, inlier_mask = cv.recoverPose(E_, inliers_src, inliers_point, self.K, mask=inlier_mask)
                            # Storing the maximum inliers
                            if(num < inlier_num):
                                num = inlier_num
                                best_E = E_
                        # Essential matrix extraction
                        inlier_num, rmat, tvec, inlier_mask = cv.recoverPose(best_E, inliers_src, inliers_point, self.K, mask=inlier_mask)
                    
                    # condition - in case there is not enough points to continue
                    else:
                        il_prev, pts_l, world = self.next_frame(il_next, ir_next, fm_method)
                        continue
                # Otherwise one solution produces 
                else:
                    # Find Essential matrix
                    E, inlier_mask = cv.findEssentialMat(inliers_src, inliers_point, self.K)
                    
                    # If the solution has found, then extract Essential matrix into rotation matrix and translation vector
                    if(E.shape[0] > 0):
                        inlier_num, rmat, tvec, inlier_mask = cv.recoverPose(E, inliers_src, inliers_point, self.K, mask=inlier_mask)
                    
                    # condition - in case there is not enough points to continue
                    else:
                        il_prev, pts_l, world = self.next_frame(il_next, ir_next, fm_method)
                        continue
                    
                # If there is enough inliers, then apply transformation and update inverse transformation matrix
                if (inlier_ratio > 0.2):
                    T = np.eye(4)
                    T[:3, :3] = np.transpose(rmat)
                    T[:3, 3] = (-np.transpose(rmat)@tvec.ravel())

                    self.inv_transform = self.inv_transform @ T

                # Translation vector
                tvec1 = self.inv_transform[:3, 3]
                x = tvec1[0]
                y = tvec1[1]
                z = tvec1[2]
            # ---------------------- End -----------------------

            # Printing x, y, z coordinates
            print("x, y, z: ", x, y, z)

            # For ELTECar dataset z position coordinate is negative 
            if(self.dataset_type == "elte"):
                self.camera_position.append([x,y,-z])
                self.ground_truth_3d.append([-self.gt_poses[i][0], 0.0, self.gt_poses[i][1]])
            else:
                self.camera_position.append([x,y,z])
                self.ground_truth_3d.append([self.gt_poses[i][0], self.gt_poses[i][1], self.gt_poses[i][2]])

            # Until here the pose estimation calculation has finished

            # Setting up next image frames and preperation for next iteration
            # ---------------------------- Start -------------------------
            
            # Feature matching or corner detection
            pts_l, pts_r, world, err_l, err_r = self.feature_matching_final(il_next,ir_next, fm_method)
            # print("after feature match", pts_l.shape[0])

            # Transforming 3D homogenous points by inverse transformation matrix
            world_coords_hom = np.hstack((world, np.ones((world.shape[0], 1))))
            world_coords_transformed = self.inv_transform@np.transpose(world_coords_hom)
            world_coords_transformed = np.transpose(world_coords_transformed)

            # Creating filter for 3D points
            mask = np.ones((world_coords_transformed.shape[0]), dtype=bool)
            mask[(world_coords_transformed[:, 2] <= 0) | (world_coords_transformed[:, 0] >= 1000) | (world_coords_transformed[:, 2] >= 1000) | (world_coords_transformed[:, 1] >= 1000)] = False

            # Applying filter for 3D and 2D points
            world = world_coords_transformed[mask, :]
            pts_l = pts_l[mask, :]
 
            # Set previous left image frame as current left image frame
            il_prev = il_next
            # ---------------------------- End -------------------------


            # Draw trajectory in each iteration
            if(self.dataset_type == "kitti"):
                self.draw_trajectory(x, z, int(self.gt_poses[i][0]), int(self.gt_poses[i][2]), il_next)
            else:
                self.draw_trajectory(x, z, int(self.gt_poses[i][0]), int(self.gt_poses[i][1]), il_next)
        # Displying full trajectory
        self.display_3d_points(filename) 

