<h1 align="center"> <b>3D reconstruction from a vehicle mounted camera</b>  </h1>

<h3> Brief introduction </h3>
<p>This thesis work worked on a solution of visual odometry using consecutive stereo images and proposed two different methods on rectified and original frame pairs, which follow separate computation in vehicle pose estimation. First, find feature correspondences one pair based on three dimensional points produced by disparity map, the rectified stereo pair, and the other pair tracked over consequent frame using optical flow. The process yields three pairs of feature matches, which are corresponding image points and spatial coordinates. Hence, outlier removal step considered with a pose estimation under the planar motion algorithm adopted with RANSAC. Afterwards, vehicle pose estimation em- ployed a solution to Perspective-3-Point problem. Another solution for unrectified stereo pairs starts with SIFT feature matching technique, followed by correspond- ing image point triangulation for three dimensional points and optical flow tracks to find consecutive frame matches. Therefore, three-point feature matching has be- come available to track image sequences. Outlier removal, in this case, utilized idea of monocular visual odometry. Hence, one point under the circular planar motion algo- rithm was implemented to get fewer contaminated feature points. Vehicle ego-motion is produced by a five-point pose estimation algorithm. </p>

<img src="https://github.com/bbaigalmaa/3D-reconstruction-from-a-vehicle-mounted-camera/assets/25894954/35ed2c1a-f26d-4316-9b15-c5d207ab3fba" width="500" height="450">
<img src="https://github.com/bbaigalmaa/3D-reconstruction-from-a-vehicle-mounted-camera/assets/25894954/5969ba0e-2847-4a74-a82d-a0fe64e089b1" width="500" height="500"> 


This thesis work has utilized **KITTI** Benchmark and **ELTECar** datasets. Those can be found from https://www.cvlibs.net/datasets/kitti/eval_odometry.php and https://www.hackademix.hu/timeline-of-the-contest/

**KITTI** Benchmark has plenty of samples and implementaions. Those can be find in the tables of above mentioned site for KITTI. 
On the other hand, **ELTECar** dataset is set of images, GPS coordinates and Lidar points captured by vehicle mounted cameras used in Eötvös Loránd University, Faculty of Informatics.

<img src="https://github.com/bbaigalmaa/3D-reconstruction-from-a-vehicle-mounted-camera/assets/25894954/3b57ae26-e7af-44a9-a970-8cb2ea548753" width="500" height="500">

The above picture shows the ELTECar Camera setup, where this thesis work utilized DEV1 and DEV0 cameras. Each camera's measurements are relative to the Lidar optical center, Y axis. The cameras are monocular HikVision/HikRobot MV-CA020-20GC model with normal optics. The cameras are placed around the front wheel of the car, facing relative to the forward axis at 20 and 60 degree angles. Other than the cameras, ELTECar installed GPS, LIDAR, IMU and those sensors' current detailed information can be found in https://www.hackademix.hu/wp-content/uploads/2023/06/Sensor_pack_summary_2023.pdf. Generally, all the ELTECar datasets have the same format and two different routes(for now). However, in the case of simplicity, this elaboration is going to choose a specific dataset and focus on explaining the procedure on the dataset. The specific dataset is the second route captured on 2023.10.06. This ELTECar Second Route 4th testing dataset is shared in this oneDrive folder, which requires login with a university account - https://ikelte-my.sharepoint.com/personal/kovbando_inf_elte_hu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fkovbando%5Finf%5Felte%5Fhu%2FDocuments%2Fosszerendelt%5Fnyers%5Fadatok%2F20231006&ga=1. The total size of the dataset is 22.1GB and here, in this elaboration, the dataset is referenced as 20231006 due to it being extracted into the same named folder. Folder structure and ground truth generation steps are the following.

The dataset contains following (see also 20231006/info.txt): 
<ul>
  <li>All image files - 20231006_ELTEkorV2_pictures.rar</li>
  <li>Raw lidar data - 20231006_ELTEkorV2_polars.rar</li>
  <li>Lidar scans in ‘xyz’ text format - 20231006_ELTEkorV2_cartesians.rar</li>
  <li>Route video - 20231006_ELTEkorV2.mp4</li>
  <li>Trajectory based on GPS sensor - 20231006_ELTEkorV2.kml</li>
  <li>GPS coordinates - 20231006_ELTEkorV2.csv</li>
</ul> 

Now, we need to prepare ground truth trajectory points from GPS. To do so, the steps are following in ELTECar Second Route 4th testing dataset:
<ol>
  <li>Extract ground truth source code - https://ikelte-my.sharepoint.com/personal/hajder_inf_elte_hu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fhajder%5Finf%5Felte%5Fhu%2FDocuments%2FBosch%2FBoschVerseny%20V3%2Fvisualize%5Fgroundtruth%5Fgps%5Fvectors%2Ezip&parent=%2Fpersonal%2Fhajder%5Finf%5Felte%5Fhu%2FDocuments%2FBosch%2FBoschVerseny%20V3&ga=1</li>
  <li>As mentioned in the readme.txt, Create new folder named “images” and copy DEV0 images (PATH/20231006/20231006_ELTEkorV2_pictures/DEV0/Dev0_Image_w1920_h1200_fn*.jpg - in case of 4th second route testing data)</li>
  <li>Open command prompt in the folder and run “python ProcessGPSData.py PATH/20231006/20231006_ELTEkorV2.csv” including the argument directed to .csv file of the dataset </li>
  <li>Above command (in previous step) generates 2 files and displays a figure:
		<ul>
            	    <li>CorrectedGPSData.csv - GPS filtered data</li>
		    <li>pts2D.mat - vehicle 2D virtual trajectory</li>
		    <li>Figure displays 2D trajectory - can be saved for future reference of the figure </li> </ul>
  <li>An addition, velocity ground truth is able to be generated by running “python DrawSpeed.py PATH/20231006/20231006_ELTEkorV2.csv video_with_speed.avi”. </li>
</ol> 
The above steps are the same for other ELTECar datasets by replacing DEV0 images and changing the .csv file path.

Then, to make simple to read images, the ELTECar dataset was renamed to a numerical ordering same as KITTI images. However, it is not necessary to be renamed. Reasons why the thesis work has implemented this renaming step are: 1. Reading image files one by one took a long time.
                        2. When reading all the files from the folder, the image ordering changed.
So the solution I found was to rename the images once and use the renamed image folder instead of the original one.
To do so, we need to execute ELTECarDatasetRename/rename_ELTECar.py with following parameters:
<ul>
<li> start - starting frame number (1, if it needs to be starting from first frame)</li>
    <li>last - last image frame number to be renamed </li>
    <li>path_read - image path which "Dev*_Image_w1920_h1200_fn*.jpg" exists</li>
    <li>path_write - path to write renamed images</li>
    <li>camera - specifiying which camera images are renaming with "DEV0" or "DEV1" or "DEV2" or "DEV3"</li>
</ul>
For example, in the 20231006 dataset we can run the following from opened command prompt in the folder, ELTECarDatasetRename. This inculdes general template and sample. <br>
TEMPLATE  -  "python rename_ELTECar.py start last path_read path_write camera" <br>
SAMPLE    -  "python rename_ELTECar.py 1 6145 PATH/FourthTestingDataForSecondRoute/20231006/20231006_ELTEkorV2_pictures/DEV0/ PATH/DEV0_renamed DEV0" <br>
