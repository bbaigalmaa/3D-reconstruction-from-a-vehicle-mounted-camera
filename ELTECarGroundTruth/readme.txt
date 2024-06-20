This zip contains two applications (in ProcessGPSData.py and DrawSpeed.py)  and a file for CVS reading in read_gpsfromcsv.py.

The tool is designed to visualize the ground truth speed vectors from the GPS.

Usage:

(1) Download the rar file of the pictures and the CSV file from our website.
(2) Create a folder called 'images/'. Extract Dev0* files from the rar into the folder images/
(3) Run the ProcessGPSData script. It is the only one argument is the CSV file. Note that the script saves several files that are required for the next steps. For example, the filtered data are saved to 'CorrectedGPSData.csv'. 2D GPS trajectory is saved as 'pts2D.mat". They are not GPS trajectories, they are points in a virtual map.
(4) Run  DrawSpeed.py. It has two arguments: the first argument is the CSV file, the second one is the name of the output AVI file. Speed vectors are saved as 'speedvectors.mat'.
(5) Enjoy the results :)



