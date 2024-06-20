from read_gpsfromcsv import ReadGpsFromCSV
import cv2
import numpy as np
import glob
import math
import sys


pts,colors,velocities,alts,GPSOrig,avgDir1,avgDir2,IDs=ReadGpsFromCSV(sys.argv[1])


NUM_IDS=pts.shape[0]

START=int(IDs[0])
START=2360
END=int(IDs[NUM_IDS-2])


speedVectors=np.zeros((NUM_IDS,2))



pts2D=np.loadtxt("pts2D.mat")


frameSize = (1920, 1200)

out = cv2.VideoWriter(sys.argv[2],cv2.VideoWriter_fourcc(*'DIVX'), 10, frameSize)



# font
font = cv2.FONT_HERSHEY_SIMPLEX

# fontScale
fontScale = 1

# White color in BGR
color = (200, 200, 200)

# Line thickness of 2 px
thickness = 2



baseFileName='images/Dev0_Image_w1920_h1200_fn'
#baseFileName2='result/res'
#for filename in glob.glob():
for idx in range(0,NUM_IDS-1):
    fileName=baseFileName+str(int(IDs[idx]))+'.jpg'
#    fileName2=baseFileName2+str(int(IDs[idx]))+'.png'
    print("Processing: ",fileName)
    img = cv2.imread(fileName)

    if (img.shape[0]==1200):

        ptPrev=pts[idx-1,:]
        ptCurr=pts[idx,:]
        ptNext=pts[idx+1,:]


        gpsOrig=GPSOrig[idx,:]

    #    prevCoord1=np.dot(avgDir1,ptPrev)
    #    prevCoord2=np.dot(avgDir2,ptPrev)

    #    currCoord1=np.dot(avgDir1,ptCurr)
    #    currCoord2=np.dot(avgDir2,ptCurr)

    #    nextCoord1=np.dot(avgDir1,ptNext)
    #    nextCoord2=np.dot(avgDir2,ptNext)


        prevCoord1=pts2D[idx-1,0]
        prevCoord2=pts2D[idx-1,1]

        currCoord1=pts2D[idx,0]
        currCoord2=pts2D[idx,1]

        nextCoord1=pts2D[idx+1,0]
        nextCoord2=pts2D[idx+1,1]


        VecCurr=np.zeros(2)
        VecPrev=np.zeros(2)

        VecCurr[0]=nextCoord1-currCoord1
        VecCurr[1]=nextCoord2-currCoord2

        VecPrev[0]=currCoord1-prevCoord1
        VecPrev[1]=currCoord2-prevCoord2

        angleNext=math.atan2(VecCurr[1],VecCurr[0])
        angleCurr=math.atan2(VecPrev[1],VecPrev[0])




        velVec=4*np.linalg.norm(VecCurr) #4: due to 4FPS
    #    np.linalg.norm(VecPrev)


        diffAngle=angleNext-angleCurr
    #    diffAngle=angleNext

        speedVectors[idx,0]=3.6*velVec*math.cos(diffAngle+math.pi/2)
        speedVectors[idx,1]=3.6*velVec*math.sin(diffAngle+math.pi/2)

        vecX=20.0*velVec*math.cos(diffAngle+math.pi/2)
        vecY=20.0*velVec*math.sin(diffAngle+math.pi/2)






    # Using cv2.putText() method
        image = cv2.putText(img, 'GPS Lat/Lon: '+str(round(gpsOrig[0],6))+' '+str(round(gpsOrig[1],6)) , (1300, 950), font,  fontScale, color, thickness, cv2.LINE_AA)
        image = cv2.putText(img, 'Velocity Vector: ['+str(round(3.6*velVec*math.cos(diffAngle+math.pi/2),3)) +","+str(round(3.6*velVec*math.sin(diffAngle+math.pi/2),3))+"]" , (1300, 1000), font,  fontScale, color, thickness, cv2.LINE_AA)
    #    image = cv2.putText(img, 'DiffAngle '+str(180*diffAngle/3.1415), (1300, 1050), font,  fontScale, color, thickness, cv2.LINE_AA)
        image = cv2.putText(img, 'GPS Vel: '+str(round(3.6*velocities[idx-1],5))+' km/h', (1300, 1100), font,  fontScale, color, thickness, cv2.LINE_AA)
        image = cv2.putText(img, 'Velocity: '+str(round(3.6*velVec,2))+' km/h', (1400, 600), font,  fontScale, color, thickness, cv2.LINE_AA)
        image = cv2.putText(img, 'Frame # '+str(idx), (200, 1150), font,  fontScale, color, thickness, cv2.LINE_AA)

        endPointX=1600+int(vecX)
        endPointY=500-int(vecY)
        cv2.line(img,  (1600,500), (endPointX,endPointY), (255,255,255), 2)

        arrowEndX1=int(endPointX+(20.0*velVec/4)*math.cos(-1.0*diffAngle+math.pi/4))
        arrowEndY1=int(endPointY+(20.0*velVec/4)*math.sin(-1.0*diffAngle+math.pi/4))
        cv2.line(img,  (arrowEndX1,arrowEndY1), (endPointX,endPointY), (255,255,255), 2)

        arrowEndX2=int(endPointX+(20.0*velVec/4)*math.cos(-1.0*diffAngle+3*math.pi/4))
        arrowEndY2=int(endPointY+(20.0*velVec/4)*math.sin(-1.0*diffAngle+3*math.pi/4))
        cv2.line(img,  (arrowEndX2,arrowEndY2), (endPointX,endPointY), (255,255,255), 2)

        out.write(img)

#    cv2.imwrite(fileName2,img)

#'Vector # '+str(round(VecCurr[0],3)+","+str(round(VecCurr[0],3))

out.release()

np.savetxt("speedvectors.mat",speedVectors)
