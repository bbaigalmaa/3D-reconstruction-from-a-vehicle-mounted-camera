from read_gpsfromcsv import ReadGpsFromCSV
import numpy as np
import glob
import math
import matplotlib.pyplot as plt
import sys



def WriteBandoCSV(fileName,IDs,GPSAngles,alts,vels):
    f = open(fileName, "w")
    f.write("ID;Lat;Lon;Alt;Vel;Ax;Ay;Az;Mx;My;Mz,\n")

    for idx in range(0,GPSAngles.shape[0]):
        f.write(str(int(IDs[idx]))+";"+str(GPSAngles[idx,0])+";"+str(GPSAngles[idx,1])+";"+str(alts[idx])+";"+str(vels[idx])+";Ax;Ay;Az;Mx;My;Mz\n")
    f.close()



pts,colors,velocities,alts,GPSOrig,avgDir1,avgDir2,IDs=ReadGpsFromCSV(sys.argv[1])




#Filter points:
NUM=pts.shape[0]



pts2D=np.zeros((NUM,2))

for idx in range(0,NUM):
#        GPSOrig[idx+1,0]=(GPSOrig[idx,0]+GPSOrig[idx+2,0])/2.0
#        GPSOrig[idx+1,1]=(GPSOrig[idx,1]+GPSOrig[idx+2,1])/2.0
     pt=pts[idx,:]

     x=np.dot(avgDir1,pt)
     y=np.dot(avgDir2,pt)
     pts2D[idx,0]=x
     pts2D[idx,1]=y



#0: no error
#1: Ii is the same as the next one
#2: Its ditance from the next one is 1.5 times larger than from the previous one

errorFlags=np.zeros(NUM)

for idx in range(1,NUM-1):
     ptCurr=pts2D[idx,:]
     ptPrev=pts2D[idx-1,:]
     ptNext=pts2D[idx+1,:]

     distNext=np.linalg.norm(ptNext-ptCurr)
     distPrev=np.linalg.norm(ptPrev-ptCurr)

     if (distNext<1e-2):  #1 cm
         errorFlags[idx]=1.0
     elif (distNext>1.75*distPrev):
         errorFlags[idx]=2.0



#Correction:

for idx in range(1,NUM-2):
    if ((errorFlags[idx]==1.0)and(errorFlags[idx+1]==2.0)):
        errorFlags[idx]=0.0
        errorFlags[idx+1]=0.0
        pts2D[idx+1,0]=(pts2D[idx+1,0]+pts2D[idx+2,0])/2.0
        pts2D[idx+1,1]=(pts2D[idx+1,1]+pts2D[idx+2,1])/2.0
        GPSOrig[idx+1,0]=(GPSOrig[idx+1,0]+GPSOrig[idx+2,0])/2.0
        GPSOrig[idx+1,1]=(GPSOrig[idx+1,1]+GPSOrig[idx+2,1])/2.0
        alts[idx+1]=(alts[idx+1]+alts[idx+2])/2.0
        velocities[idx]=(velocities[idx+1]+velocities[idx+2])/2.0



#Filtering

for idx in range(2,NUM-2):
    pts2D[idx,0]=(pts2D[idx-2,0]+pts2D[idx-1,0]+pts2D[idx,0]+pts2D[idx+1,0]+pts2D[idx+2,0])/5.0
    pts2D[idx,1]=(pts2D[idx-2,1]+pts2D[idx-1,1]+pts2D[idx,1]+pts2D[idx+1,1]+pts2D[idx+2,1])/5.0
    GPSOrig[idx,0]=(GPSOrig[idx-2,0]+GPSOrig[idx-1,0]+GPSOrig[idx,0]+GPSOrig[idx+1,0]+GPSOrig[idx+2,0])/5.0
    GPSOrig[idx,1]=(GPSOrig[idx-2,1]+GPSOrig[idx-1,1]+GPSOrig[idx,1]+GPSOrig[idx+1,1]+GPSOrig[idx+2,1])/5.0
    alts[idx]=(alts[idx-2]+alts[idx-1]+alts[idx]+alts[idx+1]+alts[idx+2])/5.0
#    velocities[idx]=(velocities[idx+1]+velocities[idx+2])/2.0


"""

for idx in range(2,NUM):
    if ((errorFlags[idx]==1.0)and(errorFlags[idx-2]==2.0)):
        errorFlags[idx]=0.0
        errorFlags[idx-2]=0.0

        newX=(pts2D[idx-1,0]+pts2D[idx-2,0])/2.0
        newY=(pts2D[idx-1,1]+pts2D[idx-2,1])/2.0
        newGPSLat=(GPSOrig[idx-1,0]+GPSOrig[idx-2,0])/2.0
        newGPSLon=(GPSOrig[idx-1,1]+GPSOrig[idx-2,1])/2.0
        newAlt=(alts[idx-1]+alts[idx-2])/2.0
        newVel=(velocities[idx-1]+velocities[idx-2])/2.0


        oldX=pts2D[idx-1,0]
        oldY=pts2D[idx-1,1]
        oldGPSLat=GPSOrig[idx-1,0]
        oldGPSLon=GPSOrig[idx-1,1]
        oldAlt=alts[idx-1]
        oldVel=velocities[idx-1]

        pts2D[idx-1,0]=newX
        pts2D[idx-1,1]=newY
        GPSOrig[idx-1,0]=newGPSLat
        GPSOrig[idx-1,1]=newGPSLon
        alts[idx-1]=newAlt
        velocities[idx-1]=newVel

        pts2D[idx,0]=oldX
        pts2D[idx,1]=oldY
        GPSOrig[idx,0]=oldGPSLat
        GPSOrig[idx,1]=oldGPSLon
        alts[idx]=oldAlt
        velocities[idx]=oldVel


"""

"""

for idx in range(3,NUM):
    if ((errorFlags[idx]==1.0)and(errorFlags[idx-3]==2.0)):
        errorFlags[idx]=0.0
        errorFlags[idx-3]=0.0

        newX=(pts2D[idx-2,0]+pts2D[idx-3,0])/2.0
        newY=(pts2D[idx-2,1]+pts2D[idx-3,1])/2.0
        newGPSLat=(GPSOrig[idx-2,0]+GPSOrig[idx-3,0])/2.0
        newGPSLon=(GPSOrig[idx-2,1]+GPSOrig[idx-3,1])/2.0
        newAlt=(alts[idx-2]+alts[idx-3])/2.0
        newVel=(velocities[idx-2]+velocities[idx-3])/2.0

        oldX=pts2D[idx-2,0]
        oldY=pts2D[idx-2,1]
        oldGPSLat=GPSOrig[idx-2,0]
        oldGPSLon=GPSOrig[idx-2,1]
        oldAlt=alts[idx-2]
        oldVel=velocities[idx-2]

        old2X=pts2D[idx-1,0]
        old2Y=pts2D[idx-1,1]
        old2GPSLat=GPSOrig[idx-1,0]
        old2GPSLon=GPSOrig[idx-1,1]
        old2Alt=alts[idx-1]
        old2Vel=velocities[idx-1]

        pts2D[idx-2,0]=newX
        pts2D[idx-2,1]=newY
        GPSOrig[idx-2,0]=newGPSLat
        GPSOrig[idx-2,1]=newGPSLon
        alts[idx-2]=newAlt
        velocities[idx-2]=newVel

        pts2D[idx-1,0]=oldX
        pts2D[idx-1,1]=oldY
        GPSOrig[idx-1,0]=oldGPSLat
        GPSOrig[idx-1,1]=oldGPSLon
        alts[idx-1]=oldAlt
        velocities[idx-1]=oldVel

        pts2D[idx,0]=old2X
        pts2D[idx,1]=old2Y
        GPSOrig[idx,0]=old2GPSLat
        GPSOrig[idx,1]=old2GPSLon
        alts[idx]=old2Alt
        velocities[idx]=old2Vel
"""
"""
for idx in range(4,NUM):
    if ((errorFlags[idx]==1.0)and(errorFlags[idx-4]==2.0)):
        errorFlags[idx]=0.0
        errorFlags[idx-4]=0.0

        newX=(pts2D[idx-3,0]+pts2D[idx-4,0])/2.0
        newY=(pts2D[idx-3,1]+pts2D[idx-4,1])/2.0

        oldX=pts2D[idx-3,0]
        oldY=pts2D[idx-3,1]

        old2X=pts2D[idx-2,0]
        old2Y=pts2D[idx-2,1]

        old3X=pts2D[idx-1,0]
        old3Y=pts2D[idx-1,1]

        pts2D[idx-3,0]=newX
        pts2D[idx-3,1]=newY

        pts2D[idx-2,0]=oldX
        pts2D[idx-2,1]=oldY

        pts2D[idx-1,0]=old2X
        pts2D[idx-1,1]=old2Y

        pts2D[idx,0]=old3X
        pts2D[idx,1]=old3Y
"""



WriteBandoCSV("CorrectedGPSData.csv",IDs,GPSOrig,alts,velocities)

START=0
END=pts.shape[0]

np.savetxt("pts2D.mat",pts2D)


plt.plot(pts2D[range(START,END,6),0],pts2D[range(START,END,6),1],"rx",alpha=0.5)
plt.plot(pts2D[range(START+1,END,6),0],pts2D[range(START+1,END,6),1],"gx",alpha=0.5)
plt.plot(pts2D[range(START+2,END,6),0],pts2D[range(START+2,END,6),1],"bx",alpha=0.5)
plt.plot(pts2D[range(START+3,END,6),0],pts2D[range(START+3,END,6),1],"ro",alpha=0.5)
plt.plot(pts2D[range(START+4,END,6),0],pts2D[range(START+4,END,6),1],"go",alpha=0.5)
plt.plot(pts2D[range(START+5,END,6),0],pts2D[range(START+5,END,6),1],"bo",alpha=0.5)
#plt.plot(pts2D[:,0],pts2D[:,1],"rx")
plt.show()


