import csv
import math
import numpy as np


#Convert gps coordinatex to spatial locations
#Earth radius is set to 6371000 meter
#
#Input
#lat,lon,alt: GPS coordinates (in radian)
#
#Output: coordinates of vector [x,y,z]

def ConvertGpsToXyz(lat,lon,alt):
#    r = 6371000 + alt
    r = 6371000
    x = r*math.cos(lat)*math.cos(lon)
    y = r*math.cos(lat)*math.sin(lon)
    z = r*math.sin(lat)

    vec1=np.zeros(3)
    vec1[0] = -1.0*r*math.sin(lat)*math.cos(lon)
    vec1[1] = -1.0*r*math.sin(lat)*math.sin(lon)
    vec1[2] = r*math.cos(lat)

    length1=np.linalg.norm(vec1)
    vec1=vec1/length1

    vec2=np.zeros(3)
    vec2[0] = r*math.cos(lat)*math.sin(lon)
    vec2[1] = -1.0*r*math.cos(lat)*math.cos(lon)
    vec2[2] = 0
    length2=np.linalg.norm(vec2)
    vec2=vec2/length2

    return x,y,z,vec1,vec2



#Read the GPS data from ELTE cvs format
#Colors are based on the speed. black: zero white: highest speed within the interval

def ReadGpsFromCSV(filename):

    #NUMP will store the number of points
    NUMP=0

    minAlt=float('inf')
    maxAlt=float('-inf')

    minX=float('inf')
    maxX=float('-inf')

    minY=float('inf')
    maxY=float('-inf')

    minZ=float('inf')
    maxZ=float('-inf')

    minVel=float('inf')
    maxVel=float('-inf')

    avgDir1=np.zeros(3)
    avgDir2=np.zeros(3)

    #Load GPS data, convert it to spatial points, store minimal and maximal coordinates for x,y,z and altitude
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile,delimiter=';')
        for row in reader:
                NUMP+=1
                lat=float(row['Lat'])
                lon=float(row['Lon'])
                alt=float(row['Alt'])



                #Convert to m/s
                vel=float(row['Vel'])/3.6

                x,y,z,dir1,dir2=ConvertGpsToXyz(lat,lon,alt)
                avgDir1=avgDir1+dir1
                avgDir2=avgDir2+dir2



                if (alt>maxAlt):
                    maxAlt=alt
                if (alt<minAlt):
                    minAlt=alt

                if (x>maxX):
                    maxX=x
                if (x<minX):
                    minX=x

                if (y>maxY):
                    maxZ=y
                if (y<minY):
                    minY=y

                if (z>maxZ):
                    maxZ=z
                if (z<minZ):
                    minZ=z

                if (vel>maxVel):
                    maxVel=vel
                if (vel<minVel):
                    minVel=vel


    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile,delimiter=';')


        pts=np.zeros((NUMP,3))
        colors=np.zeros((NUMP,3))

        dirs=np.zeros((NUMP,3));

        alts=np.zeros(NUMP)

        velocities=np.zeros(NUMP)

        IDs=np.zeros(NUMP)


        #Original lt/lon
        GPSOrig=np.zeros((NUMP,2))


        #Point counter
        cnt=0;

        minAlt=float('inf')
        maxAlt=float('-inf')
        minVel=float('inf')
        maxVel=float('-inf')

        for row in reader:
            id=int(row['ID'])
            lat=float(row['Lat'])
            lon=float(row['Lon'])
#Convert km/h to m/s
            vel=float(row['Vel'])/3.6
            alt=float(row['Alt'])
            id=int(row['ID'])


            IDs[cnt]=id

            GPSOrig[cnt,0]=lat
            GPSOrig[cnt,1]=lon

            #Convert degree to radian
            lat=math.pi*lat/180.0;
            lon=math.pi*lon/180.0;

            #Convert GPS to 3D location
            x,y,z,d1,d2=ConvertGpsToXyz(lat,lon,alt)


            pts[cnt,0]=x
            pts[cnt,1]=y
            pts[cnt,2]=z

            velocities[cnt]=vel;

            alts[cnt]=alt;

            if (alt<minAlt):
                minAlt=alt

            if (alt>maxAlt):
                maxAlt=alt

            if (vel>maxVel):
                maxVel=vel

            if (vel<minVel):
                minVel=vel


            cnt=cnt+1

#Sometimes center points are useful.
#        centerAlt=(maxAlt-minAlt)/2.0
#        centerX=(maxX-minX)/2.0
#        centerY=(maxY-minY)/2.0
#        centerZ=(maxZ-minZ)/2.0


    for idx in range(0,NUMP):
    #colors(idx,1:3)=round(255*(alt-minAlt)/(maxAlt-minAlt)*[1,1,1]);

#        currColor=int(255.0*(alts[idx]-minAlt)/(maxAlt-minAlt))
        currColor=int(255.0*(velocities[idx]-minVel)/(maxVel-minVel))

        colors[idx,0]=currColor
        colors[idx,1]=currColor
        colors[idx,2]=currColor


#Calculate average directionns
    length1=np.linalg.norm(avgDir1)
    length2=np.linalg.norm(avgDir2)

    avgDir1=avgDir1/length1;
    avgDir2=avgDir2/length2;


    return pts,colors,velocities,alts,GPSOrig,avgDir1,avgDir2,IDs

