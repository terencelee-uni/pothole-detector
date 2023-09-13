import numpy as np
import os, json, cv2, random
from classify import *
from inputData import *
from safety import *
from submit import *
import threading as th
import logging


INPUTTYPE = "WEBCAM"    # Webcam or Local
INPUTLOCATION = "/"

def get_immediate_subdirectories(a_dir): #only grab dirs no recursion
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def imageProcess():
    subDir = get_immediate_subdirectories("data")

    potholeDetected = []
    # for x in subDir: #iterate each image cluster
    #     cluster = x
    #     clusterImages = []        
    #     out = classify(x) # out contains all images in cluster that have potholes
    #     if len(out) > 0:
    #         for z in out:
    #             temp = z.split("mask")[1]  #get orig image name
    #             clusterImages.append([temp,[temp[1],temp[2],temp[3],temp[4]]])    # type format [raw image path, [datetime, direction, gps, frame]]

    #     potholeDetected.append(clusterImages)
    potholeDetected.append("data/2023-09-13|03/mask|03:14|2023-09-13|North|37.422131|122.084801|frame1.jpg")
    return potholeDetected


keep_going = True
def key_capture_thread(): #exit inf loop on keypress
    global keep_going
    input()
    keep_going = False

def main():
    # if(INPUTTYPE.lower() == "webcam"):
    #     videoThread = Thread(target=getInput, arg=(1, ""))
    #     getInput()
    # else:
    #     vidPath = input("Input relative path to pre-recorded video:\n")
    #     videoThread = Thread(target=getInput, arg=(1, vidPath))
        
    # videoThread.start()
     # Start grabbing frames from camera and GPS store to /data/

    # th.Thread(target=key_capture_thread, args=(), name='key_capture_thread', daemon=True).start()
    # while keep_going:
    #     event.wait(60)  # Check in every minute
    #     logging.debug("Capture still running")

    images = imageProcess()  # Recording finished, parse captured frames and run detection
    print(images)
    for x in images:
        lat = x.split("|")[5]
        lon = x.split("|")[6]
        direction = x[4]
        dateTime = "%s|%s" % (x.split("|")[2], x.split("|")[3])
        print("Pothole detected at Date/Time - %s, Lat: %s Long: %s \n See below for safety rating\n" % (dateTime,lat,lon))
        im = cv2.imread(x)
        cv2.imshow("pothole", im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        rating = safety(x)

        if rating >= 0.7:
            print("Pothole has failed safety check, submitting relevant information to Department of Transit")
            #submit(lat, lon, direction, dateTime, x)   #Do not enable for demo mode

main() #run main loop
