import numpy as np
import os, json, cv2, random
import classify, submit, safety, inputData
import threading as th
import logging


def get_immediate_subdirectories(a_dir): #only grab dirs no recursion
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def imageProcess(flag):
    subDir = get_immediate_subdirectories("data")

    potholeDetected = []
    for x in subDir: #iterate each image cluster
        cluster = x
        clusterImages = []        
        out = classify(x) # out contains all images in cluster that have potholes
        if len(out) > 0:
            for z in out:
                temp = z.split("mask")[1]  #get orig image name
                clusterImages.append([temp,[temp[1],temp[2],temp[3],temp[4]]])    # type format [raw image path, [datetime, direction, gps, frame]]

        potholeDetected.append(clusterImages)
    
    return potholeDetected


keep_going = True
def key_capture_thread(): #exit inf loop on keypress
    global keep_going
    input()
    keep_going = False

def main():
    video.lower() = input('Webcam or pre-recorded?\n')
    if(video == "webcam"):
        getInput(1, "")
    else:
        vidPath = input("Input relative path to pre-recorded video:\n")
        getInput(0, vidPath)
     # Start grabbing frames from camera and GPS store to /data/

    th.Thread(target=key_capture_thread, args=(), name='key_capture_thread', daemon=True).start()
    while keep_going:
        event.wait(60)  # Check in every minute
        logging.debug("Capture still running")

    images = imageProcess()  # Recording finished, parse captured frames and run detection
    for x in images:
        lat = x[1][2].split("-")[1][:-3]
        lon = x[1][2].split("-")[2]
        direction = x[1][1]
        print("Pothole detected at Date/Time - %s, Lat: %s Long: %s - Name: %s\n See below for safety rating\n" % (x[1][0],lat,lon,x[0]))
        rating = safety(x[0], "mask"+x[0])

        if rating >= 0.7:
            print("Pothole has failed safety check, submitting relevant information to Department of Transit")
            submit(x[0], "mask"+x[0], lat, lon, x[1][0], direction)

main() #run main loop