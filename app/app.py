import numpy as np
import os, json, cv2, random
import classify
import threading as th




def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def imageProcess():
    subDir = get_immediate_subdirectories("data")

    potholeDetected = []
    for x in subDir:
        out = classify(x) # Image bunch exported from camera for detection
        out.append(x)
        potholeDetected.append(out) # get list containing image cluster and masks of detected potholes

    for x in potholeDetected:
def getImage():



keep_going = True
def key_capture_thread():
    global keep_going
    input()
    keep_going = False

def main():
    th.Thread(target=key_capture_thread, args=(), name='key_capture_thread', daemon=True).start()
    while keep_going:
        print('Waiting for image \n')

main()