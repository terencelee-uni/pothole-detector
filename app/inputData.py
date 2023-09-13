from gpsdclient import GPSDClient #https://gpsd.gitlab.io/gpsd/ | https://github.com/tfeldmann/gpsdclient
import numpy as np
import cv2 as cv
from datetime import datetime, date

def getInput(flag, vidPath): #flag determines whether pre-recorded video or live video feed
    global keep_going   #set interrupt flag

    if(flag):
        cap = cv.VideoCapture(0)
    else:
        cap = cv.VideoCapture(vidPath)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    # not sure if frame count starts at 0 or 1? have to test this
    fps = int(cap.get(cv2.CAP_PROP_FPS))    # assume 60 fps camera but should work for other fps

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        if not keep_going:
            cap.release()
            break
        cluster = str(date.today()) + "|" + datetime.now().strftime("%H")
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            cap.release()
            break
        oldLat = 0.0
        oldLon = 0.0
        direction = ""
        counter = 0
        if frameCounter % abs(fps/2) == 0:  # 2 frames per second
            gps = getGPS()
            counter += 1
            latDiff = gps.split("-")[1][:-3] - oldLat   #track direction traveling
            lonDiff = gps.split("-")[2] - oldLon
            if abs(latDiff) > abs(lonDiff):
                if latDiff > 0:
                    direction = "North"
                else:
                    direction = "South"
            else:
                if lonDiff > 0:
                    direction = "East"
                else:
                    direction = "West"
            if counter == 60:    #update direction once every 30 seconds
                counter = 0
                oldLat = gps.split("-")[1][:-3]
                oldLon = gps.split("-")[2]
            dateTime = str(datetime.now().strftime("%H:%M")) + "|" + str(date.today())
            #cv2.imshow()
            cv2.imwrite("data/%s/" % cluster + dateTime + "|" + direction + "|" + gps + "|" + "frame%d" + ".jpg" % int(frameCounter/(fps/2)), frame)
        if cv.waitKey(1) == ord('q'):
            break
        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

    
    
def getGPS():   # google maps gps search works [Lat, Longitude] https://gpsd.gitlab.io/gpsd/gpsd.html#_logging
    with GPSDClient() as client:    #gpsd py library
        for result in client.dict_stream(convert_datetime=True, filter=["TPV"]):
            return "Lat-%sLon-%s" % (result.get("lat", "n/a"), result.get("lon", "n/a"))