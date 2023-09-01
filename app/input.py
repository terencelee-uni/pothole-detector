from gpsdclient import GPSDClient #https://gpsd.gitlab.io/gpsd/ | https://github.com/tfeldmann/gpsdclient
import numpy as np
import cv2 as cv
from datetime import datetime, date

def getInput():
    cap = cv.VideoCapture(0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    # not sure if frame count starts at 0 or 1? have to test this
    fps = int(cap.get(cv2.CAP_PROP_FPS))    # assume 60 fps camera but should work for other fps

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        cluster = date.today() + "-" + datetime.now().strftime("%H")
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            cap.release()
            break
        if frameCounter % abs(fps/2) == 0:  # 2 frames per second
            dateTime = datetime.now().strftime("%H:%M") + "-" + date.today()
            #cv2.imshow()
            cv2.imwrite("data/%s/" % cluster + dateTime + "frame%d.jpg" % int(frame/(fps/2)), frame)
        if cv.waitKey(1) == ord('q'):
            break
        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

    
    
def getGPS():   # google maps gps search works [Lat, Longitude]
    with GPSDClient() as client:    #gpsd py library
        for result in client.dict_stream(convert_datetime=True, filter=["TPV"]):
            return "Lat-%sLon-%s" % (result.get("lat", "n/a"), result.get("lon", "n/a"))