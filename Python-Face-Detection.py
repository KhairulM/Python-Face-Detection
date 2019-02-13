import cv2
import numpy as np
import urllib
import argparse
from imutils.video import FPS
import time

# Setting up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type = str, help = "Path to the video file that you want to use")
ap.add_argument("-s", "--stream", type = str, help = "URL to the video stream that you want to use")
args = vars(ap.parse_args())

noStream = False
noVideo = False

if not args.get("stream", False):
    noStream = True
if not args.get("video", False):
    noVideo = True 

if not noStream and not noVideo:
    print("[ERROR] : Could only get stream from one source")
    quit()

# Setting up cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Setting up video capture
if noStream and noVideo:
    vs = cv2.VideoCapture(0)
    time.sleep(1.0)
elif noStream:
    vs = cv2.VideoCapture(args["video"])

# Initializing FPS counter
fps = FPS().start()

while True:
    if noStream:
        ret, frame = vs.read()
    else:
        # Use urllib to get the image and convert into a cv2 usable format
        imgResp=urllib.request.urlopen(args["stream"])
        imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
        frame=cv2.imdecode(imgNp,-1)

    # Resizing the frame so we can do faster calculation
    f_height, f_width = frame.shape[:2]
    frame = cv2.resize(frame, ((f_width*500) // f_height, 500), interpolation = cv2.INTER_AREA)

    # Grayscale image
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting faces
    faces = face_cascade.detectMultiScale(grayFrame, scaleFactor = 1.2, minNeighbors = 5)
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Updating fps
    fps.update()
    fps.stop()

    # Printing fps to frame
    cv2.putText(frame, "FPS : {:.2f}".format(fps.fps()), (0, f_height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

if noStream:
    vs.release()

cv2.destroyAllWindows()


    