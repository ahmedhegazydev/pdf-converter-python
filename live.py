# This script will detect faces via your webcam.
# Tested with OpenCV3

import cv2
import numpy as np

# import cvlib as cv
# from imutils.video import VideoStream
# from imutils.video import FPS
# import argparse
# import imutils


cap = cv2.VideoCapture(0)

template = cv2.imread("E:/Python projects/MatchingImages/ahmed_after_cuttting.png",
                      cv2.IMREAD_GRAYSCALE)
template1 = cv2.imread("E:/Python projects/MatchingImages/kamal2.png",
                       cv2.IMREAD_GRAYSCALE)
template2 = cv2.imread("E:/Python projects/MatchingImages/medhat.png",
                       cv2.IMREAD_GRAYSCALE)

w, h = template.shape[::-1]
w1, h1 = template1.shape[::-1]
w2, h2 = template2.shape[::-1]

font = cv2.FONT_HERSHEY_PLAIN

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while (True):
    # Capture frame-by-frame
    # ret, frame = cap.read()
    _, frame = cap.read()
    # Our operations on the frame come here
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
    res1 = cv2.matchTemplate(gray_frame, template1, cv2.TM_CCOEFF_NORMED)
    res2 = cv2.matchTemplate(gray_frame, template2, cv2.TM_CCOEFF_NORMED)

    thres = 1
    loc = np.where(res >= thres)

    loc1 = np.where(res1 >= thres)
    loc2 = np.where(res2 >= thres)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        # flags = cv2.CV_HAAR_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for pt in zip(*loc[::-1]):
            # cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)
            cv2.putText(frame, 'Ahmed', (pt[0] + w, pt[1] + h), font, 3, (255, 255, 255), 1)
        for pt1 in zip(*loc1[::-1]):
            # cv2.rectangle(frame, pt1, (pt1[0] + w1, pt1[1] + h1), (0, 255, 0), 3)
            cv2.putText(frame, 'Kamal', (pt1[0] + w1, pt1[1] + h1), font, 3, (255, 255, 255), 1)
        for pt2 in zip(*loc2[::-1]):
            # cv2.rectangle(frame, pt2, (pt2[0] + w2, pt2[1] + h2), (0, 255, 0), 3)
            cv2.putText(frame, 'Medhat', (pt2[0] + w2, pt2[1] + h2), font, 3, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
