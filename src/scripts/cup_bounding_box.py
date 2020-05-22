#/usr/bin/python env

import numpy as np
import cv2 as cv

cup_cascade = cv.CascadeClassifier('/home/shubham/fyp_ws/src/position_calc/src/cup_tracking_cascades/cascade.xml')

def haar_classifier(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cup_cascades = cup_cascade.detectMultiScale(gray, 2, 5, 0, minSize=(128,128), maxSize=(128,128))
    cv.imshow('img',frame)
    cv.waitKey(1)
    for (x,y,w,h) in cup_cascades:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv.imshow('imgage_capture',frame)
        cv.waitKey(1)
    cv.imshow('imgage_capture',frame)
    cv.waitKey(1)

    return cup_cascades, frame
