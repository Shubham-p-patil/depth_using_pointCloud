import numpy as np
import cv2


cup_cascade = cv2.CascadeClassifier('/home/shubham/identical-object-tracking-under-occlusion/cup_tracking_cascades/cascade.xml')

img = cv2.imread('/home/shubham/identical-object-tracking-under-occlusion/training_data/positive/cup_positive_0.png')
cv2.imshow('frame',img)
cv2.waitKey(5)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('frame',gray)
cv2.waitKey(5)

cup_cascades = cup_cascade.detectMultiScale(gray, 1.3, 5)
print cup_cascades
for (x,y,w,h) in cup_cascades:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            # cv2.imshow('imgage_capture',roi_color)
            # cv2.waitKey(5)

ret,thresh = cv2.threshold(roi_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('frame',thresh)
cv2.waitKey(5)

image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
print len(contours)

img = cv2.drawContours(roi_color, contours, 0, (0,255,0), 3)
cv2.imshow('frame',img)
cv2.waitKey(0)

cv2.destroyAllWindows()