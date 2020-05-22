# This script is used to capture a video
# Press 'q' to end recording

import numpy as np
import cv2

cap = cv2.VideoCapture(2)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('video.avi',fourcc, 20.0, (640,480),True)

i = 1
while(cap.isOpened() and i<469):
    img = "../videos/video{}.jpg".format(i)			# Enter the path to save the video along with the name 
    frame = cv2.imread(img)
    i = i+1
    if True:
        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
