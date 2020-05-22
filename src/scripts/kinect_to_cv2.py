#!/usr/bin/env python
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from depth_calc.msg import array
from cv_bridge import CvBridge, CvBridgeError

cup_cascade = cv2.CascadeClassifier('/home/shubham/fyp_ws/src/position_calc/src/cup_tracking_cascades/cascade.xml')

class image_converter:

	def __init__(self):
		# self.image_pub = rospy.Publisher("image_topic_2",Image)

		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/camera/rgb/image_color",Image,self.callback)

  	def callback(self,data):
		frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cup_cascades = cup_cascade.detectMultiScale(gray, 2, 5, 0, minSize=(128,128), maxSize=(128,128))
		arr = []
		
		for (x,y,w,h) in cup_cascades:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
			cv2.circle(frame, (x+64,y+64), 1, (255,0,0), 3)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = frame[y:y+h, x:x+w]
			arr = [x+64,y+64]
			pub.publish(arr)
	
		cv2.imshow('imgage_capture',frame)
		cv2.waitKey(1)

		# cv2.imshow("Image window", cv_image)
		# cv2.waitKey(3)


def main(args):
  	rospy.init_node('image_converter', anonymous=True)
	ic = image_converter()

	rospy.spin()
	
	cv2.destroyAllWindows()

if __name__ == '__main__':
	pub = rospy.Publisher('centroid', array, queue_size = 10)
    
	main(sys.argv)