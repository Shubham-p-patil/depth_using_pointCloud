#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
# from std_msgs.msg import Int32
from vision.msg import array
import cv2
import cv_bridge
import numpy as np

# face_cascade = cv2.CascadeClassifier('/home/awadhut/classifier/cascade.xml')

a = array()

class Frame:

	def __init__(self):
		self.bridge = cv_bridge.CvBridge()
		# cv2.namedWindow("window1", 1)
		cv2.namedWindow("window2", 2)
		# cv2.namedWindow("window3", 3)
		self.image_sub = rospy.Subscriber('camera/rgb/image_raw',Image, self.img_callback)

	def img_callback(self,data):
		global M,cx,cy,a
		image = self.bridge.imgmsg_to_cv2(data,desired_encoding='bgr8')
		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		# print(image.shape)
		image[201,200] = [255,255,255]
		image[200,200] = [255,255,255]
		image[202,200] = [255,255,255]
		image[203,200] = [255,255,255]
		image[200,201] = [255,255,255]
		image[201,201] = [255,255,255]
		image[202,201] = [255,255,255]
		image[203,201] = [255,255,255]
		px2 = hsv[202,201]
		px1 = hsv[200,201]
		px3 = hsv[201,201]
		px4 = hsv[203,201]
		print(px1,px2,px3,px4)
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
		mask_blue = cv2.inRange(hsv, np.array((0., 0.,215.)), np.array((130.,55.,255.)))
		opening_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
		dilation = cv2.dilate(opening_blue,kernel,iterations = 1)
		contours_blue, hierarchy_b = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(image, contours_blue, -1, (0,255,0), 2)
		cx = [0]*len(contours_blue)
		cy = [0]*len(contours_blue)
		arr = []
		# print(len(contours_blue))
		for i in range(0,len(contours_blue)):
			M = cv2.moments(contours_blue[i])
			cx[i] = int(M['m10']/M['m00'])
			arr.append(cx[i])
			cy[i] = int(M['m01']/M['m00'])
			arr.append(cy[i])

		# print(arr)
		image[arr[1],arr[0]] = [255,0,0]
		image[arr[1],arr[0]+1] = [255,0,0]
		image[arr[1]+1,arr[0]] = [255,0,0]
		image[arr[1]+1,arr[0]+1] = [255,0,0]
		pub.publish(arr)
		cv2.imshow('window2',image)
		# cv2.imshow('window3',dilation)
		cv2.waitKey(1)



if __name__ == '__main__':

	rospy.init_node('image_subscriber', anonymous = True)
	pub = rospy.Publisher('centroid', array, queue_size = 10)
	fgbg = cv2.BackgroundSubtractorMOG()
	frame = Frame()
	rospy.spin()
