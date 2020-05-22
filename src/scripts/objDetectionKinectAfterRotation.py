#!/usr/bin/env python


from __future__ import print_function

import roslib
import sys
import rospy
import cv2
import cv_bridge
from std_msgs.msg import String
from depth_calc.msg import array
import numpy as np
import os
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import freenect
from sensor_msgs.msg import Image
import time
import imutils

# image = []
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append(
    "/home/shubham/fyp_ws/src/depth_calc/src/tf_objDetection/models/research/")
sys.path.append(
    "/home/shubham/fyp_ws/src/depth_calc/src/tf_objDetection/models/research/object_detection")

from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util
# What model to Use
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join(
    '/home/shubham/fyp_ws/src/depth_calc/src/tf_objDetection/models/research/object_detection', MODEL_NAME, 'frozen_inference_graph.pb')
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(
    '/home/shubham/fyp_ws/src/depth_calc/src/tf_objDetection/models/research/object_detection', 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
count = 0
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Initialisation Of cap object for capturing frame
cap = cv2.VideoCapture(0)

#function to get RGB image from kinect
# def get_video():
#     array,_ = freenect.sync_get_video()
#     array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
#     return array
class Frame:
    
	def __init__(self):
            print("initialize")
            self.bridge = cv_bridge.CvBridge()
            self.image_sub = rospy.Subscriber('/camera/rgb/image_raw',Image, self.img_callback)
            print("done")

	def img_callback(self,data):
            global image
            image = self.bridge.imgmsg_to_cv2(data,desired_encoding='bgr8')

class pixelCoordinate_converter:

    def __init__(self):

        arr = []
        display_str = ''
        box = []
        detection_graph.as_default()
        sess = tf.Session(graph=detection_graph)
        self.coordinates_generator(arr,display_str,box,sess)

    def coordinates_generator(self,arr,display_str,box,sess):
        global image
        global count
        while not rospy.is_shutdown():
            #get a frame from RGB camera
            image_np = imutils.rotate_bound(image,angle=180)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            image_tensor = detection_graph.get_tensor_by_name(
                'image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name(
                'detection_scores:0')

            classes = detection_graph.get_tensor_by_name(
                'detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')

            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes)
            scores = np.squeeze(scores)
            
            max_boxes_to_draw = 5
            for i in range(max_boxes_to_draw):
                if scores[i] > 0.5:
                    box = tuple(boxes[i].tolist())
                    if classes[i] in category_index.keys():
                        class_name = category_index[classes[i]]['name']
                    else:
                        class_name = 'N/A'
                    display_str = str(class_name)
                    
                    im_width = 640
                    im_height = 480
                    ymin, xmin, ymax, xmax = box
                    (left, right, bottom, top) = (xmin * im_width,
                                                xmax * im_width, ymin * im_height, ymax * im_height)
                    arr = [int(round((left+right)/2)), int(round((top+bottom)/2))]
                    arr_ = [640-int(round(left)), 640-int(round(right)), 480-int(round(top)), 480-int(round(bottom))]
                    
                    cv2.circle(image_np, (arr[0],arr[1]), 5, (255,0,0), 3)
                    cv2.circle(image_np, (0,0), 5, (0,255,0), 3)
                    cv2.circle(image_np, (480,0), 5, (0,0,255), 3)
                    cv2.circle(image_np, (640,480), 5, (255,0,0), 3)
                    cv2.rectangle(image_np,(arr_[0],arr_[3]),(arr_[1],arr_[2]),(255,0,0), 3)

                    if display_str == 'bottle' and count>20:   
                        print (display_str)
                        print (box)
                        print("pixelCoordinates of bounding box:")
                        print(arr_)
                        print("Centroid pixelCoordinates of bounding box:")
                        print(arr)
                        print("Centroid pixelCoordinates of box after rotation:")
                        print(640-arr[0],480-arr[1])

                        arr = [640-arr[0],480-arr[1]]
                        pub_c.publish(arr)
                        pub_box.publish(arr_)
                        count=0

            
            count = count + 1
            cv2.imshow('image_capture', image_np)
            cv2.waitKey(1)


def main(args):
    
    pc = pixelCoordinate_converter()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    rospy.init_node('pixel_coordinates', anonymous=True)
    pub_c = rospy.Publisher('centroid', array, queue_size=10)
    pub_box = rospy.Publisher('bounding_box_coordinates', array, queue_size=10)

    Frame()
    time.sleep(10)
    main(sys.argv)
    # rospy.spin()
