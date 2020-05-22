#!/usr/bin/env python
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image as ImageMsg
from depth_calc.msg import array
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython import get_ipython
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("/home/shubham/fyp_ws/src/depth_calc/src/tf_objDetection/models/research/")
sys.path.append("/home/shubham/fyp_ws/src/depth_calc/src/tf_objDetection/models/research/object_detection")

from object_detection.utils import ops as utils_ops

# This is needed to display the images.
# get_ipython().run_line_magic('matplotlib', 'inline')

from utils import label_map_util

from utils import visualization_utils as vis_util
# What model to download.# What  

MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join('/home/shubham/fyp_ws/src/depth_calc/src/tf_objDetection/models/research/object_detection',MODEL_NAME ,'frozen_inference_graph.pb')
print(os.path)

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/home/shubham/fyp_ws/src/depth_calc/src/tf_objDetection/models/research/object_detection','data', 'mscoco_label_map.pbtxt')
print(PATH_TO_LABELS)

NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


class image_converter:

	def __init__(self):
		# self.image_pub = rospy.Publisher("image_topic_2",Image)

		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/camera/rgb/image_color",ImageMsg,self.callback)

  	def callback(self,data):
            # print(data)
            arr = []
            display_str=''
            box = []
            with detection_graph.as_default():
                with tf.Session(graph=detection_graph) as sess:
                    image_np = self.bridge.imgmsg_to_cv2(data, "bgr8") # very first webcam attach
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)

                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                    #print num_detections
                    

                    boxes = np.squeeze(boxes)
                    classes= np.squeeze(classes)
                    scores = np.squeeze(scores)

                    max_boxes_to_draw = 20
                    cv2.imshow('imgage_capture',image_np)
                    cv2.waitKey(1)
                    i = 0
                    # for i in range(min(max_boxes_to_draw, boxes.shape[0])):
                    if scores[i]>0.5:
                        box = tuple(boxes[i].tolist())
                        if classes[i] in category_index.keys():
                            class_name = category_index[classes[i]]['name']
                        else:
                            class_name = 'N/A'
                        display_str = str(class_name)
                        print( "Mera wala")
                        print (display_str)
                        print (box)
                        print ("Mera wala end")
                        im_width = 640
                        im_height = 480
                        ymin, xmin, ymax, xmax = box
                        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
                        arr = [round((left+right)/2),round((top+bottom)/2)]
                        print(arr)
                        pub.publish(arr)

                    # Visualization of the results of a detection.
                    # vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                    #     np.squeeze(boxes),
                    #     np.squeeze(classes).astype(np.int32),
                    #     np.squeeze(scores),
                    #     category_index,                         
                    #     use_normalized_coordinates=True,
                    #     line_thickness=8)

                    # cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
                    # cv2.waitKey(1)



def main(args):
      	rospy.init_node('image_converter', anonymous=True)
	ic = image_converter()

	rospy.spin()
	
	cv2.destroyAllWindows()

if __name__ == '__main__':
	pub = rospy.Publisher('centroid', array, queue_size = 10)
    
	main(sys.argv)