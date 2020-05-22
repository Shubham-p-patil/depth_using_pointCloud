#!/usr/bin/env python

import rospy
from std_msgs.msg import String, UInt32
from sensor_msgs.msg import PointCloud2
import ros_numpy
import numpy as np
import math
from depth_calc.msg import array as pcl_array,array_float
import cv2
import pandas as pd


def rotate(s,theta=0,axis='x'):
    """
    Counter Clock wise rotation of a vector s, along the axis by angle theta
    s:= array/list of scalars. Contains the vector coordinates [x,y,z]
    theta:= scalar, <degree> rotation angle for counterclockwise rotation
    axis:= str, rotation axis <x,y,z>
    """
    theta = np.radians(theta) # degree -> radians
    r = 0
    if axis.lower() == 'x':
        r = [s[0],
             s[1]*np.cos(theta) - s[2]*np.sin(theta),
             s[1]*np.sin(theta) + s[2]*np.cos(theta)]
    elif axis.lower() == 'y':
        r = [s[0]*np.cos(theta) + s[2]*np.sin(theta),
             s[1],
             -s[0]*np.sin(theta) + s[2]*np.cos(theta)]
    elif axis.lower() == 'z':
        r = [s[0] * np.cos(theta) - s[1]*np.sin(theta),
             s[0] * np.sin(theta) + s[1]*np.cos(theta),
             s[2]]
    else:
        print "Error! Invalid axis rotation"
    return r
#rot_vector

def on_new_point_cloud(data):
    global pcl_array
    pcl_array = ros_numpy.point_cloud2.pointcloud2_to_array(data,squeeze = True)

def on_new_centroid_array(data):
    global pcl_array
    centroid_array = data.array
    left = centroid_array[0]
    right = centroid_array[1]
    top = centroid_array[2]
    bottom = centroid_array[3]
    co_ordinates = []
    try:
        roi = pcl_array[bottom:top, left:right]
        # if remove_nans:
        #     mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        #     pcl_array_withoutNans = cloud_array[mask]
        del co_ordinates[:]
        print(roi.shape)
        for i in range(roi.shape[0]):
            for j in range(roi.shape[1]):
                # print(i,j)
                co_ordinates.append([roi['x'][i,j],roi['y'][i,j],roi['z'][i,j]])
        
        # print(co_ordinates)
        pcl_df = pd.DataFrame(data=co_ordinates,columns=['x','y','z'])
        print("Shape of dataframe")
        print(pcl_df.shape)
        # print(pcl_df)

        #Calculating the median of x,y,z coordinates
        pcl_median_df = pcl_df.median()
        print(pcl_median_df)
        arr = [pcl_median_df[0],pcl_median_df[1],pcl_median_df[2]]

        #Before Rotation publishing the coordinates
        pub_median.publish(arr)

        #After Rotation publishing the coordinates
        arr_ = rotate(arr,theta=180,axis='z')
        pub_median_.publish(arr_)

        # cv2.imshow('imgage_capture', roi)
        # cv2.waitKey(1)

    except TypeError,ValueError:
        pass

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/camera/depth_registered/points", PointCloud2, on_new_point_cloud)
    rospy.Subscriber("/bounding_box_coordinates", pcl_array, on_new_centroid_array)

if __name__ == '__main__':
   listener()
   pub_median = rospy.Publisher("position_median", array_float, queue_size = 10)
   pub_median_ = rospy.Publisher("position_median_after_rotation", array_float, queue_size = 10)

   rospy.spin()
