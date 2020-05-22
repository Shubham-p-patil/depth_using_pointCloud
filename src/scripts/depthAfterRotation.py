#!/usr/bin/env python

import rospy
from std_msgs.msg import String, UInt32
from sensor_msgs.msg import PointCloud2
import ros_numpy
import numpy as np
import math
from depth_calc.msg import array,array_float


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
    global array
    array = ros_numpy.point_cloud2.pointcloud2_to_array(data,squeeze = True)

def on_new_centroid_array(data):
    global array
    centroid_array = data.array
    try:
        x = array[centroid_array[1],centroid_array[0]][0]
        y = array[centroid_array[1],centroid_array[0]][1]
        z = array[centroid_array[1],centroid_array[0]][2]
        arr = [x,y,z]
        arr = rotate(arr,theta=180,axis='z')
        pub.publish(arr)
        print("--------------------------------------------")
        print("Centroid:",centroid_array)
        print("XYZ:",arr)

    except TypeError,ValueError:
        pass

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/camera/depth_registered/points", PointCloud2, on_new_point_cloud)
    rospy.Subscriber("/centroid", array, on_new_centroid_array)

if __name__ == '__main__':
   listener()
   pub = rospy.Publisher("postion_after_rotation", array_float, queue_size = 10)
   rospy.spin()
