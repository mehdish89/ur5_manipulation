#!/usr/bin/env python  

import roslib
# roslib.load_manifest('learning_tf')
import rospy
import math
import tf
import geometry_msgs.msg
from apriltags_ros.msg import AprilTagDetectionArray
# import turtlesim.srv


def callback(data):
    # rospy.loginfo("I heard %s",data)
    
    for detection in data.detections:
        if(detection.id == 5):
            global trans, rot
            global last_stamp

            last_stamp = detection.pose.header.stamp

            p = detection.pose.pose.position
            r = detection.pose.pose.orientation
            
            trans = [p.x, p.y, p.z]
            rot = [r.x, r.y, r.z, r.w]

            print("Gotcha " + str(last_stamp))
            # try:
            #     (trans,rot) = listener.lookupTransform('/camera_rgb_optical_frame', 'zero', rospy.Time(0))
            # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            #     print("Exception caught")
            #     continue

if __name__ == '__main__':
    rospy.init_node('tf_publisher')

    rospy.Subscriber("/tag_detections", AprilTagDetectionArray, callback)

    global last_stamp

    last_stamp = rospy.Time(0)

    listener = tf.TransformListener()
    br = tf.TransformBroadcaster()

    rate = rospy.Rate(50)

    global trans, rot

    trans = [0, 0, 0];
    rot = [0, 0, 0, 1];


    while not rospy.is_shutdown():
        print(trans, rot)

        last_stamp = rospy.Time().now()

        br.sendTransform(trans,
                         rot,
                         last_stamp,
                         '/zero', 
                         '/camera_rgb_optical_frame')

        # try:
        #     (ztrans,zrot) = listener.lookupTransform('tool', 'zero', rospy.Time(0))
        # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        #     print("Exception caught")
        #     continue

        # ztrans = (ztrans[0]+0.03,ztrans[1]+0.01, ztrans[2]+0.065) # tag offset

        # br.sendTransform(ztrans,
        #                  zrot,
        #                  rospy.Time.now(),
        #                  "zero",
        #                  "ee_link")

        

        ###########  Transform from \zero to \world  ############

        # - Translation: [-1.082, -0.712, -0.088]
        # - Rotation: in Quaternion [0.051, -0.028, 0.004, 0.998]
        #             in RPY (radian) [0.103, -0.057, 0.005]
        #             in RPY (degree) [5.892, -3.281, 0.262]

        # - Translation: [-1.078, -0.552, -0.169]
        # - Rotation: in Quaternion [-0.011, -0.007, 0.035, 0.999]
        #             in RPY (radian) [-0.022, -0.014, 0.071]
        #             in RPY (degree) [-1.289, -0.778, 4.064]



        # ztrans = (-1.082, -0.712, -0.088)
        # zrot = (0.051, -0.028, 0.004, 0.998)

        xoff = rospy.get_param("/camera/driver/x_offset")
        yoff = rospy.get_param("/camera/driver/y_offset")
        zoff = rospy.get_param("/camera/driver/z_offset")

        thoff = rospy.get_param("/camera/driver/th_offset")


        # ztrans = (-1.078 + xoff, -0.552 + yoff, -0.169 + zoff)
        # # zrot = (-0.011, -0.007, 0.035, 0.999) 
        # zrot = (0, 0, 0.035 + thoff, 1) 

        ztrans = (-1.100 + xoff, -0.497 + yoff, -0.144 + zoff)
        # zrot = (-0.011, -0.007, 0.035, 0.999) 
        zrot = (0, 0, 0.045 + thoff, 1)

        br.sendTransform(ztrans,
                         zrot,
                         rospy.Time.now(),
                         "/world",
                         "/zero")
        
        rate.sleep()