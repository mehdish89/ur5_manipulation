#!/usr/bin/env python  
import roslib
# roslib.load_manifest('learning_tf')
import rospy
import math
import tf
import geometry_msgs.msg
# import turtlesim.srv

if __name__ == '__main__':
    rospy.init_node('tf_publisher')

    listener = tf.TransformListener()
    br = tf.TransformBroadcaster()

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        # try:
        #     (trans,rot) = listener.lookupTransform('/ee_link', '/world', rospy.Time(0))
        # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        #     continue

        # trans = (trans[0],trans[1], trans[2]-0.04) # tag offset

        ###########  Transform from \zero to \world  ############

        # - Translation: [-1.082, -0.712, -0.088]
        # - Rotation: in Quaternion [0.051, -0.028, 0.004, 0.998]
        #             in RPY (radian) [0.103, -0.057, 0.005]
        #             in RPY (degree) [5.892, -3.281, 0.262]


        trans = (-1.082, -0.712, -0.088)
        rot = (0.051, -0.028, 0.004, 0.998) 

        br.sendTransform(trans,
                         rot,
                         rospy.Time.now(),
                         "world",
                         "zero")
        
        rate.sleep()