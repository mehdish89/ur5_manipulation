#!/usr/bin/env python  

import rospy
import math
import numpy as np
import sys

from std_msgs.msg import String

def callback(msg):
	for i in msg.data:
		print("{0} --> {1}".format(i, ord(i)))

def listener():
	fh = open("script.txt", "r") 

	pub = rospy.Publisher('/ur_driver/URScript', String, queue_size=1)
	# rospy.Subscriber('/ur_driver/URScript', String, callback)
	rospy.init_node('ur_script_runner')



	rospy.sleep(0.3)
	
	rate = rospy.Rate(500)

	lines = fh.readlines()
	for line in lines:
		sys.stdout.write(">> " + line)
		pub.publish(line)
		rate.sleep()

	# rospy.spin()

if __name__ == '__main__':
    listener()