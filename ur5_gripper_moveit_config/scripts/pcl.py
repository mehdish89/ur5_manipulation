#!/usr/bin/env python  

import rospy
import math
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import colorsys as cs

import cv2
import cv2.cv as cv

import struct
import ctypes


def callback(cloud):
	# rospy.loginfo(cloud)
	global obz

	pc = pc2.read_points(cloud, skip_nans=True, field_names=("x", "y", "z", "rgb"))
	npc = []
	obz = []

	bh = rospy.get_param("/camera/driver/h_range")
	bs = rospy.get_param("/camera/driver/s_range")
	bv = rospy.get_param("/camera/driver/v_range")

	gh = rospy.get_param("/camera/driver/h_goal")
	gs = rospy.get_param("/camera/driver/s_goal")
	gv = rospy.get_param("/camera/driver/v_goal")

	count = 0


	for p in pc:

		count += 1

		x = p[0]
		y = p[1]
		z = p[2]
		rgb = p[3]

		if((x,y,z) == (0,0,0)):
			continue

		

		# cast float32 to int so that bitwise operations are possible
		s = struct.pack('>f' ,rgb)
		i = struct.unpack('>l',s)[0]
		# you can get back the float value by the inverse operations
		pack = ctypes.c_uint32(i).value
		r = (pack & 0x00FF0000)>> 16
		g = (pack & 0x0000FF00)>> 8
		b = (pack & 0x000000FF)


		hsv = np.array(cs.rgb_to_hsv(r/255., g/255., b/255.))
		ghsv = np.array((gh, gs, gv))#cs.rgb_to_hsv(207./255., 127./255., 0./255.))
		# ghsv = np.array((14, 92, 48))

		# print bh


		bd = np.array((bh, bs, bv))
		lb = ghsv - bd
		ub = ghsv + bd

		# print("####")

		# print(hsv)
		# print(ghsv)


		if np.sum(np.logical_and(hsv>lb, hsv<ub))<3:
			# print(np.logical_and(hsv>lb, hsv<ub))
			continue


		

		# g = 0
		# b = 0

		# print (r,g,b)

		pack = ctypes.c_float((r << 16) + (g << 8) + b).value

		s = struct.pack('>l' ,pack)
		rgb = struct.unpack('>f',s)[0]

		# print(p)

		if (r,g,b) == (96, 157, 198):
			continue


		npc.append((x,y,z,rgb))
		obz.append((x, y, z, r, g, b))
	
	print(count)

	msg = pc2.create_cloud(cloud.header, cloud.fields, npc)

	# print(msg)

	


	global pub
	pub.publish(msg)
	# print msg
    
    # fmt = pc2._get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names=("x", "y", "z"))
    # width, height, point_step, row_step, data, isnan = cloud.width, cloud.height, cloud.point_step, cloud.row_step, cloud.data, math.isnan
    # unpack_from = struct.Struct(fmt).unpack_from

    # print(dir(fmt))

    
def listener():
	global pub
	pub = rospy.Publisher('/camera/depth/object', PointCloud2, queue_size=1)
	rospy.init_node('object_filter')
	rospy.Subscriber("/camera/depth/pointsSOURCE", PointCloud2, callback, queue_size=1)
    # spin() simply keeps python from exiting until this node is stopped
	rospy.spin()

"""
def on_scan(self, scan):
    rospy.loginfo("Got scan, projecting")
    cloud = self.laser_projector.projectLaser(scan)
    gen = pc2.read_points(cloud, skip_nans=True, field_names=("x", "y", "z"))
    self.xyz_generator = gen
"""

if __name__ == '__main__':
    listener()