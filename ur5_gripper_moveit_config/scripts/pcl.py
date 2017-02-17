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

from visualization_msgs.msg import Marker

from geometry_msgs.msg import Point

from scipy.cluster.vq import vq, kmeans, whiten

import tf

def visualize(points):
	marker = Marker()
	marker.header.frame_id ="camera_depth_optical_frame"
	marker.type = 8
	marker.id = 1
	
	# marker.pose.position.x = p[0]
	# marker.pose.position.y = p[1]
	# marker.pose.position.z = p[2]

	color = 2

	marker.scale.x = 0.01
	marker.scale.y = 0.01
	marker.scale.z = 0.01

	marker.color.a = 1.0
	marker.color.r = color % 2
	marker.color.g = (color/2) % 2
	marker.color.b = (color/4) % 2

	for p in points:
		point = Point()
		point.x = p[0]
		point.y = p[1]
		point.z = p[2]

		marker.points.append(point)

	mpub = rospy.Publisher('/objects', Marker, queue_size=1)
	mpub.publish(marker)

def cluster(npc):
	print("clustering")
	points = np.array(npc)
	points = points[:, 0:3]

	dev = np.std(points, axis=0)#, ddof=1)

	# print("whitening")
	points = whiten(points)
	# print(points)

	
	for k in range(1,10):
		# k = 5
		clusters, dist = kmeans(points, k)
		print(k, dist)
		if(dist<0.20):
			break
		# print(clusters)
	
	clusters *= dev

	sorted_clusters = sorted(clusters, key=lambda x:x[2])

	visualize(sorted_clusters)

	# print(sorted_clusters)

	return sorted_clusters

	



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


		npc.append([x,y,z,rgb])
		obz.append((x, y, z, r, g, b))
	
	print(count)

	msg = pc2.create_cloud(cloud.header, cloud.fields, npc)

	# print("before clustering")
	clusters = cluster(npc)

	last_stamp = rospy.Time.now()

	br = tf.TransformBroadcaster()
	br.sendTransform(clusters[0],
                     np.array([0, 0, 0, 1]) ,
                     last_stamp,
                     '/cube', 
                     '/camera_rgb_optical_frame')	

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