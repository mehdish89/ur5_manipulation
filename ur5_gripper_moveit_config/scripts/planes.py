#!/usr/bin/env python  

import rospy
import math
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import colorsys as cs
import tf

import cv2
import cv2.cv as cv

import struct
import ctypes

from random import randint

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from geometry_msgs.msg import Point

global pthr
global thr



thr = 4 * 0.001
pthr = 0.3



def rand_plane(npc):

	if(len(npc)==0):
		return np.array((0, 0, 0, 0))

	ia = randint(0,len(npc)-1)
	ib = randint(0,len(npc)-1)
	ic = randint(0,len(npc)-1)

	if(ia == ib and ib == ic): return rand_plane(npc)

	p1 = np.array((npc[ia][0], npc[ia][1], npc[ia][2]))
	p2 = np.array((npc[ib][0], npc[ib][1], npc[ib][2]))
	p3 = np.array((npc[ic][0], npc[ic][1], npc[ic][2]))
	
	# These two vectors are in the plane
	v1 = p3 - p1
	v2 = p2 - p1

	# the cross product is a vector normal to the plane
	cp = np.cross(v1, v2)
	a, b, c = cp
	cp = cp / math.sqrt(a**2 + b**2 + c**2)
	a, b, c = cp


	# This evaluates a * x3 + b * y3 + c * z3 which equals d
	d = -np.dot(cp, p3)

	return np.array([a, b, c, d])





def dist(plane, point):
	plane = np.array(plane)	
	point = np.array(point)

	
	a = plane[0]
	b = plane[1]
	c = plane[2]
	d = plane[3]

	x = point[0]
	y = point[1]
	z = point[2]

	
	# print(point, plane)

	return abs(a*x+b*y+c*z+d)/math.sqrt(a**2 + b**2 + c**2)

def clear_markers():
	global pub
	msg = MarkerArray()
	for i in range(30):
		marker = Marker()
		marker.id = i
		marker.action = 2
		msg.markers.append(marker)

	pub.publish(msg)


def read_pcl(cloud):
	pc = pc2.read_points(cloud, skip_nans=True, field_names=("x", "y", "z", "rgb"))
	npc = []

	for p in pc:
		npc.append(p)
	return npc	

def calc_points(npc, plane):
	global thr
	
	count = 0	
	mp = np.array((0.,0.,0.))

	for p in npc:
		ds = dist(plane, p)			
		if(ds<thr):
			mp += p[0:3]
			count += 1

	mp /= float(count)

	return count, mp

def create_marker(npc, plane):
	marker = Marker()
	marker.header.frame_id ="camera_depth_optical_frame"
	marker.type = 8
	marker.id = plane[5]
	
	# marker.pose.position.x = p[0]
	# marker.pose.position.y = p[1]
	# marker.pose.position.z = p[2]

	color = plane[5]+1

	marker.scale.x = 0.001
	marker.scale.y = 0.001
	marker.scale.z = 0.001

	marker.color.a = 1.0
	marker.color.r = color % 2
	marker.color.g = (color/2) % 2
	marker.color.b = (color/4) % 2

	for p in npc:
		ds = dist(plane, p)			
		if(ds<thr):
			point = Point()
			point.x = p[0]
			point.y = p[1]
			point.z = p[2]

			marker.points.append(point)
	return marker

def filter_top3(planes, valids):
	global pthr

	vplanes = [planes[i] for i in valids]
	splanes = sorted(vplanes, key=lambda x: x[4])

	indices = []

	for i in range(len(splanes)):
		pl = splanes[i]
		is_new = True
		for j in indices:
			plane = planes[j]
			if(abs(np.dot(pl[0:3], plane[0:3]))>pthr):
				is_new = False
				break
		if(is_new):
			indices.append(int(pl[5]))
	return indices

def middle_point(p1, d1, p2, d2):
	p1 = p1[0:3]
	p2 = p2[0:3]

	d1 = d1[0:3]
	d2 = d2[0:3]

	n = np.cross(d1, d2)
	n1 = np.cross(d1, n)
	n2 = np.cross(d2, n)	

	c1 = p1 + (np.dot(p2-p1, n2)/np.dot(d1, n2)) * d1
	c2 = p2 + (np.dot(p1-p2, n1)/np.dot(d2, n1)) * d2

	return (c1 + c2)/2

def average(npc):
	count = 0
	mp = np.array((0.,0.,0.))

	for p in npc:
		mp += p[0:3]
		count += 1

	if(count>0):
		mp /= float(count)
	return mp

def callback(cloud):
	global pub
	global pthr
	global thr
	print("start")

	npc = read_pcl(cloud)	

	valids = []
	planes = []
	means = []

	# clear_markers()
	msg = MarkerArray()

	for i in range(100):	
		

		plane = rand_plane(npc)
		count, mp = calc_points(npc, plane)
		plane = np.append(plane, [count, len(planes)])

		if(count<200): continue

		is_new = True
		current = valids[:]

		for i in current:
			pl = planes[i]
			if(abs(np.dot(pl[0:3], plane[0:3]))>pthr):
				if(plane[4]>pl[4]):
					valids.remove(i)
					is_new = True
					
				else:
					is_new = False

		if(not is_new): 
			continue

		marker = create_marker(npc, plane)

		msg.markers.append(marker)

		valids.append(len(planes))
		planes.append(plane)
		means.append(mp)
	
	clear_markers()

	# valids = range(len(planes))
	valids = filter_top3(planes, valids)

	cands = []
	for i in valids:
		for j in valids:
			print(i, j)
			if(j>i):
				p = middle_point(means[i], planes[i], means[j], planes[j])
				cands.append(p)
	
	p = np.array((0., 0., 0.))
	for q in cands:
		p += q / float(len(cands))

	if(float(len(valids)<=1)):
		p = average(npc)

	msg.markers = [msg.markers[i] for i in valids]
	planes = [planes[i] for i in valids]
	means = [means[i] for i in valids]

	for i in range(len(msg.markers)):
		color = i+1
		msg.markers[i].id = i
		msg.markers[i].color.r = color % 2
		msg.markers[i].color.g = (color/2) % 2
		msg.markers[i].color.b = (color/4) % 2


	marker = Marker()
	marker.header.frame_id ="camera_depth_optical_frame"
	marker.type = 2
	marker.id = len(valids)
	
	marker.pose.position.x = p[0]
	marker.pose.position.y = p[1]
	marker.pose.position.z = p[2]

	color = len(valids)+1

	marker.scale.x = 0.01
	marker.scale.y = 0.01
	marker.scale.z = 0.01

	marker.color.a = 1.0
	marker.color.r = color % 2
	marker.color.g = (color/2) % 2
	marker.color.b = (color/4) % 2

	msg.markers.append(marker)	
	
	pub.publish(msg)

	trans = p[0:3]
	rot = (0, 0, 0, 1)

	last_stamp = rospy.Time.now()

	br.sendTransform(trans,
                     rot,
                     last_stamp,
                     '/cube', 
                     '/camera_rgb_optical_frame')

	print(planes)
	print(means)

    
def listener():
	print(dist([1,1,1,-3],[0,0,3]))	
	global pub
	global br

	br = tf.TransformBroadcaster()
	pub = rospy.Publisher('/camera/depth/planes_center', MarkerArray, queue_size=1)
	rospy.init_node('plane_detector')
	rospy.Subscriber("/camera/depth/object", PointCloud2, callback, queue_size=1)
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