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

"""
def callback(cloud):
	# rospy.loginfo(cloud)
	global obz

	pc = pc2.read_points(cloud, skip_nans=True, field_names=("x", "y", "z", "rgb"))
	npc = []
	obz = []

	bh = rospy.get_param("/camera/driver/h_range")
	bs = rospy.get_param("/camera/driver/s_range")
	bv = rospy.get_param("/camera/driver/v_range")

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
		ghsv = np.array(cs.rgb_to_hsv(207./255., 127./255., 0./255.))
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

# import cv2
# import numpy as np
"""

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False

 
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
 
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False

		x1 = refPt[0][1]
		x2 = refPt[1][1]
		y1 = refPt[0][0]
		y2 = refPt[1][0]

 		# print(x1,x2, y1, y2)

		# draw a rectangle around the region of interest
		# cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		global image

		# print(refPt)

		xm = (x1+x2)/2
		ym = (y1+y2)/2		

		xb = float(abs(x2 - x1)/2)
		yb = float(abs(y2 - y1)/2)

		print("##########")
		print(image[xm, ym])
		print(xb/size, yb/size)




		image = image[min(x1,x2):max(x1,x2),min(y1,y2):max(y1,y2)].copy()


		# print(image2.shape)


		cv2.imshow("image", image)


def generate(h, s, v, hb, sb, vb):

	# h *= 255
	# s *= 255
	# v *= 255

	# hb *= 255
	# sb *= 255
	# vb *= 255
	global size

	size = 400
	hsize  = size/2

	height = size
	width = size

	rgb = np.zeros((height,width,3), np.uint8)
	hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
	hsv = np.zeros((height,width,3), np.float32)
	# hsv = rgb

	for x in range(-hsize, hsize-1):
		for y in range(-hsize, hsize-1):
			hp = h +(x * hb * 1./hsize) 
			sp = s +(y * sb * 1./hsize) 
			# print (x, y)

			xp = x + hsize
			yp = y + hsize

			hsv[xp][yp][0] = hp # * 255
			hsv[xp][yp][1] = sp #* 255
			hsv[xp][yp][2] = v #* 255

	# hsv *= 255.
	# print(hsv)

	global image
	image = hsv

	rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

	cv2.namedWindow("image")
	cv2.setMouseCallback("image", click_and_crop)

	cv2.imshow("image", hsv)
	cv2.waitKey()
	cv2.destroyAllWindows()

    
def listener():

	generate(0.5, 0.5, 0.7, 0.5, 0.5, 0.5)

	# global pub
	# pub = rospy.Publisher('/camera/depth/object', PointCloud2, queue_size=1)
	# rospy.init_node('object_filter')
	# rospy.Subscriber("/camera/depth/pointsSOURCE", PointCloud2, callback, queue_size=1)
 #    # spin() simply keeps python from exiting until this node is stopped
	# rospy.spin()


if __name__ == '__main__':
    listener()