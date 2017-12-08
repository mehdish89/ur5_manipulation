#!/usr/bin/env python

import cv2
import cv2.cv as cv
import numpy as np

import sys
import getopt

from math import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons

import threading
import time

import rospy
from geometry_msgs.msg import PointStamped

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from scipy.cluster.vq import vq, kmeans, whiten

from itertools import product

from threading import Lock

from std_msgs.msg import Float32MultiArray as FArray

from shape_detection import detect_boxes




def cluster(inp, thr = .20):
	# print("clustering")
	points = np.array(inp)
	points = points[:, 0:2]

	dev = np.std(points, axis=0)#, ddof=1)

	# print("whitening")
	points = whiten(points)
	# print(points)

	
	for k in range(1,10):
		# k = 5
		clusters, dist = kmeans(points, k)
		print(rospy.Time.now())
		# print(k, clusters)
		if(dist<thr):
			break
		# print(clusters)
	
	clusters *= dev



	# sorted_clusters = sorted(clusters, key=lambda x:x[2])

	# visualize(sorted_clusters)

	# print(sorted_clusters)

	return clusters


def length(line):
	ps1 = np.array(line[0])
	ps2 = np.array(line[1])
	l = ps2 - ps1

	return sqrt(sum(l**2))


def find_scale(src, dst):
	return length(dst)/length(src)

def find_trans(src, dst):
	ps1 = np.array(src[0])
	ps2 = np.array(src[1])

	pd1 = np.array(dst[0])
	pd2 = np.array(dst[1])

	psm = (ps1 + ps2)/2
	pdm = (pd1 + pd2)/2

	trans = pdm - psm
	return trans

def find_rot(src, dst):
	ps1 = np.array(src[0])
	ps2 = np.array(src[1])

	pd1 = np.array(dst[0])
	pd2 = np.array(dst[1])

	psv = (ps2 - ps1)
	pdv = (pd2 - pd1)

	psv /= sqrt(sum(psv**2))
	pdv /= sqrt(sum(pdv**2))

	angle = np.arccos(np.dot(psv, pdv))

	if(np.cross(psv, pdv) < 0):
		angle *= -1

	return angle

def scale(src, ctr, scale):
	scaled = (src - ctr) * scale + ctr
	return scaled

def rotate(src, ctr, rot):
	rot_mat = np.array([[cos(rot), -sin(rot)],
						[sin(rot), cos(rot)]])	
	
	rotated = np.array(src-ctr)

	for i in range(len(rotated)):
		rotated[i] = rot_mat.dot(rotated[i]) + ctr

	return rotated

def closest_dist(src, point):
	dist = 100000000
	for p in src:
		dist = min(dist, length([p, point]))
	return dist

def error(src, dst, show = False):
	error = 0
	for p in dst:
		if(show):
			print closest_dist(src, p)
		error += closest_dist(src, p)
	return error

def find_trs(src, dst):

	lsrc = range(len(src))
	ldst = range(len(dst))	

	slines = [[p,q] for p in [0] for q in lsrc if p != q]
	dlines = [[p,q] for p in ldst for q in ldst if p != q]

	err = 100000000

	t = None
	r = None
	s = None
	c = None

	for i in slines:
		for j in dlines:
			# print(src[i, :])
			# print(dst[i, :])
			scl = find_scale(src[i, :], dst[j, :])
			prj = scale(src, src[i[0]], scl)

			rot = find_rot(prj[i, :], dst[j,:])
			prj = rotate(prj, prj[i[0]], rot)

			trans = find_trans(prj[i,:], dst[j, :])
			# mid = np.float32(sum(dst))/len(dst)
			# avg = np.float32(sum(src))/len(src)
			# trans = mid - avg

			prj = prj + trans

			new_err = error(prj, dst)

			if(new_err < err and abs(rot)<pi/2):
				err = new_err
				t = trans
				r = rot
				s = scl
				c = i[0]



	# print('err', err)

	# if(err>40):
	# 	t = [0,0]
	# 	r = 0
	# 	s = 1

	return t, r, s, c

def publish_trs(trans, rot, scale):
	global trs_pub

	msg = FArray()
	msg.data = list(trans)
	msg.data.append(rot)
	msg.data.append(scale)

	trs_pub.publish(msg)


bridge = CvBridge()


defined = False
prod = []

counter = 0

lock = Lock()



def find_best(boxes, goal):
	err = 100000000

	lmin = 100000000

	t = None
	r = None
	s = None
	c = None
	index = 0

	for i in range(len(boxes)):
		points = boxes[i]
		trans, rot, scl, ctr = find_trs(points, goal)

		prj = np.array(points)
		prj = scale(prj, prj[ctr], scl)
		prj = rotate(prj, prj[ctr], rot)
		prj = prj + trans

		new_err = error(prj, goal)

		smean = np.mean(points, axis = 0)
		dmean = np.mean(goal, axis = 0)
		l = dmean - smean
		l = sqrt(sum(l**2))

		thr = 20

		if new_err < err-thr or (new_err<err+thr and l<lmin) :
			lmin = l
			t = trans
			r = rot
			s = scl
			c = ctr
			index = i
		
		err = min(new_err, err)

	return t, r, s, c, err, index


def handle_frame(msg):

	# global counter
	# counter+=1
	# if(counter%3>0):
	# 	return

	try:
		img = bridge.imgmsg_to_cv2(msg, "bgr8")
	except CvBridgeError as e:
		print(e)

	# img = cv2.medianBlur(img,5)
	# img = cv2.blur(img,(10,10))
	# img = cv2.GaussianBlur(img,(11,11),0)

	h = img.shape[0]
	w = img.shape[1]

	img2 = np.array(img)	
	dst = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


	
	boxes = detect_boxes(img2)

	points = []



	mid = np.array([h/2, w/2])

	gscale = 1.02

	# rh = gscale * 170/2
	# rw = gscale * 200/2

	rh = gscale * 170/2
	rw = gscale * 197/2

	# gscale = 4

	# rh = gscale * 15
	# rw = gscale * 50

	goal = [ mid + (rh, rw),
			 mid + (rh, -rw),
			 mid + (-rh, rw),
			 mid + (-rh, -rw) ]
					
	goal = np.array(goal)

	goal = np.array(sorted(goal, key = lambda x: x[0]))

	print(points)

	print(goal)

	trans, rot, scl, ctr, err, index = find_best(boxes, goal)

	# trans, rot, scl, ctr = find_trs(points, goal)

	if err>40 or scl>10:
		rot = 0.
		scl = 1.
	

	if len(boxes)==0:
		points = np.array(goal)
		return
	else:
		points = boxes[index]

	# points = np.array(sorted(points, key = lambda x: x[0]))

	prj = np.array(points)
	prj = scale(prj, prj[ctr], scl)
	prj = rotate(prj, prj[ctr], rot)
	prj = prj + trans

	# if rot==0:
	# avg = np.float32(sum(points))/len(points)
	# # trans = mid - avg

	# trans = mid - mean

	# print(mean, avg)

		
	publish_trs(trans, rot, scl)

	print trans, rot, scl


	points[:,[0, 1]] = points[:,[1, 0]]
	goal[:,[0, 1]] = goal[:,[1, 0]]
	prj[:,[0, 1]] = prj[:,[1, 0]]

	dst = cv2.cvtColor(dst,cv2.COLOR_GRAY2BGR)
	# dst = np.array(img2+dst)

	# masked = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)

	# dst = np.fmax(img, masked)


	cv2.polylines(dst, np.int32([points]), 1, (255,0,0))
	cv2.polylines(dst, np.int32([goal]), 1, (0,0,255))
	cv2.polylines(dst, np.int32([prj]), 1, (0,255,0))

	cpt1 = (np.int32(prj[ctr][0]), np.int32(prj[ctr][1]))
	cpt2 = (np.int32(points[ctr][0]), np.int32(points[ctr][1]))


	cv2.circle(dst, cpt1, 5, (0, 255, 0), -1) 
	cv2.circle(dst, cpt2, 5, (255, 0, 0), -1) 
	
	# cv2.circle(dst,(447,63), 63, (0,0,255), -1)
	# cv2.polylines(dst, goal, True, 0) 
	# dst = (dst/2 + img2/2)

	global img_pub

	msg_frame = CvBridge().cv2_to_imgmsg(dst)
	img_pub.publish(msg_frame)#, "RGB8")

	# cv2.imshow('detected circles',dst)
	# if cv2.waitKey(1) & 0xFF == ord('q'):
	# 	exit(0)
	# plt.draw()



	# if cv2.waitKey(1) & 0xFF == ord('q'):
	# 	exit(0)
	# cv2.destroyAllWindows()


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def main(argv=None):
	rospy.init_node('camera_ps3')
	rospy.Subscriber("/c230/image_raw", Image, handle_frame, queue_size = 1)
	
	global trs_pub, img_pub
	trs_pub = rospy.Publisher('/servoing_error', FArray, queue_size=1)
	img_pub = rospy.Publisher('/servoing_image', Image, queue_size=10)


	rospy.spin()

            
if __name__ == "__main__":
    sys.exit(main())


def filter_color(img):
	img2 = img
	hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

	ORANGE_MIN = np.array([   3.,   27.,  104.],np.uint8)
	ORANGE_MAX = np.array([  16.,   74.,  201.],np.uint8)

	mask = cv2.inRange(hsv, ORANGE_MIN, ORANGE_MAX)

	kernel = np.ones((5,5),np.uint8)
	mask = cv2.dilate(mask,kernel,iterations = 1)

	kernel = np.ones((5,5),np.uint8)
	mask = cv2.erode(mask,kernel,iterations = 1)


	res = cv2.bitwise_and(img, img, mask= mask)
	edge = cv2.Canny(mask,0,125)

	gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
	

	kernel = np.ones((9,9),np.uint8)
	gray = cv2.dilate(gray,kernel,iterations = 1)

	kernel = np.ones((9,9),np.uint8)
	gray = cv2.erode(gray,kernel,iterations = 1)


	dst = cv2.cornerHarris(gray,10,11,0.1)
	dst = cv2.dilate(dst,None)
	ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
	dst = np.uint8(dst)

	global defined
	if(not defined):
		global prod
		defined = True
		prod = np.array(list(product(np.arange(h), np.arange(w)))).reshape((h,w,2))

	mean = np.array([0,0])
	points = prod[gray>0]	
	mean = np.mean(points, axis=0)

	points = prod[dst>0]
	if(len(points)>0):
		points = cluster(points, .20)