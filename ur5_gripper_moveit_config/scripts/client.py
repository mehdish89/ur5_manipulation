#!/usr/bin/env python

import rospy

import socket               # Import socket module
import time
import pyqtgraph as pg
import pylab as plt
import ast
import numpy as np




import math
import numpy as np
import sys

from geometry_msgs.msg import WrenchStamped,PoseStamped

from std_msgs.msg import String

from tf import TransformerROS, TransformListener
from tf.transformations import *


# fh = open("script.txt", "r") 
pub = rospy.Publisher('/ur_driver/URScript', String, queue_size=1)

ft_pub = rospy.Publisher('/force_torque', WrenchStamped, queue_size=1)

# rospy.Subscriber('/ur_driver/URScript', String, callback)
rospy.init_node('ur_script_runner')	
rospy.sleep(0.5)


listener = TransformListener()
# tros = listener.TransformerROS()


def get_pose():
	if not listener.canTransform("base_link", "tool0", rospy.Time(0)):
		return np.array([0.] * 7)

	trans, rot = listener.lookupTransform('base', 'tool0', rospy.Time(0))

	pose = np.append(trans, rot)
	return pose




def forcetorqueToBase(ft):
	if not listener.canTransform("base_link", "tool0", rospy.Time(0)):
		return np.array([0.] * 6)

	f = ft[0:3]
	t = ft[3:6]

	
	nft = np.array([0.] * 6)
	
	trans, rot = listener.lookupTransform('base', 'tool0', rospy.Time(0))
	rotmat = listener.fromTranslationRotation(trans, rot)
	rotmat[0:3, 3] = 0.

	f_ = np.zeros(4)
	f_[0:3] = np.array(f)
	f_  = f_.reshape(4,1)
	f_ = np.dot(rotmat,f_)
	nft[0:3] = np.squeeze(f_[0:3])

	t_ = np.zeros(4)
	t_[0:3] = np.array(t)
	t_  = t_.reshape(4,1)
	t_ = np.dot(rotmat,t_)
	nft[3:6] = np.squeeze(t_[0:3])

	return nft

def rotation_around(twist, point = np.array([0.] * 3)):
	
	point = np.array(point, 'float32')
	twist = np.array(twist)

	f = twist[0:3]
	t = twist[3:6]

	# t = [1., 0., 0.]

	rotmat = euler_matrix(t[0], t[1], t[2], axes='sxyz')

	rotmat[0:3,3] = point

	backoff = np.ones(4)
	backoff[0:3] = -point

	# print(rotmat)
	# print(np.dot(rotmat, backoff))

	ntwist = twist[:]

	f_ = np.dot(rotmat, backoff)
	
	ntwist[0:3] = f_[0:3] + f

	return ntwist

def steadiness(data):

	return np.var(data)




s = socket.socket()         # Create a socket object
host = "192.168.1.10" 		# socket.gethostname() # Get local machine name
port = 63350                # Reserve a port for your service.

while not listener.canTransform("base_link", "tool0", rospy.Time(0)):
	print("please wait!!!")


plt.ion()

x_data = range(100)
y_data = np.arange(100) / 1. - 50

plts_force = []
plts_torque = []
plts_vel = []
plts = []
f, axarr = plt.subplots(4, sharex=True)


for i in range(3):
	graph1, = axarr[0].plot(x_data, y_data*2.)
	graph2, = axarr[1].plot(x_data, y_data/20.)
	graph3, = axarr[2].plot(x_data, y_data/100.)
	plts_force += [graph1]
	plts_torque += [graph2]
	plts_vel += [graph3]

plt_steady, = axarr[3].plot(x_data, y_data/10000.)

plts = plts_force + plts_torque

axarr[2].autoscale()


past = 100

trscale = 40

pose = get_pose()
pose[0:3] *= trscale
pose_data = np.tile(pose, (past,1))

print(pose_data)

s.connect((host, port))

t = 0


theta = 0.
rx = 0.
ry = 0.


while not rospy.is_shutdown():
	s.sendall("READ DATA")
	time.sleep(0.001)
	
	data = s.recv(1024)
	ft = np.array(ast.literal_eval(data))

	alpha_f = 0.3
	alpha_t = 0.3
	# alpha_v = 0.5

	std = 0.

	for i in range(6):
		y_data = plts[i].get_ydata()
		if i<3:
			y_new = y_data[-1] + alpha_f * (ft[i] - y_data[-1])
		else:
			y_new = y_data[-1] + alpha_t * (ft[i] - y_data[-1])

		ft[i] = y_new
		y_data = np.append(y_data[1:len(y_data)],y_new)
		plts[i].set_xdata(x_data)
		plts[i].set_ydata(y_data)

		if i<3:
			std += steadiness (y_data)
		else:
			std += steadiness (y_data * 20.) 


	y_data = plts_vel[0].get_ydata()
	y_new = rx
	y_data = np.append(y_data[1:len(y_data)],y_new)
	plts_vel[0].set_xdata(x_data)
	plts_vel[0].set_ydata(y_data)

	y_data = plts_vel[1].get_ydata()
	y_new = ry
	y_data = np.append(y_data[1:len(y_data)],y_new)
	plts_vel[1].set_xdata(x_data)
	plts_vel[1].set_ydata(y_data)

	y_data = plts_vel[2].get_ydata()
	y_new = (sum(abs(ft[0:3])) * 0  + sum(abs(ft[3:6])) * trscale)/200
	y_data = np.append(y_data[1:len(y_data)],y_new)
	plts_vel[2].set_xdata(x_data)
	plts_vel[2].set_ydata(y_data)



	std /= 6.	

	pose = get_pose() 

	pose[0:3] *= trscale
	pose_data = np.append(pose_data[1:], pose).reshape(past,7)
	std = sum(np.var(pose_data, axis = 0))

	# print(pose_data)

	# print(std)

	# exit(0)

	y_data = plt_steady.get_ydata()
	y_new = std
	y_data = np.append(y_data[1:len(y_data)],y_new)

	# axarr[2].relim()
	axarr[3].autoscale_view(True,True,True)

	plt_steady.set_xdata(x_data)
	plt_steady.set_ydata(y_data)

	plt.draw()
	

	fsc = 1./8000
	tsc = 1./15

	nft = ft[:]
	nft[2] += 0 #25

	nft[0:3] *= fsc
	nft[3:6] *= tsc

	dv = 0.1
	dtheta = 0.2

	print(theta)

	rx = dv * math.sin(theta)
	ry = dv * math.cos(theta)

	# ry = 0

	nft[3] += rx
	nft[4] += ry

	theta = (theta + dtheta) % (math.pi * 2)

	twist = rotation_around(nft, point = [0., 0., 0.20])
	twist = forcetorqueToBase(twist)

	

	# twist[2] += -25 * fsc



	pub.publish("speedl([{0},{1},{2},{3},{4},{5}], 10, 0.1)".format( twist[0],
																	 twist[1],
																	 twist[2], 
																	 twist[3], 
																	 twist[4], 
																	 twist[5] ))




	wrench = WrenchStamped()
	wrench.header.stamp = rospy.Time.now()
	wrench.header.frame_id = 'tool0'

	wrench.wrench.force.x = ft[0]
	wrench.wrench.force.y = ft[1]
	wrench.wrench.force.z = ft[2]

	wrench.wrench.torque.x = ft[3]
	wrench.wrench.torque.y = ft[4]
	wrench.wrench.torque.z = ft[5]

	ft_pub.publish(wrench)

	t += 1
s.close
