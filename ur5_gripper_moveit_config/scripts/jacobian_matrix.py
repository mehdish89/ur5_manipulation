from math import *
import numpy as np

def jacobian(q):

	d1 =  0.089159
	a2 = -0.42500
	a3 = -0.39225
	d4 =  0.10915
	d5 =  0.09465
	d6 =  0.0823

	#print(q)

	q1 = q[0]
	q2 = q[1]
	q3 = q[2]
	q4 = q[3]
	q5 = q[4]
	q6 = q[5]


	tx1 = a2*sin(q1)*cos(q2) - a3*sin(q1)*sin(q2)*sin(q3) + a3*sin(q1)*cos(q2)*cos(q3) - d4*cos(q1) - 0.5*d5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4)) + 0.5*d5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4)) + 0.5*d6*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*d6*(-sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*sin(q5) - d6*cos(q1)*cos(q5) 

	tx2 = a2*sin(q2)*cos(q1) + a3*sin(q2)*cos(q1)*cos(q3) + a3*sin(q3)*cos(q1)*cos(q2) + 0.5*d5*(-sin(q1)*sin(q2 + q3 + q4) - cos(q1)*cos(q2 + q3 + q4)) - 0.5*d5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4)) + 0.5*d6*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*d6*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) 

	tx3 = a3*sin(q2)*cos(q1)*cos(q3) + a3*sin(q3)*cos(q1)*cos(q2) + 0.5*d5*(-sin(q1)*sin(q2 + q3 + q4) - cos(q1)*cos(q2 + q3 + q4)) - 0.5*d5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4)) + 0.5*d6*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*d6*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) 

	tx4 = 0.5*d5*(-sin(q1)*sin(q2 + q3 + q4) - cos(q1)*cos(q2 + q3 + q4)) - 0.5*d5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4)) + 0.5*d6*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*d6*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) 

	tx5 = 0.5*d6*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*cos(q5) + 0.5*d6*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*cos(q5) + d6*sin(q1)*sin(q5) 

	tx6 = 0 

	ty1 = -a2*cos(q1)*cos(q2) + a3*sin(q2)*sin(q3)*cos(q1) - a3*cos(q1)*cos(q2)*cos(q3) - d4*sin(q1) + 0.5*d5*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1)) - 0.5*d5*(-sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1)) + 0.5*d6*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*d6*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) - d6*sin(q1)*cos(q5) 

	ty2 = a2*sin(q1)*sin(q2) + a3*sin(q1)*sin(q2)*cos(q3) + a3*sin(q1)*sin(q3)*cos(q2) + 0.5*d5*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1)) - 0.5*d5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1)) + 0.5*d6*(-sin(q1)*sin(q2 + q3 + q4) - cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*d6*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) 

	ty3 = a3*sin(q1)*sin(q2)*cos(q3) + a3*sin(q1)*sin(q3)*cos(q2) + 0.5*d5*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1)) - 0.5*d5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1)) + 0.5*d6*(-sin(q1)*sin(q2 + q3 + q4) - cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*d6*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) 

	ty4 = 0.5*d5*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1)) - 0.5*d5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1)) + 0.5*d6*(-sin(q1)*sin(q2 + q3 + q4) - cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*d6*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) 

	ty5 = 0.5*d6*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*cos(q5) + 0.5*d6*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*cos(q5) - d6*sin(q5)*cos(q1) 

	ty6 = 0 

	tz1 = 0 

	tz2 = a2*cos(q2) + a3*(-sin(q2)*sin(q3) + cos(q2)*cos(q3)) + d5*sin(q2 + q3 + q4) + 0.5*d6*(-sin(q5)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)) - 0.5*d6*(sin(q5)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)) 

	tz3 = a3*(-sin(q2)*sin(q3) + cos(q2)*cos(q3)) + d5*sin(q2 + q3 + q4) + 0.5*d6*(-sin(q5)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)) - 0.5*d6*(sin(q5)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)) 

	tz4 = d5*sin(q2 + q3 + q4) + 0.5*d6*(-sin(q5)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)) - 0.5*d6*(sin(q5)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)) 

	tz5 = 0.5*d6*(-sin(q5)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)) - 0.5*d6*(-sin(q5)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q5)) 

	tz6 = 0 

	nx1 = 0.5*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(-sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*sin(q5) - cos(q1)*cos(q5) 

	nx2 = 0.5*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) 

	nx3 = 0.5*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) 

	nx4 = 0.5*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) 

	nx5 = 0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*cos(q5) + 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*cos(q5) + sin(q1)*sin(q5) 

	nx6 = 0 

	ox1 = (0.5*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*cos(q5) + 0.5*(-sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*cos(q5) + sin(q5)*cos(q1))*cos(q6) + 1.0*sin(q1)*sin(q6)*sin(q2 + q3 + q4) 

	ox2 = (0.5*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*cos(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*cos(q5))*cos(q6) - 1.0*sin(q6)*cos(q1)*cos(q2 + q3 + q4) 

	ox3 = (0.5*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*cos(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*cos(q5))*cos(q6) - 1.0*sin(q6)*cos(q1)*cos(q2 + q3 + q4) 

	ox4 = (0.5*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*cos(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*cos(q5))*cos(q6) - 1.0*sin(q6)*cos(q1)*cos(q2 + q3 + q4) 

	ox5 = (-0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) - 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) + sin(q1)*cos(q5))*cos(q6) 

	ox6 = -(0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*cos(q5) + 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*cos(q5) + sin(q1)*sin(q5))*sin(q6) - 1.0*sin(q2 + q3 + q4)*cos(q1)*cos(q6) 

	ax1 = -(0.5*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*cos(q5) + 0.5*(-sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*cos(q5) + sin(q5)*cos(q1))*sin(q6) + 1.0*sin(q1)*sin(q2 + q3 + q4)*cos(q6) 

	ax2 = -(0.5*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*cos(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*cos(q5))*sin(q6) - 1.0*cos(q1)*cos(q6)*cos(q2 + q3 + q4) 

	ax3 = -(0.5*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*cos(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*cos(q5))*sin(q6) - 1.0*cos(q1)*cos(q6)*cos(q2 + q3 + q4) 

	ax4 = -(0.5*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*cos(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*cos(q5))*sin(q6) - 1.0*cos(q1)*cos(q6)*cos(q2 + q3 + q4) 

	ax5 = -(-0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) - 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) + sin(q1)*cos(q5))*sin(q6) 

	ax6 = -(0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*cos(q5) + 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*cos(q5) + sin(q1)*sin(q5))*cos(q6) + 1.0*sin(q6)*sin(q2 + q3 + q4)*cos(q1) 

	ny1 = 0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) - sin(q1)*cos(q5) 

	ny2 = 0.5*(-sin(q1)*sin(q2 + q3 + q4) - cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) 

	ny3 = 0.5*(-sin(q1)*sin(q2 + q3 + q4) - cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) 

	ny4 = 0.5*(-sin(q1)*sin(q2 + q3 + q4) - cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) 

	ny5 = 0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*cos(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*cos(q5) - sin(q5)*cos(q1) 

	ny6 = 0 

	oy1 = (0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*cos(q5) + 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*cos(q5) + sin(q1)*sin(q5))*cos(q6) - 1.0*sin(q6)*sin(q2 + q3 + q4)*cos(q1) 

	oy2 = (0.5*(-sin(q1)*sin(q2 + q3 + q4) - cos(q1)*cos(q2 + q3 + q4))*cos(q5) + 0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*cos(q5))*cos(q6) - 1.0*sin(q1)*sin(q6)*cos(q2 + q3 + q4) 

	oy3 = (0.5*(-sin(q1)*sin(q2 + q3 + q4) - cos(q1)*cos(q2 + q3 + q4))*cos(q5) + 0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*cos(q5))*cos(q6) - 1.0*sin(q1)*sin(q6)*cos(q2 + q3 + q4) 

	oy4 = (0.5*(-sin(q1)*sin(q2 + q3 + q4) - cos(q1)*cos(q2 + q3 + q4))*cos(q5) + 0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*cos(q5))*cos(q6) - 1.0*sin(q1)*sin(q6)*cos(q2 + q3 + q4) 

	oy5 = (-0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) - 0.5*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*sin(q5) - cos(q1)*cos(q5))*cos(q6) 

	oy6 = -(0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*cos(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*cos(q5) - sin(q5)*cos(q1))*sin(q6) - 1.0*sin(q1)*sin(q2 + q3 + q4)*cos(q6) 

	ay1 = -(0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*cos(q5) + 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*cos(q5) + sin(q1)*sin(q5))*sin(q6) - 1.0*sin(q2 + q3 + q4)*cos(q1)*cos(q6) 

	ay2 = -(0.5*(-sin(q1)*sin(q2 + q3 + q4) - cos(q1)*cos(q2 + q3 + q4))*cos(q5) + 0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*cos(q5))*sin(q6) - 1.0*sin(q1)*cos(q6)*cos(q2 + q3 + q4) 

	ay3 = -(0.5*(-sin(q1)*sin(q2 + q3 + q4) - cos(q1)*cos(q2 + q3 + q4))*cos(q5) + 0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*cos(q5))*sin(q6) - 1.0*sin(q1)*cos(q6)*cos(q2 + q3 + q4) 

	ay4 = -(0.5*(-sin(q1)*sin(q2 + q3 + q4) - cos(q1)*cos(q2 + q3 + q4))*cos(q5) + 0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*cos(q5))*sin(q6) - 1.0*sin(q1)*cos(q6)*cos(q2 + q3 + q4) 

	ay5 = -(-0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) - 0.5*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*sin(q5) - cos(q1)*cos(q5))*sin(q6) 

	ay6 = -(0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*cos(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*cos(q5) - sin(q5)*cos(q1))*cos(q6) + 1.0*sin(q1)*sin(q6)*sin(q2 + q3 + q4) 

	nz1 = 0 

	nz2 = -1.0*sin(q5)*cos(q2 + q3 + q4) 

	nz3 = -1.0*sin(q5)*cos(q2 + q3 + q4) 

	nz4 = -1.0*sin(q5)*cos(q2 + q3 + q4) 

	nz5 = -1.0*sin(q2 + q3 + q4)*cos(q5) 

	nz6 = 0 

	oz1 = 0 

	oz2 = 1.0*sin(q6)*sin(q2 + q3 + q4) - cos(q5)*cos(q6)*cos(q2 + q3 + q4) 

	oz3 = 1.0*sin(q6)*sin(q2 + q3 + q4) - cos(q5)*cos(q6)*cos(q2 + q3 + q4) 

	oz4 = 1.0*sin(q6)*sin(q2 + q3 + q4) - cos(q5)*cos(q6)*cos(q2 + q3 + q4) 

	oz5 = sin(q5)*sin(q2 + q3 + q4)*cos(q6) 

	oz6 = sin(q6)*sin(q2 + q3 + q4)*cos(q5) - 1.0*cos(q6)*cos(q2 + q3 + q4) 

	az1 = 0 

	az2 = sin(q6)*cos(q5)*cos(q2 + q3 + q4) + 1.0*sin(q2 + q3 + q4)*cos(q6) 

	az3 = sin(q6)*cos(q5)*cos(q2 + q3 + q4) + 1.0*sin(q2 + q3 + q4)*cos(q6) 

	az4 = sin(q6)*cos(q5)*cos(q2 + q3 + q4) + 1.0*sin(q2 + q3 + q4)*cos(q6) 

	az5 = -sin(q5)*sin(q6)*sin(q2 + q3 + q4) 

	az6 = 1.0*sin(q6)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q5)*cos(q6)
	
	#################################
	return np.array([[nx1, nx2, nx3, nx4, nx5, nx6],
					 [ox1, ox2, ox3, ox4, ox5, ox6],
					 [ax1, ax2, ax3, ax4, ax5, ax6],
					 [tx1, tx2, tx3, tx4, tx5, tx6],
					 [ny1, ny2, ny3, ny4, ny5, ny6],
					 [oy1, oy2, oy3, oy4, oy5, oy6],
					 [ay1, ay2, ay3, ay4, ay5, ay6],
					 [ty1, ty2, ty3, ty4, ty5, ty6],
					 [nz1, nz2, nz3, nz4, nz5, nz6],
					 [oz1, oz2, oz3, oz4, oz5, oz6],
					 [az1, az2, az3, az4, az5, az6],
					 [tz1, tz2, tz3, tz4, tz5, tz6],
					 [  0,   0,   0,   0,   0,   0],
					 [  0,   0,   0,   0,   0,   0],
					 [  0,   0,   0,   0,   0,   0],
					 [  0,   0,   0,   0,   0,   0]])
					
















