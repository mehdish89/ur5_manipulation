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

	#####


	ty1 = -a2*cos(q1)*cos(q2) + a3*sin(q2)*sin(q3)*cos(q1) - a3*cos(q1)*cos(q2)*cos(q3) - d4*sin(q1) + 0.5*d5*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1)) - 0.5*d5*(-sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1)) + 0.5*d6*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*d6*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) - d6*sin(q1)*cos(q5)

	ty2 = a2*sin(q1)*sin(q2) + a3*sin(q1)*sin(q2)*cos(q3) + a3*sin(q1)*sin(q3)*cos(q2) + 0.5*d5*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1)) - 0.5*d5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1)) + 0.5*d6*(-sin(q1)*sin(q2 + q3 + q4) - cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*d6*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5)

	ty3 = a3*sin(q1)*sin(q2)*cos(q3) + a3*sin(q1)*sin(q3)*cos(q2) + 0.5*d5*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1)) - 0.5*d5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1)) + 0.5*d6*(-sin(q1)*sin(q2 + q3 + q4) - cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*d6*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5)

	ty4 = 0.5*d5*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1)) - 0.5*d5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1)) + 0.5*d6*(-sin(q1)*sin(q2 + q3 + q4) - cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*d6*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5)

	ty5 = 0.5*d6*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*cos(q5) + 0.5*d6*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*cos(q5) - d6*sin(q5)*cos(q1)

	ty6 = 0

	#####

	tz1 = 0

	tz2 = a2*cos(q2) + a3*(-sin(q2)*sin(q3) + cos(q2)*cos(q3)) + d5*sin(q2 + q3 + q4) + 0.5*d6*(-sin(q5)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)) - 0.5*d6*(sin(q5)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5))

	tz3 = a3*(-sin(q2)*sin(q3) + cos(q2)*cos(q3)) + d5*sin(q2 + q3 + q4) + 0.5*d6*(-sin(q5)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)) - 0.5*d6*(sin(q5)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5))

	tz4 = d5*sin(q2 + q3 + q4) + 0.5*d6*(-sin(q5)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)) - 0.5*d6*(sin(q5)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5))
	
	tz5 = 0.5*d6*(-sin(q5)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)) - 0.5*d6*(-sin(q5)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q5))

	tz6 = 0

#################################	
	
	rx1 = 0

	rx2 = 1.0*sin(q5)*cos(q2 + q3 + q4)/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2)

	rx3 = 1.0*sin(q5)*cos(q2 + q3 + q4)/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2)

	rx4 = 1.0*sin(q5)*cos(q2 + q3 + q4)/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2)

	rx5 = 1.0*sin(q2 + q3 + q4)*cos(q5)/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2)

	rx6 = 0

	#####

	ry1 = 0
	ry2 = ((1.0*sin(q6)*sin(q2 + q3 + q4) - cos(q5)*cos(q6)*cos(q2 + q3 + q4))/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2) + 1.0*(-1.0*sin(q6)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)*cos(q6))*sin(q5)**2*sin(q2 + q3 + q4)*cos(q2 + q3 + q4)/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(3/2))*(sin(q6)*sin(q2 + q3 + q4)*cos(q5) - 1.0*cos(q6)*cos(q2 + q3 + q4))/(((-1.0*sin(q6)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)*cos(q6))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1) + (sin(q6)*sin(q2 + q3 + q4)*cos(q5) - 1.0*cos(q6)*cos(q2 + q3 + q4))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1))*(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2)) - ((sin(q6)*cos(q5)*cos(q2 + q3 + q4) + 1.0*sin(q2 + q3 + q4)*cos(q6))/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2) + 1.0*(sin(q6)*sin(q2 + q3 + q4)*cos(q5) - 1.0*cos(q6)*cos(q2 + q3 + q4))*sin(q5)**2*sin(q2 + q3 + q4)*cos(q2 + q3 + q4)/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(3/2))*(-1.0*sin(q6)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)*cos(q6))/(((-1.0*sin(q6)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)*cos(q6))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1) + (sin(q6)*sin(q2 + q3 + q4)*cos(q5) - 1.0*cos(q6)*cos(q2 + q3 + q4))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1))*(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2))

	ry3 = ((1.0*sin(q6)*sin(q2 + q3 + q4) - cos(q5)*cos(q6)*cos(q2 + q3 + q4))/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2) + 1.0*(-1.0*sin(q6)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)*cos(q6))*sin(q5)**2*sin(q2 + q3 + q4)*cos(q2 + q3 + q4)/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(3/2))*(sin(q6)*sin(q2 + q3 + q4)*cos(q5) - 1.0*cos(q6)*cos(q2 + q3 + q4))/(((-1.0*sin(q6)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)*cos(q6))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1) + (sin(q6)*sin(q2 + q3 + q4)*cos(q5) - 1.0*cos(q6)*cos(q2 + q3 + q4))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1))*(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2)) - ((sin(q6)*cos(q5)*cos(q2 + q3 + q4) + 1.0*sin(q2 + q3 + q4)*cos(q6))/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2) + 1.0*(sin(q6)*sin(q2 + q3 + q4)*cos(q5) - 1.0*cos(q6)*cos(q2 + q3 + q4))*sin(q5)**2*sin(q2 + q3 + q4)*cos(q2 + q3 + q4)/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(3/2))*(-1.0*sin(q6)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)*cos(q6))/(((-1.0*sin(q6)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)*cos(q6))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1) + (sin(q6)*sin(q2 + q3 + q4)*cos(q5) - 1.0*cos(q6)*cos(q2 + q3 + q4))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1))*(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2))

	ry4 = ((1.0*sin(q6)*sin(q2 + q3 + q4) - cos(q5)*cos(q6)*cos(q2 + q3 + q4))/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2) + 1.0*(-1.0*sin(q6)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)*cos(q6))*sin(q5)**2*sin(q2 + q3 + q4)*cos(q2 + q3 + q4)/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(3/2))*(sin(q6)*sin(q2 + q3 + q4)*cos(q5) - 1.0*cos(q6)*cos(q2 + q3 + q4))/(((-1.0*sin(q6)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)*cos(q6))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1) + (sin(q6)*sin(q2 + q3 + q4)*cos(q5) - 1.0*cos(q6)*cos(q2 + q3 + q4))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1))*(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2)) - ((sin(q6)*cos(q5)*cos(q2 + q3 + q4) + 1.0*sin(q2 + q3 + q4)*cos(q6))/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2) + 1.0*(sin(q6)*sin(q2 + q3 + q4)*cos(q5) - 1.0*cos(q6)*cos(q2 + q3 + q4))*sin(q5)**2*sin(q2 + q3 + q4)*cos(q2 + q3 + q4)/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(3/2))*(-1.0*sin(q6)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)*cos(q6))/(((-1.0*sin(q6)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)*cos(q6))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1) + (sin(q6)*sin(q2 + q3 + q4)*cos(q5) - 1.0*cos(q6)*cos(q2 + q3 + q4))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1))*(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2))

	ry5 = -(-1.0*sin(q6)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)*cos(q6))*(-sin(q5)*sin(q6)*sin(q2 + q3 + q4)/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2) + 1.0*(sin(q6)*sin(q2 + q3 + q4)*cos(q5) - 1.0*cos(q6)*cos(q2 + q3 + q4))*sin(q5)*sin(q2 + q3 + q4)**2*cos(q5)/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(3/2))/(((-1.0*sin(q6)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)*cos(q6))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1) + (sin(q6)*sin(q2 + q3 + q4)*cos(q5) - 1.0*cos(q6)*cos(q2 + q3 + q4))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1))*(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2)) + (sin(q6)*sin(q2 + q3 + q4)*cos(q5) - 1.0*cos(q6)*cos(q2 + q3 + q4))*(sin(q5)*sin(q2 + q3 + q4)*cos(q6)/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2) + 1.0*(-1.0*sin(q6)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)*cos(q6))*sin(q5)*sin(q2 + q3 + q4)**2*cos(q5)/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(3/2))/(((-1.0*sin(q6)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)*cos(q6))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1) + (sin(q6)*sin(q2 + q3 + q4)*cos(q5) - 1.0*cos(q6)*cos(q2 + q3 + q4))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1))*(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2))

	ry6 = -(-1.0*sin(q6)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)*cos(q6))*(1.0*sin(q6)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q5)*cos(q6))/(((-1.0*sin(q6)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)*cos(q6))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1) + (sin(q6)*sin(q2 + q3 + q4)*cos(q5) - 1.0*cos(q6)*cos(q2 + q3 + q4))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1))*(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)) + (sin(q6)*sin(q2 + q3 + q4)*cos(q5) - 1.0*cos(q6)*cos(q2 + q3 + q4))**2/(((-1.0*sin(q6)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q5)*cos(q6))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1) + (sin(q6)*sin(q2 + q3 + q4)*cos(q5) - 1.0*cos(q6)*cos(q2 + q3 + q4))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1))*(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1))

	rz1 = (0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) - sin(q1)*cos(q5))**2/(((0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) - sin(q1)*cos(q5))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1) + (0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*sin(q5) + cos(q1)*cos(q5))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1))*(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)) - (0.5*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(-sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*sin(q5) - cos(q1)*cos(q5))*(0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*sin(q5) + cos(q1)*cos(q5))/(((0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) - sin(q1)*cos(q5))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1) + (0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*sin(q5) + cos(q1)*cos(q5))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1))*(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1))

	rz2 = ((0.5*(-sin(q1)*sin(q2 + q3 + q4) - cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5))/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2) + 1.0*(0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*sin(q5) + cos(q1)*cos(q5))*sin(q5)**2*sin(q2 + q3 + q4)*cos(q2 + q3 + q4)/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(3/2))*(0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) - sin(q1)*cos(q5))/(((0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) - sin(q1)*cos(q5))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1) + (0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*sin(q5) + cos(q1)*cos(q5))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1))*(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2)) - ((0.5*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5))/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2) + 1.0*(0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) - sin(q1)*cos(q5))*sin(q5)**2*sin(q2 + q3 + q4)*cos(q2 + q3 + q4)/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(3/2))*(0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*sin(q5) + cos(q1)*cos(q5))/(((0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) - sin(q1)*cos(q5))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1) + (0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*sin(q5) + cos(q1)*cos(q5))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1))*(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2))

	rz3 = ((0.5*(-sin(q1)*sin(q2 + q3 + q4) - cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5))/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2) + 1.0*(0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*sin(q5) + cos(q1)*cos(q5))*sin(q5)**2*sin(q2 + q3 + q4)*cos(q2 + q3 + q4)/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(3/2))*(0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) - sin(q1)*cos(q5))/(((0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) - sin(q1)*cos(q5))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1) + (0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*sin(q5) + cos(q1)*cos(q5))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1))*(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2)) - ((0.5*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5))/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2) + 1.0*(0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) - sin(q1)*cos(q5))*sin(q5)**2*sin(q2 + q3 + q4)*cos(q2 + q3 + q4)/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(3/2))*(0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*sin(q5) + cos(q1)*cos(q5))/(((0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) - sin(q1)*cos(q5))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1) + (0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*sin(q5) + cos(q1)*cos(q5))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1))*(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2))

	rz4 = ((0.5*(-sin(q1)*sin(q2 + q3 + q4) - cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5))/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2) + 1.0*(0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*sin(q5) + cos(q1)*cos(q5))*sin(q5)**2*sin(q2 + q3 + q4)*cos(q2 + q3 + q4)/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(3/2))*(0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) - sin(q1)*cos(q5))/(((0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) - sin(q1)*cos(q5))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1) + (0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*sin(q5) + cos(q1)*cos(q5))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1))*(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2)) - ((0.5*(-sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5))/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2) + 1.0*(0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) - sin(q1)*cos(q5))*sin(q5)**2*sin(q2 + q3 + q4)*cos(q2 + q3 + q4)/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(3/2))*(0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*sin(q5) + cos(q1)*cos(q5))/(((0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) - sin(q1)*cos(q5))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1) + (0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*sin(q5) + cos(q1)*cos(q5))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1))*(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2))

	rz5 = -((0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*cos(q5) + 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*cos(q5) + sin(q1)*sin(q5))/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2) + 1.0*(0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) - sin(q1)*cos(q5))*sin(q5)*sin(q2 + q3 + q4)**2*cos(q5)/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(3/2))*(0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*sin(q5) + cos(q1)*cos(q5))/(((0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) - sin(q1)*cos(q5))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1) + (0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*sin(q5) + cos(q1)*cos(q5))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1))*(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2)) + ((0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*cos(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*cos(q5) - sin(q5)*cos(q1))/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2) + 1.0*(0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*sin(q5) + cos(q1)*cos(q5))*sin(q5)*sin(q2 + q3 + q4)**2*cos(q5)/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(3/2))*(0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) - sin(q1)*cos(q5))/(((0.5*(-sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) + 0.5*(sin(q1)*sin(q2 + q3 + q4) + cos(q1)*cos(q2 + q3 + q4))*sin(q5) - sin(q1)*cos(q5))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1) + (0.5*(sin(q1)*cos(q2 + q3 + q4) - sin(q2 + q3 + q4)*cos(q1))*sin(q5) + 0.5*(sin(q1)*cos(q2 + q3 + q4) + sin(q2 + q3 + q4)*cos(q1))*sin(q5) + cos(q1)*cos(q5))**2/(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1))*(-1.0*sin(q5)**2*sin(q2 + q3 + q4)**2 + 1)**(1/2))

	rz6 = 0
	
#################################
	return np.array([[tx1, tx2, tx3, tx4, tx5, tx6],
					[ty1, ty2, ty3, ty4, ty5, ty6],
					[tz1, tz2, tz3, tz4, tz5, tz6],						
					[rx1, rx2, rx3, rx4, rx5, rx6],
					[ry1, ry2, ry3, ry4, ry5, ry6],
					[rz1, rz2, rz3, rz4, rz5, rz6]])
					
















