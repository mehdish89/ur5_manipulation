#!/usr/bin/env python  

import sys
import numpy as np
import numpy 
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import tf

import math

import threading

from jacobian_matrix import jacobian
from forward import forward

from math import *

from std_msgs.msg import Float32MultiArray as FArray

from robotiq_c_model_control.msg import CModel_robot_input as GripperState
from robotiq_c_model_control.msg import CModel_robot_output as GripperCommand

from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Vector3Stamped


import time
import roslib; roslib.load_manifest('ur_driver')
import actionlib
# from control_msgs.msg import *
from trajectory_msgs.msg import *

from pid import PID

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']


"""
    Gripper State Flags:

        uint8 gACT      # Activated
        uint8 gGTO      # Operating
        uint8 gSTA      # 
        uint8 gOBJ      # Object Grasp Status
        uint8 gFLT      # Fault Status
        uint8 gPR       # Goal Position
        uint8 gPO       # Current Position
        uint8 gCU       # 
"""
def state_callback(msg):

    global is_grasped
    global is_active
    is_grasped = bool(msg.gOBJ == 2)
    is_active = bool(msg.gACT and msg.gGTO)



def plan_to(group, trans, rot = None):
    pose_target = group.get_current_pose().pose

    # print(pose_target)

    if rot is not None:
        pose_target.orientation.x = rot[0]
        pose_target.orientation.y = rot[1]
        pose_target.orientation.z = rot[2]
        pose_target.orientation.w = rot[3]


    pose_target.position.x = trans[0]
    pose_target.position.y = trans[1]
    pose_target.position.z = trans[2]

    group.set_pose_target(pose_target)

    return group.plan()

def plan_diff(group, dtrans = np.array([0.] * 3), drot = np.array([0.] * 4)):
    pose_target = group.get_current_pose().pose

    # print(pose_target)

    trans = np.array([0.] * 3)
    rot = np.array([0.] * 4)

    rot[0] = pose_target.orientation.x + drot[0]
    rot[1] = pose_target.orientation.y + drot[1]
    rot[2] = pose_target.orientation.z + drot[2]
    rot[3] = pose_target.orientation.w + drot[3]


    trans[0] = pose_target.position.x + dtrans[0]
    trans[1] = pose_target.position.y + dtrans[1]
    trans[2] = pose_target.position.z + dtrans[2]

    return plan_to(group, trans, rot)

"""
RESTART
rACT: 0
rGTO: 0
rATR: 0
rPR: 0
rSP: 0
rFR: 0
---
ACTIVATE
rACT: 1
rGTO: 1
rATR: 0
rPR: 0
rSP: 255
rFR: 150
"""

def reset():
    global grip_pub
    cmd = GripperCommand()
    grip_pub.publish(cmd)


def activate():
    global grip_pub
    reset()    

    cmd = GripperCommand()
    cmd.rACT = 1
    cmd.rGTO = 1
    cmd.rSP = 255
    cmd.rFR = 150
    grip_pub.publish(cmd)



def gripper(val):

    global is_active
    if(not is_active):
        activate()

    cmd = GripperCommand()
    cmd.rACT = 1
    cmd.rGTO = 1
    # cmd.rSP = 30
    cmd.rSP = 70
    cmd.rFR = 20

    # cmd.rFR = 100

    cmd.rPR = val
    grip_pub.publish(cmd)


def check_forward():
    global a
    print "============ Ready..."
    # str = sys.stdin.readline()

    # print(str)
    # if(str=='q\n'):
    #     sys.exit()




def servoing_callback(msg):
    global gtrans, grot, gscale

    gtrans = [msg.data[0], msg.data[1]]
    grot = msg.data[2]
    gscale = msg.data[3]


# def create_from_axis_angle(xx, yy, zz, a)
#     # Here we calculate the sin( theta / 2) once for optimization
#     factor = sin( a / 2.0 );

#     # Calculate the x, y and z of the quaternion
#     x = xx * factor;
#     y = yy * factor;
#     z = zz * factor;

#     # Calcualte the w value by cos( theta / 2 )
#     w = cos( a / 2.0 );

#     return Quaternion(x, y, z, w).normalize();


def to_quaternion(qmsg):
    q = np.zeros((4, ), dtype=np.float64)

    q[0] = qmsg.x
    q[1] = qmsg.y
    q[2] = qmsg.z
    q[3] = qmsg.w

    return q

def from_quaternion(q):
    qmsg = Quaternion()

    qmsg.x = q[0]
    qmsg.y = q[1]
    qmsg.z = q[2]
    qmsg.w = q[3]

    return qmsg


def fromRotationMatrixToJointValues(Q, dR):
    J = jacobian(Q)
    # print('J = ', J)
    r = dR.reshape(-1)
    # print('r = ', r)
    Ji = np.linalg.pinv(J)
    # print('Ji = ', Ji)
    dQ = np.dot(Ji, r)
    # print('dQ = ', dQ)
    return dQ


def stop_speed(acc = 5):

    print("STOP called!!!")

    global speed_pub

    traj = JointTrajectory()
    traj.joint_names = JOINT_NAMES
    point = JointTrajectoryPoint(positions=[0.]*6, velocities=[0.]*6, accelerations=[acc], time_from_start=rospy.Duration(0))
    traj.points.append(point)

    speed_pub.publish(traj)

def plan_speed(dRT, dt = 1, dQ0 = np.array([0.] * 6), acc = 5):
    traj = JointTrajectory()
    traj.joint_names = JOINT_NAMES

    Q = group.get_current_joint_values()
    dQ = fromRotationMatrixToJointValues(Q, dRT) + dQ0

    point = JointTrajectoryPoint(positions=Q, velocities=dQ/dt, accelerations=[acc], time_from_start=rospy.Duration(0))
    traj.points.append(point)

    return traj

timer = threading.Timer(10, stop_speed)

def set_speed(vx, vy, vz, vth = 0, acc = 1, t = 0.5):
    global timer
    timer.cancel()


    Rc = listener.fromTranslationRotation((0, 0, 0), (0, 0, 0, 0))
    Rn = listener.fromTranslationRotation((vx, vy, vz), (0, 0, 0, 0))
    vR = Rn - Rc

    vQ0 = np.array([0.]*6)
    vQ0[-1] = vth

    traj = plan_speed(vR, dQ0 = vQ0, dt = 1, acc = acc)

    timer = threading.Timer(t, stop_speed)
    timer.start()

    speed_pub.publish(traj)



def plan_traj(dRT, Q0 = np.array([0.] * 6), dQ0 = np.array([0.] * 6), N = 10, dt = 0.05):
    g = FollowJointTrajectoryGoal()
    g.trajectory = JointTrajectory()
    g.trajectory.joint_names = JOINT_NAMES

    Q = group.get_current_joint_values()
    Qi = np.array(Q)

    print('dQ0 = ', dQ0)

    # start_point = JointTrajectoryPoint(positions=Qi, velocities=[0]*6, time_from_start=rospy.Duration(0))
    # g.trajectory.points.append(start_point)

    M=1
    dQ = np.array(dQ0)

    for i in range(N+1):
        point = JointTrajectoryPoint(positions=Qi, velocities=dQ/dt, time_from_start=rospy.Duration(dt*i))
        g.trajectory.points.append(point)
        dQ = fromRotationMatrixToJointValues(Qi, dRT * M / N ) + (Q0 * M/N)
        Qi = np.array(Qi + dQ)
    
    # g.trajectory.points[-1].velocities = [0.5] * 6

    return g, dQ

# def exec_traj(traj):

def move(g):
    # client.cancel_goal()
    # rospy.sleep(0.05)
    client.send_goal(g)
    # rospy.sleep(0.4)   
    # return 
    try:
        print('trying')
        client.wait_for_result()
        print('done')
    except KeyboardInterrupt:
        client.cancel_goal()
        print('canceled')
        raise


def alog(x):
    if(x>=0):
        return math.log(1+x)
    else:
        return -math.log(1-x)



def servo(group, dtrans = np.array([0.,0.]), drot = 0., dscale = 0.):   
    global gtrans, grot, gscale
    global listener


    dx = 1000000
    dy = 1000000
    dQ = np.array([0.] * 6)    

    tf_count = 0

    x_PID = PID(P=1., I=0.000, D=0.0000)
    y_PID = PID(P=1., I=0.000, D=0.0000)
    z_PID = PID(P=2, I=0.0001, D=0.0000)
    th_PID = PID(P=0.8, I=0.000, D=0.0000)
    
    x_PID.setPoint(0.)
    y_PID.setPoint(0.)
    z_PID.setPoint(0.)
    th_PID.setPoint(0.)

    trans = gtrans - dtrans  
    rot = grot - drot
    scale = gscale - dscale

    while abs(scale-1)>0.05 or abs(rot)>0.005 or abs(dx)>0.0005 or abs(dy)>0.0005:

        print("c", tf_count)

        if(tf_count%5==0):
            print("update")
            
            eps = 0.00000001

            dth = (rot) / 2
            dx = -trans[1]/15000
            dy = -trans[0]/15000
            dz = -(scale-1+eps) / 50

            if abs(dz)<0.003:
                dz = abs(dz)*0.003/dz
            pose_target = group.get_current_pose().pose


            v = Vector3Stamped()
            v.header.stamp = listener.getLatestCommonTime("base_link", "tool0")
            v.header.frame_id = "tool0"
            v.vector.x = dx
            v.vector.y = dy

            gv = listener.transformVector3("base_link", v)


            # print(v)
            # print(gv)

            

            dx = gv.vector.x
            dy = gv.vector.y

        tf_count+=1

        dth = (rot) / 2
        dz = -(scale-1) / 50
        # dx = dx * 2
        # dy = dy * 2
        # dz = dz * 2

        # vx = 1 * alog(dx * 2) 
        # vy = 1 * alog(dy * 2) 
        # vz = 0.75 * alog(dz) * 0

        dth2 = th_PID.update(rot)


        print("dth=", dth)
        print("dth2=", dth2)



        # vx = 1.5 * alog(dx) * 0
        # vy = 1.5 * alog(dy) * 0
        # vz = 1 * alog(dz) * 0

        # vth = 3. * alog(dth) 

        vx = -x_PID.update(dx)
        vy = -y_PID.update(dy)
        vz = -z_PID.update(dz)
        vth = -dth2


        
        print(vth)
        # set_speed(2*vx, 2*vy, 2*vz, vth = vth, acc = 0.5)
        # rospy.sleep(0.0002)        
        set_speed(2*vx, 2*vy, 2*vz, vth = vth, acc = 10)
        # set_speed(0, 0, 0, vth = dth, acc = 1)
        # print(dth)

        rospy.sleep(0.01)

        trans = gtrans - dtrans  
        rot = grot - drot
        scale = gscale - dscale


        # qm = to_quaternion(pose_target.orientation)
        # qr = tf.transformations.quaternion_about_axis(dth, (0,1,0))
        # q = tf.transformations.quaternion_multiply(qm, qr)

        # pose_target.orientation = from_quaternion(q)


        # Rc = listener.fromTranslationRotation((0, 0, 0), (0, 0, 0, 0))



        # Q = group.get_current_joint_values()
        

        # Rn = listener.fromTranslationRotation((dx, dy, dz), (0, 0, 0, 0))

        # # dR = listener.fromTranslationRotation((dx, dy, dz), (0, 0, 0, 1))
        
        # dR = Rn - Rc
        # print(Q)
        # print('dR = ', dR)
        # print(dx, dy, dz, dth, scale)

        # # dQ = fromRotationMatrixToJointValues(Q, dR)

        # Q0 = np.array([0.] * 6)
        # Q0[-1] = dth

        # goal, dQ = plan_vel(dR, Q0, dQ)

        # move(goal)

        # print pln

        # Qn = Q + dQ

        # Qn = np.array(pln[-1])

        # Qn[5] += dth

        # group.set_joint_value_target(Qn)
        # group.set_pose_target(pose_target)
        # group.plan()
        # print(pose_target)
        

        # check_forward()
        # group.go(wait=True) 
        # rospy.sleep(rospy.Duration(.3))
    stop_speed()
    print("done")




if __name__ == '__main__':

    global grip_pub, speed_pub, listener

    print "============ Starting tutorial setup"
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_python',
                    anonymous=True)

    rospy.Subscriber("/servoing_error", FArray, servoing_callback)

    rospy.Subscriber("/CModelRobotInput", GripperState, state_callback)
    
    grip_pub = rospy.Publisher('/CModelRobotOutput', GripperCommand, queue_size=10)

    speed_pub = rospy.Publisher('/ur_driver/joint_speed', JointTrajectory, queue_size=0)

    robot = moveit_commander.RobotCommander()

    scene = moveit_commander.PlanningSceneInterface()

    group = moveit_commander.MoveGroupCommander("manipulator")

    display_trajectory_publisher = rospy.Publisher(
                                    '/move_group/display_planned_path',
                                    moveit_msgs.msg.DisplayTrajectory)

    print "============ Waiting for RVIZ..."
    rospy.sleep(1)
    print "============ Starting tutorial "

    print "============ Reference frame: %s" % group.get_planning_frame()

    print "============ Reference frame: %s" % group.get_end_effector_link()

    print "============ Robot Groups:"
    print robot.get_group_names()

    print "============ Printing robot state"
    print robot.get_current_state()
    print "============"

    rate = rospy.Rate(50)

    # client = actionlib.SimpleActionClient('follow_joint_trajectory', FollowJointTrajectoryAction)
    # print "Waiting for server..."
    # client.wait_for_server()
    # print "Connected to server"

    # group.set_planner_id("RRTkConfigDefault")

    listener = tf.TransformListener()

    start = group.get_current_joint_values()

    steps = 0.

    sys.stdin.readline()

    # for i in range(10):
    #     set_speed(0, -0.1, 0, acc = 10)
    #     rospy.sleep(0.1)

    # stop_speed()

    # servo(group, dscale = 0.)
    # servo(group, dscale = -0.35)
    # move1(group)

    # set_speed(0., 0.1, 0., acc = 10)

    # rospy.sleep(0.2)

    # set_speed(0., 0.3, 0., acc = 10)

    # rospy.sleep(0.2)

    # stop_speed()

    # set_speed(0., -0.1, 0., acc = 10)

    # rospy.sleep(0.2)

    # set_speed(0., -0.2, 0., acc = 10)

    # rospy.sleep(0.2)

    # traj = plan_speed(dR, dt = 0.05)

    # speed_pub.publish(traj)

    # goal, dQ = plan_vel(dR)

    # move(goal)

    # goal, dQ = plan_vel(dR, np.array([0]*6), dQ)

    # move(goal)

    # print(goal)

    rot0 = np.array([-0.650476083998,
                    0.759441816647,
                    0.00827562458826,
                    0.00777851607368])

    # sys.exit(0)

    while not rospy.is_shutdown():
        gripper(0)

        sys.stdin.readline()
        
        print "============ Generating plan"
        

        ## [0.758, 0.329, 0.391
        # state = robot.get_current_state()        

        # pose_target.orientation.w = 1.0

        try:
            (gtrans,grot) = listener.lookupTransform('/base_link', 'cube', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("Exception caught")
            continue


        # pose_target.position.x = 0.758
        # pose_target.position.y = 0.329
        # pose_target.position.z = 0.391

        

        # while(is_grasped):
            # rate.sleep()


        gtrans = np.array(gtrans)
        gtrans[2] += 0.25

        result = plan_to(group, gtrans)
        if(result.joint_trajectory.header.frame_id == ''):
            continue

        print "============ Waiting while RVIZ displays plan..."
        rospy.sleep(1)

        check_forward()
        

        # Uncomment below line when working with a real robot
        group.go(wait=True)   
        check_forward()  

        servo(group)
        check_forward()  

        # servo(group, dscale = -0.35)
        # check_forward()  

        # plan_diff(group, dtrans = np.array([0, 0, -0.02]))
        # check_forward()
    
        # group.go(wait=True)   
        # check_forward()  

        gripper(255)
        for i in range(8):
            if(is_grasped):
                break
            rospy.sleep(0.5)    
        check_forward()  

        plan_diff(group, dtrans = np.array([0, 0, 0.05]))
        check_forward()

        group.go(wait=True)
        if(not is_grasped):
            continue
        check_forward()

        plan_to(group, np.array((0.66 + steps, 0.32, 0.40  + steps)), rot0)
        check_forward()

        group.go(wait=True) 
        check_forward()

        plan_to(group, np.array((0.66 + steps, 0.32, 0.342 - 0.002 + steps)), rot0)
        check_forward()

        group.go(wait=True) 
        check_forward() 

        gripper(0)
        steps += 0.01

        rate.sleep()
