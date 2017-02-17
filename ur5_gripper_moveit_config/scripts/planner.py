#!/usr/bin/env python  

import sys
import numpy as np
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import tf


from robotiq_c_model_control.msg import CModel_robot_input as GripperState
from robotiq_c_model_control.msg import CModel_robot_output as GripperCommand


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



def plan_to(group, trans):
    pose_target = group.get_current_pose().pose

    # print(pose_target)

    pose_target.orientation.x = 0.00201201946573
    pose_target.orientation.y = 0.999877520803
    pose_target.orientation.z = 0.00806714448786
    pose_target.orientation.w = 0.0132595757704


    pose_target.position.x = trans[0]
    pose_target.position.y = trans[1]
    pose_target.position.z = trans[2]

    group.set_pose_target(pose_target)

    return group.plan()
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
    cmd.rSP = 30
    cmd.rFR = 20
    cmd.rPR = val
    grip_pub.publish(cmd)


def check_forward():
    global a
    # print "============ Ready..."
    # str = sys.stdin.readline()

    # print(str)
    # if(str=='q\n'):
    #     sys.exit()


if __name__ == '__main__':

    global grip_pub

    print "============ Starting tutorial setup"
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_python',
                    anonymous=True)

    rospy.Subscriber("/CModelRobotInput", GripperState, state_callback)
    grip_pub = rospy.Publisher('/CModelRobotOutput', GripperCommand, queue_size=10)

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

    group.set_planner_id("RRTkConfigDefault")

    listener = tf.TransformListener()

    start = group.get_current_joint_values()

    steps = 0.

    while not rospy.is_shutdown():
        gripper(0)

        sys.stdin.readline()
        
        print "============ Generating plan"
        

        ## [0.758, 0.329, 0.391
        # state = robot.get_current_state()        

        # pose_target.orientation.w = 1.0

        try:
            (trans,rot) = listener.lookupTransform('/base_link', 'cube', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("Exception caught")
            continue


        # pose_target.position.x = 0.758
        # pose_target.position.y = 0.329
        # pose_target.position.z = 0.391

        

        # while(is_grasped):
            # rate.sleep()


        trans = np.array(trans)
        trans[2] += 0.20

        result = plan_to(group, trans)
        if(result.joint_trajectory.header.frame_id == ''):
            continue

        print "============ Waiting while RVIZ displays plan..."
        rospy.sleep(1)

        check_forward()
        

        # Uncomment below line when working with a real robot
        group.go(wait=True)   


        check_forward()  

        trans[2] -= 0.03

        result = plan_to(group, trans)
        if(result.joint_trajectory.header.frame_id == ''):
            continue

        check_forward()

        # Uncomment below line when working with a real robot
        group.go(wait=True) 

        check_forward()  

        gripper(255)

        for i in range(8):
            if(is_grasped):
                break
            rospy.sleep(0.5)

        

        check_forward()  

        trans[2] += 0.03

        result = plan_to(group, trans)
        if(result.joint_trajectory.header.frame_id == ''):
            continue

        check_forward()

        # Uncomment below line when working with a real robot
        group.go(wait=True)


        if(not is_grasped):
            continue

        check_forward()

        # group.set_joint_value_target(start)  
        # group.plan()

        plan_to(group, np.array((0.66, 0.32, 0.40 + steps)))

        check_forward()

        group.go(wait=True) 

        check_forward()

        plan_to(group, np.array((0.66, 0.32, 0.345 + steps)))

        check_forward()

        group.go(wait=True) 

        check_forward() 

        gripper(0)
        steps += 0.01

        rate.sleep()
