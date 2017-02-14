#!/usr/bin/env python  

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg


if __name__ == '__main__':
    print "============ Starting tutorial setup"
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_python',
                    anonymous=True)

    robot = moveit_commander.RobotCommander()

    scene = moveit_commander.PlanningSceneInterface()

    group = moveit_commander.MoveGroupCommander("manipulator")

    display_trajectory_publisher = rospy.Publisher(
                                    '/move_group/display_planned_path',
                                    moveit_msgs.msg.DisplayTrajectory)

    print "============ Waiting for RVIZ..."
    rospy.sleep(10)
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

    while not rospy.is_shutdown():
        sys.stdin.readline()
        
        print "============ Generating plan"
        pose_target = geometry_msgs.msg.Pose()

        ## [0.758, 0.329, 0.391

        pose_target.orientation.w = 1.0
        pose_target.position.x = 0.758
        pose_target.position.y = 0.329
        pose_target.position.z = 0.391
        group.set_pose_target(pose_target)

        plan1 = group.plan()

        print "============ Waiting while RVIZ displays plan..."
        rospy.sleep(5)

        print "============ Ready..."

        rate.sleep()
