#!/usr/bin/env python
# license removed for brevity
import rospy
from sensor_msgs.msg import JointState
from robotiq_c_model_control.msg import CModel_robot_input as GripperState

"""
	Gripper State Flags:

	    uint8 gACT 		# Activated
		uint8 gGTO		# Operating
		uint8 gSTA 		# 
		uint8 gOBJ 		# Object Grasp Status
		uint8 gFLT 		# Fault Status
		uint8 gPR 		# Goal Position
		uint8 gPO 		# Current Position
		uint8 gCU		# 
"""
def state_callback(msg):

	global pub

	fwd = JointState()
	if(msg.gACT and msg.gGTO):
		fwd.name = ['gripper_robotiq_85_left_knuckle_joint', 
					'gripper_robotiq_85_right_knuckle_joint',
					'gripper_robotiq_85_left_inner_knuckle_joint', 
					'gripper_robotiq_85_right_inner_knuckle_joint', 
					'gripper_robotiq_85_left_finger_tip_joint', 
					'gripper_robotiq_85_right_finger_tip_joint']

		pos = 0.85*(float(msg.gPO-3)/(230-3))

		fwd.position = [pos, pos, pos, pos, -pos, -pos]

		# print pos

		pub.publish(fwd)



if __name__ == '__main__':
	global pub
	rospy.init_node('robotiq_c2_joint_state_publisher', anonymous=True)
	pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
	rospy.Subscriber("/CModelRobotInput", GripperState, state_callback)

	rospy.spin()