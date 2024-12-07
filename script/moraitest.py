import rospy
from std_msgs.msg import Float32, String, Header, Bool
from morai_msgs.msg import CtrlCmd, EgoVehicleStatus
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Vector3
import math,csv
import os
class MoraiVehicleControlNode:
    def __init__(self):
        rospy.init_node('vehicle_control_node', anonymous=True)

        self.steer = 0.0
        self.accel = 0.0
        self.brake = 0.0
        self.current_v = 0.0
        self.prev_v = 0.0
        self.gear = None
        self.accel_pose = 0
        self.start_position = None
        self.initialized = False  
        self.path = Path()
        self.path.header = Header()
        self.path.header.frame_id = "map"  
        self.path.poses = []
        self.lon_en = None
        self.pa_en = None
        ##
        self.morai_v = 0.0
        self.morai_st = 0.0
        ##

        rospy.Subscriber('/target_actuator', Vector3, self.actu_cb)
        rospy.Subscriber('/Ego_topic', EgoVehicleStatus, self.ego_cb,queue_size = 20)
        
        self.pub_path = rospy.Publisher('/ego_path',Path,queue_size = 10)
        self.pub_morai = rospy.Publisher('/ctrl_cmd', CtrlCmd, queue_size=10)

        self.steering_ratio = 700
        self.accel_ratio = 0.1
        self.control_loop()

    def actu_cb(self,msg):
        self.accel = msg.x
        self.steer = msg.y

    def ego_cb(self, msg):
        if not self.initialized:
            self.start_position = msg.position  
            self.initialized = True  
            rospy.loginfo("Initial position set to: x = {}, y = {}, z = {}".format(
                self.start_position.x, self.start_position.y, self.start_position.z))

        current_position = msg.position

        relative_position = Vector3(
            current_position.x - self.start_position.x,
            current_position.y - self.start_position.y,
            current_position.z - self.start_position.z
        )

        self.morai_st = msg.wheel_angle
        self.morai_v = math.sqrt(msg.velocity.x**2 + msg.velocity.y**2)
        new_pose = PoseStamped()
        new_pose.header = Header()
        new_pose.header.stamp = rospy.Time.now()
        new_pose.header.frame_id = "map"
        
        new_pose.pose.position.x = relative_position.x
        new_pose.pose.position.y = relative_position.y
        new_pose.pose.position.z = relative_position.z

        new_pose.pose.orientation.x = 0.0
        new_pose.pose.orientation.y = 0.0
        new_pose.pose.orientation.z = 0.0
        new_pose.pose.orientation.w = 1.0

        self.path.poses.append(new_pose)

        self.pub_path.publish(self.path)

    def control_loop(self):
        rate = rospy.Rate(20)  
        while not rospy.is_shutdown():
            moraicmd = CtrlCmd()

            # steering_ = self.steer / self.steering_ratio
            moraicmd.accel = self.accel
            moraicmd.steering = self.steer * 0.0005 * 2.7

            # print("Str: ",self.steer)
            # print("Vel: ",self.current_v)
            print(self.morai_st)
            print(self.morai_v)
            self.pub_morai.publish(moraicmd)
            rate.sleep()
            moraicmd.accel = 0.005
            self.pub_morai.publish(moraicmd)
            rate.sleep()

if __name__ == '__main__':
    try:
        MoraiVehicleControlNode()
    except rospy.ROSInterruptException:
        pass
