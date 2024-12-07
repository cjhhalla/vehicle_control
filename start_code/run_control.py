import rospy
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker
import numpy as np
from math import radians, degrees, atan2, cos, sin, sqrt
from numpy.linalg import norm
# from gps_common.msg import GPSFix
from nav_msgs.msg import Odometry, Path
import tf
import math
from geopy.distance import geodesic
from geometry_msgs.msg import Vector3
from novatel_oem7_msgs.msg import BESTGNSSPOS
from collections import deque


waypoints = [
    ## 0
    (37.386009, 126.652560),
    (37.386079, 126.652448),
    (37.386158, 126.652316),
    (37.386220, 126.652214),
    ## 1
    (37.386984, 126.648744),
    (37.386888, 126.648652),
    (37.386739, 126.648509),
    ## 2
    (37.385387, 126.648815),
    (37.385303, 126.648952),
    (37.385217, 126.649091),
    ## 3
    (37.384463, 126.650332),
    (37.384385, 126.650459),
    (37.384304, 126.650591),
    ## 4
    (37.383018, 126.652703),
    (37.382930, 126.652844),
    ## 5
    (37.382712, 126.653205),
    (37.382649, 126.653308),
    (37.382617, 126.653372),
    (37.382613, 126.653459),
    (37.382606, 126.653401),
    (37.382649, 126.653520),
    (37.382690, 126.653564),
    (37.382720, 126.653590),
    ## 6
    (37.383019, 126.653874),
    (37.383143, 126.653992),
    (37.383281, 126.654125),
    ## 7
    (37.383611, 126.654440),
    (37.383746, 126.654570),
    (37.383867, 126.654686),
]

waypoint_sections = {
    0: waypoints[0:4],   # #0
    1: waypoints[4:7],   # #1
    2: waypoints[7:10],  # #2
    3: waypoints[10:13], # #3
    4: waypoints[13:15], # #4
    5: waypoints[15:23], # #5
    6: waypoints[23:26], # #6
    7: waypoints[26:29], # #7
}



class PurePursuit:
    def __init__(self):
        self.L = 3
        self.k = 0.1  # 0.1~1
        self.Lfc = 6  
    def euc_distance(self, pt1, pt2):
        return norm([pt2[0] - pt1[0], pt2[1] - pt1[1]])

    def global_to_local(self, target_point, position, yaw):
        dx = target_point[0] - position[0]
        dy = target_point[1] - position[1]

        x_local = dx * cos(-yaw) - dy * sin(-yaw)
        y_local = dx * sin(-yaw) + dy * cos(-yaw)

        return x_local, y_local 

    def run(self, vEgo, target_point, position, yaw):
        lfd = self.Lfc + self.k * vEgo
        lfd = np.clip(lfd, 3, 60)
        rospy.loginfo(f"Lfd: {lfd}")
        x_local , y_local = self.global_to_local(target_point,position, yaw)
        diff = np.sqrt(x_local**2 + y_local**2)
        
        if diff > 0:
            dis = np.linalg.norm(diff)
            if dis >= lfd:
                theta = atan2(y_local, x_local)
                steering_angle = atan2(2 * self.L * sin(theta), lfd)
                return degrees(steering_angle), target_point
        return 0.0, target_point  

class PID:
    def __init__(self, kp, ki, kd, dt=0.05):
        self.K_P = kp
        self.K_I = ki
        self.K_D = kd
        self.pre_error = 0.0
        self.integral_error = 0.0
        self.dt = dt

    def run(self, target, current):
        error = sqrt((target[0] - current[0])**2 + (target[1] - current[1])**2)
        derivative_error = (error - self.pre_error) / self.dt
        self.integral_error += error * self.dt
        self.integral_error = np.clip(self.integral_error, -5, 5)

        pid = self.K_P * error + self.K_I * self.integral_error + self.K_D * derivative_error
        pid = np.clip(pid, -100, 100)
        self.pre_error = error
        return pid

class Start:
    def __init__(self):
        self.pure_pursuit = PurePursuit()
        self.pid = PID(kp=1.0, ki=0.1, kd=0.01)
        self.local_path_sub = rospy.Subscriber('/ego_waypoint', Marker, self.local_cb)
        self.global_gps_sub = rospy.Subscriber('/current_global_waypoint', Marker, self.global_cb)
        self.pose_sub = rospy.Subscriber('/ego_pos', PoseStamped, self.pose_cb)
        # self.vel_sub = rospy.Subscriber('/vehicle/curr_v', Float32, self.vel_cb)
        self.gps_pos_sub = rospy.Subscriber('/gps_ego_pose',Point,self.gps_pos_cb)
        # self.odom_sub = rospy.Subscriber('/novatel/oem7/odom', Odometry, self.odom_cb,queue_size=20)
        self.start_flag_sub = rospy.Subscriber('/start_flag',Bool, self.flag_cb)
        self.point_sub = rospy.Subscriber('/last_target_point', Marker, self.point_callback)
        self.gps_sub = rospy.Subscriber('/novatel/oem7/bestgnsspos', BESTGNSSPOS, self.bestgps_cb)
        self.yaw_sub = rospy.Subscriber('/vehicle/yaw_rate_sensor',Float32,self.yaw_cb)
        self.rl_sub = rospy.Subscriber('/vehicle/velocity_RL', Float32, self.rl_callback)
        self.rr_sub = rospy.Subscriber('/vehicle/velocity_RR', Float32, self.rr_callback)
        self.actuator_pub = rospy.Publisher('/target_actuator', Vector3, queue_size=10)
        self.light_pub = rospy.Publisher('vehicle/left_signal', Float32, queue_size =10)
        self.global_odom_pub = rospy.Publisher('/global_odom_frame_point',Marker,queue_size=10)

        self.curr_v = 0
        self.pose = PoseStamped()
        self.global_waypoints_x = None
        self.global_waypoints_y = None
        self.local_waypoint = Point()
        self.steer_ratio = 12
        self.current_point = None
        self.curr_lat = None
        self.curr_lon = None
        self.current_waypoint_idx = 0

        self.global_pose_x = None
        self.global_pose_y = None
        self.yaw_rate = None
        self.is_start = False

        self.moving_average_window = 1  
        self.point_history_x = deque(maxlen=self.moving_average_window)
        self.point_history_y = deque(maxlen=self.moving_average_window)

        self.rl_v = 0
        self.rr_v = 0

    # def odom_cb(self,msg):
    #     self.yaw = msg.twist.twist.angular.z
    def rl_callback(self,msg):
        self.rl_v = msg.data

    def rr_callback(self,msg):
        self.rr_v = msg.data


    def get_yaw_from_pose(self, pose):
        orientation_q = pose.pose.orientation
        quaternion = (
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w,
        )
        _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)
        return yaw

    def bestgps_cb(self,msg):
        self.curr_lat = msg.lat
        self.curr_lon = msg.lon

    def flag_cb(self,msg):
        self.is_start = msg.data

    def yaw_cb(self,msg):
        self.yaw_rate = radians(msg.data)

    def gps_pos_cb(self,msg):
        self.global_pose_x = msg.x
        self.global_pose_y = msg.y

    def point_callback(self,msg):
        self.current_point = Point()
        self.current_point.x = msg.pose.position.x 
        self.current_point.y = msg.pose.position.y    

        self.point_history_x.append(self.current_point.x)
        self.point_history_y.append(self.current_point.y)

        self.current_point.x = sum(self.point_history_x) / len(self.point_history_x)
        self.current_point.y = sum(self.point_history_y) / len(self.point_history_y)

    # def vel_cb(self, msg):
    #     self.curr_v = msg.data

    def local_cb(self, msg):
        self.local_waypoint = Point()
        x_ = msg.pose.position.x
        y_ = msg.pose.position.y
        self.local_waypoint.x = x_
        self.local_waypoint.y = y_

    def global_cb(self, msg):
        self.global_waypoints_x = msg.pose.position.x
        self.global_waypoints_y = msg.pose.position.y


    def pose_cb(self, msg):
        self.pose = msg
        self.yaw = self.get_yaw_from_pose(self.pose)

    def generate_global_waypoints(self):
        if not self.global_waypoints or not self.pose:
            return None

        curr_x = self.pose.pose.position.x
        curr_y = self.pose.pose.position.y

        closest_idx = None
        closest_dist = float('inf')
        for i, waypoint in enumerate(self.global_waypoints):
            dist = np.sqrt((curr_x - waypoint[0])**2 + (curr_y - waypoint[1])**2)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i

        if closest_idx is None:
            rospy.logwarn("No closest waypoint found.")
            return None

        global_waypoints = self.global_waypoints[closest_idx]
        return global_waypoints

    def is_near_corner(self, lat, lon, threshold=10):
        current_pos = (lat, lon)
        for idx, corner in enumerate(CORNERS):
            distance = geodesic(current_pos, corner).meters
            if distance < threshold:
                return idx
        return None

    def find_waypoint_section(self, curr_lat, curr_lon, sections, threshold=10) :
        curr_position = (curr_lat, curr_lon)

        for section_id, waypoints in sections.items():
            for waypoint in waypoints:
                distance = geodesic(curr_position, waypoint).meters
                if distance <= threshold:
                    return section_id  

        return -1

    def global_to_local(self, target_point, position, yaw):
        dx = target_point[0] - position[0]
        dy = target_point[1] - position[1]

        x_local = dx * cos(-yaw) - dy * sin(-yaw)
        y_local = dx * sin(-yaw) + dy * cos(-yaw)

        return x_local, y_local 


    def pub_global_waypoint(self,x,y):
        marker = Marker()
        marker.header.frame_id = "map" 
        marker.header.stamp = rospy.Time.now()
        marker.ns = "waypoints"
        marker.id = 0 
        marker.type = Marker.SPHERE 
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 5 
        marker.scale.y = 5
        marker.scale.z = 5

        marker.color.a = 1.0  
        marker.color.r = 1.0 
        marker.color.g = 1.0 
        marker.color.b = 1.0

        self.global_odom_pub.publish(marker)


    def run_control_loop(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            
            if not self.is_start:
                rospy.loginfo("Not setting yet...")
                rate.sleep()
                continue

            waypoint_sec = self.find_waypoint_section(self.curr_lat, self.curr_lon, waypoint_sections)
            self.curr_v = (self.rl_v + self.rr_v)/7.2
            # rospy.loginfo(f"current velocity: {self.curr_v}")
            if  self.global_pose_x is not None and self.global_pose_y is not None and self.global_waypoints_x is not None and self.global_waypoints_y is not None and waypoint_sec != -1:
                waypoint_x = self.global_waypoints_x
                waypoint_y = self.global_waypoints_y

                waypoint = (waypoint_x,waypoint_y)
                position = (self.global_pose_x, self.global_pose_y)

                yaw = self.yaw

                x_local , y_local = self.global_to_local(waypoint,position, yaw)
                self.pub_global_waypoint(x_local, y_local)

                rospy.loginfo(f"current velocity: {self.curr_v}")
                target_steering, target_position = self.pure_pursuit.run(self.curr_v, waypoint, position, yaw)
                throttle = self.pid.run(target_position, position)
                throttle *= 0.9
                throttle = np.clip(throttle,0,6)
            
                # accel = throttle / 150
                if waypoint_sec == 5:
                    light = Float32()
                    light.data = 1
                    self.light_pub.publish(light)   
                steer = target_steering * self.steer_ratio / 2.5
                rospy.loginfo("Using Global Waypoint")
                rospy.loginfo(f"accel value: {accel}")
                rospy.loginfo(f"steer value: {steer}")
                temp = Vector3()
                temp.x = accel
                temp.y = steer
                temp.z = 0

                self.actuator_pub.publish(temp)
                
                self.global_waypoints_x = None
                self.global_waypoints_y = None

            elif self.current_point is not None and waypoint_sec == -1:
                
                way_x = self.current_point.x
                way_y = self.current_point.y 
                position = (0, 0)
                waypoint = (way_x, way_y)
                
                yaw = self.yaw_rate
                rospy.loginfo(f"current velocity: {self.curr_v}")
                target_steering, target_position = self.pure_pursuit.run(self.curr_v, waypoint, position, yaw)
                current_speed = self.curr_v
                throttle = self.pid.run(target_position, position)
                throttle *= 0.9
                throttle = np.clip(throttle,0,6)
                accel = throttle
                steer = target_steering * self.steer_ratio
                temp = Vector3()
                temp.x = accel
                temp.y = steer
                temp.z = 1
                rospy.loginfo("Using Local Waypoint")
                rospy.loginfo(f"accel value: {accel}")
                rospy.loginfo(f"steer value: {steer}")
                self.actuator_pub.publish(temp)
                self.current_point = None

            else:
                # rospy.logwarn("No waypoints or current_point available. Skipping loop.")
                rate.sleep()
                continue

            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('control_node')
    start = Start()

    try:
        start.run_control_loop()
    except rospy.ROSInterruptException:
        pass
