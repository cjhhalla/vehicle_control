#! /usr/bin/env python3
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
    (37.386011, 126.652558),
    (37.386027, 126.652530),
    (37.386046, 126.652500),
    (37.386066, 126.652467),
    (37.386087, 126.652433),
    (37.386109, 126.652396),
    (37.386132, 126.652358),
    (37.386156, 126.652317),
    (37.386186, 126.652270),
    ## 1
    (37.386988, 126.648747),
    (37.386944, 126.648705),
    (37.386901, 126.648664),
    (37.386859, 126.648623),
    (37.386818, 126.648584),
    (37.386776, 126.648543),
    ## 2
    (37.385401, 126.648788),
    (37.385360, 126.648856),
    (37.385322, 126.648918),
    (37.385288, 126.648973),
    (37.385254, 126.649029),
    ## 3
    (37.384455, 126.650347),
    (37.384404, 126.650428),
    (37.384354, 126.650510),
    (37.384303, 126.650592),
    ## 4
    (37.383819, 126.651387),
    (37.383778, 126.651455),
    (37.383737, 126.651523),
    (37.383716, 126.651557),
    ## 5
    (37.383053, 126.652649),
    (37.383014, 126.652713),
    (37.382981, 126.652767),
    (37.382954, 126.652811),
    ## 6
    (37.382702, 126.653218),
    (37.382683, 126.653249),
    (37.382659, 126.653287),
    (37.382637, 126.653323),
    (37.382621, 126.653351),
    (37.382611, 126.653376),
    (37.382606, 126.653398),
    (37.382606, 126.653430),
    (37.382614, 126.653463),
    (37.382632, 126.653498),
    (37.382658, 126.653532),
    ## 7
    (37.383053, 126.653905),
    (37.383119, 126.653969),
    (37.383170, 126.654019),
    (37.383241, 126.654088),
    ## 8
    (37.383612, 126.654445),
    (37.383697, 126.654525),
    (37.383768, 126.654591),
    (37.383836, 126.654656)
]

waypoint_sections = {
    0: waypoints[0:9],    # #0 (0~8)
    1: waypoints[9:15],   # #1 (9~14)
    2: waypoints[15:20],  # #2 (15~19)
    3: waypoints[20:24],  # #3 (20~23)
    4: waypoints[24:28],  # #4 (24~27)
    5: waypoints[28:32],  # #5 (28~31)
    6: waypoints[32:44],  # #6 (32~43)
    7: waypoints[44:48],  # #7 (44~47)
    8: waypoints[48:58],  # #8 (48~57)
}




class PurePursuit:
    def __init__(self):
        self.L = 3
        self.k = 0.15  # 0.1~1
        self.Lfc = 6  
        self.alpha = 1.5
    def euc_distance(self, pt1, pt2):
        return norm([pt2[0] - pt1[0], pt2[1] - pt1[1]])

    def global_to_local(self, target_point, position, yaw):
        dx = target_point[0] - position[0]
        dy = target_point[1] - position[1]

        x_local = dx * cos(-yaw) - dy * sin(-yaw)
        y_local = dx * sin(-yaw) + dy * cos(-yaw) 

        return x_local, y_local 

    def vel_global_to_local(self, target_point, position, yaw):
        dx = target_point[0] - position[0]
        dy = target_point[1] - position[1]

        x_local = dx * cos(-yaw) - dy * sin(-yaw)
        y_local = dx * sin(-yaw) + dy * cos(-yaw)

        return x_local, y_local 

    def run(self, vEgo, target_point, position, yaw, sEgo):
        # gamma = math.tan(radians(abs(sEgo)))/self.L
        # lfc = self.Lfc / (1+self.alpha*gamma)
        lfd = self.Lfc + self.k * vEgo
        lfd = np.clip(lfd, 5, 10)
        rospy.loginfo(f"Lfd: {lfd}")
        x_local , y_local = self.vel_global_to_local(target_point,position, yaw)
        diff = np.sqrt(x_local**2 + y_local**2)
        
        if diff > 0:
            dis = np.linalg.norm(diff)
            if dis >= lfd:
                theta = atan2(y_local, x_local)
                steering_angle = atan2(2 * self.L * sin(theta), lfd)
                return degrees(steering_angle), target_point
        return 0.0, target_point  

    def run_global(self, vEgo, target_point, position, yaw, sEgo):
        # gamma = math.tan(radians(abs(sEgo)))/self.L
        # lfc = self.Lfc/(1+self.alpha*gamma)
        lfd = self.Lfc + self.k * vEgo
        lfd = np.clip(lfd, 5, 10)
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
        self.steer_sub = rospy.Subscriber('/vehicle/steering_angle', Float32, self.steer_callback)
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
        
        self.curr_steer = 0


    def steer_callback(self,msg):
        self.curr_steer = msg.data

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
        self.current_point.y = msg.pose.position.y +0.04  

        # self.point_history_x.append(self.current_point.x)
        # self.point_history_y.append(self.current_point.y)

        # self.current_point.x = sum(self.point_history_x) / len(self.point_history_x)
        # self.current_point.y = sum(self.point_history_y) / len(self.point_history_y)

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
            last_waypoint = waypoints[-1]

            for i,waypoint in enumerate(waypoints):
                distance = geodesic(curr_position, waypoint).meters
                if i== len(waypoint) -1 and distance <=1:
                    return -1
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
                target_steering, target_position = self.pure_pursuit.run_global(self.curr_v, waypoint, position, yaw,self.curr_steer)
                throttle = self.pid.run(target_position, position)
                throttle *= 0.7
                throttle = np.clip(throttle,0,6.5)
            
                # accel = throttle / 150
                if waypoint_sec == 6:
                    light = Float32()
                    light.data = 1
                    self.light_pub.publish(light)   
                    steer = target_steering * self.steer_ratio
                else:
                    steer = target_steering * self.steer_ratio / 10
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
                target_steering, target_position = self.pure_pursuit.run(self.curr_v, waypoint, position, yaw,self.curr_steer)
                current_speed = self.curr_v
                throttle = self.pid.run(target_position, position)
                throttle *= 0.7
                throttle = np.clip(throttle,0,6.5)
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
