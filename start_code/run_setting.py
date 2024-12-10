#! /usr/bin/env python3
import rospy
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Point, PoseStamped
from morai_msgs.msg import CtrlCmd , EgoVehicleStatus
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from math import radians, degrees, atan2, cos, sin, sqrt
from numpy.linalg import norm
# from gps_common.msg import GPSFix
from nav_msgs.msg import Odometry, Path
import tf
import math
from geopy.distance import geodesic
from novatel_oem7_msgs.msg import BESTGNSSPOS

class PathTracker:
    def __init__(self, waypoints):
        self.waypoint = waypoints
        self.marker_pub = rospy.Publisher('/ego_waypoint', Marker, queue_size=10)
        self.marker_gps_pub = rospy.Publisher('/ego_gps_waypoint', Marker, queue_size=10)
        self.point_gps_pub = rospy.Publisher('/point_globak_waypoint', Point, queue_size=10)
        self.marker_all_gps_pub = rospy.Publisher('/ego_all_gps_waypoint', Marker, queue_size=10)
        self.pose_pub = rospy.Publisher('/ego_pos', PoseStamped, queue_size=10)
        self.pub_path = rospy.Publisher('/ego_path_visualization', Path, queue_size=10)
        self.ctrl_cmd_pub = rospy.Publisher('/ctrl_cmd', CtrlCmd, queue_size=1)
        self.gps_pos = rospy.Publisher('/gps_ego_pose', Point, queue_size=10)
        self.global_wp_pub = rospy.Publisher('/current_global_waypoint', Marker, queue_size=10)
        self.start_flga_pub = rospy.Publisher('/start_flag',Bool,queue_size = 10)

        self.waypoint_sub = rospy.Subscriber('/last_target_point', Marker, self.path_callback)
        self.vel_sub = rospy.Subscriber('/vehicle/curr_v', Float32, self.ego_cb)
        self.odom_sub = rospy.Subscriber('/novatel/oem7/odom', Odometry, self.odom_cb,queue_size=20)
        self.gps_sub = rospy.Subscriber('/novatel/oem7/bestgnsspos', BESTGNSSPOS, self.bestgps_cb)

        # Variables
        self.path = Point()
        self.curr_v = 0
        self.pose = None
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.init_yaw = 0.0
        self.init_x = 0.0
        self.init_y = 0.0
        self.initial_pose = False
        self.yaw_vec = []
        self.x_pos_vec = []
        self.y_pos_vec = []

        self.pose_stamped = PoseStamped()
        self.path_vis = Path()
        self.path_vis.header.frame_id = "map"
        self.path_vis.poses = []

        self.lat = None
        self.lon = None
        self.initial_gps = None
        self.gps_data = []
        self.initialized_gps = False

        self.steer_ratio = 3

        self.test_x = 0
        self.test_y = 0

        self.curr_lat = None
        self.curr_lon = None
        self.current_waypoint_idx = 0

        self.pose_stamped = PoseStamped()
        
        self.last_process_time = rospy.Time.now()
        self.process_interval = rospy.Duration(0.01)

    def bestgps_cb(self,msg):
        self.curr_lat = msg.lat
        self.curr_lon = msg.lon
        if(self.curr_lat is not None and self.curr_lon is not None):
            gps_point = (self.curr_lat,self.curr_lon)
            self.initialize_gps(gps_point)

    def odom_cb(self, msg):
        current_time = rospy.Time.now()
        if current_time - self.last_process_time < self.process_interval:
            return
        self.pose = msg
        orientation_q = msg.pose.pose.orientation
        quaternion = (orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)
        self.roll, self.pitch, self.yaw = tf.transformations.euler_from_quaternion(quaternion)

        if not self.initial_pose:
            self.yaw_vec.append(self.yaw)
            self.x_pos_vec.append(msg.pose.pose.position.x)
            self.y_pos_vec.append(msg.pose.pose.position.y)
            
            if len(self.yaw_vec) > 50 and len(self.x_pos_vec) > 50 and len(self.y_pos_vec) > 50:
                yaw_value = np.array(self.yaw_vec)
                yaw_unwarp = np.unwrap(yaw_value)
                self.init_yaw = -np.median(yaw_unwarp)
                self.init_x = np.median(self.x_pos_vec)
                self.init_y = np.median(self.y_pos_vec)

                rospy.loginfo(f"Initial yaw: {self.init_yaw}")
                rospy.loginfo(f"Initial x_pos: {self.init_x}")
                rospy.loginfo(f"Initial y_pos: {self.init_y}")

                self.initial_pose = True

        else:
            x_global = msg.pose.pose.position.x
            y_global = msg.pose.pose.position.y
            x_local, y_local = self.local_coordinates(x_global, y_global)
            transformed_quaternion = self.transform_orientation(quaternion)
            self.publish_local_pose(x_local, y_local, transformed_quaternion)

        self.last_process_time = current_time

    def publish_local_pose(self, x_local, y_local, quaternion):
        self.pose_stamped = PoseStamped()
        self.pose_stamped.header.stamp = rospy.Time.now()
        self.pose_stamped.header.frame_id = "map" 
        self.pose_stamped.pose.position.x = x_local
        self.pose_stamped.pose.position.y = y_local
        self.pose_stamped.pose.position.z = 0.0  

        self.pose_stamped.pose.orientation.x = quaternion[0]
        self.pose_stamped.pose.orientation.y = quaternion[1]
        self.pose_stamped.pose.orientation.z = quaternion[2]
        self.pose_stamped.pose.orientation.w = quaternion[3]
        self.path_vis.poses.append(self.pose_stamped)

        self.pose_pub.publish(self.pose_stamped)
        self.pub_path.publish(self.path_vis)

        # rospy.loginfo(f"Published Local Pose: X={x_local:.2f}, Y={y_local:.2f}")

    def local_coordinates(self, x, y):
        if not self.initial_pose:               
            rospy.logwarn("Initial position not set yet!")
            return None, None

        dx = x - self.init_x
        dy = y - self.init_y

        x_local = dx * np.cos(-self.init_yaw) - dy * np.sin(-self.init_yaw)
        y_local = dx * np.sin(-self.init_yaw) + dy * np.cos(-self.init_yaw)

        return x_local, y_local

    def transform_orientation(self, quaternion):
        yaw_adjustment = -self.init_yaw
        adjustment_quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw_adjustment)

        transformed_quaternion = tf.transformations.quaternion_multiply(adjustment_quaternion, quaternion)
        return transformed_quaternion

    def initialize_gps(self, gps_point):
        if len(self.gps_data) < 50:
            self.gps_data.append(gps_point)
            rospy.loginfo(f"Collecting GPS data for initialization: {len(self.gps_data)}/50")

        if len(self.gps_data) == 50 and not self.initialized_gps:
            avg_lat = np.median([data[0] for data in self.gps_data])
            avg_lon = np.median([data[1] for data in self.gps_data])
            self.initial_gps = (avg_lat, avg_lon)
            self.initialized_gps = True
            rospy.loginfo(f"Initial GPS position set: {self.initial_gps}")

    def gps_to_enu(self, gps_point):
        east_offset = geodesic((self.initial_gps[0], gps_point[1]), self.initial_gps).meters
        north_offset = geodesic((gps_point[0], self.initial_gps[1]), self.initial_gps).meters

        if gps_point[1] < self.initial_gps[1]:
            east_offset = -east_offset
        if gps_point[0] < self.initial_gps[0]:
            north_offset = -north_offset

        return east_offset, north_offset

    def get_yaw_from_pose(self, pose):
        orientation_q = pose.pose.pose.orientation
        quaternion = (
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w,
        )
        _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)
        return yaw

    def enu_to_odom(self, enu_point, current_pose):
        if not self.initial_pose:
            rospy.logwarn("Initial position not set yet!")
            return 0, 0

        x_enu, y_enu = enu_point

        gps_offset = np.array([0.0,0.0])  # x: 1m, y: 0m

        offset_x = gps_offset[0] * np.cos(-self.init_yaw) - gps_offset[1] * np.sin(-self.init_yaw)
        offset_y = gps_offset[0] * np.sin(-self.init_yaw) + gps_offset[1] * np.cos(-self.init_yaw)

        x_enu_adjusted = x_enu - offset_x
        y_enu_adjusted = y_enu - offset_y

        x_local = x_enu_adjusted * np.cos(-self.init_yaw) - y_enu_adjusted * np.sin(-self.init_yaw)
        y_local = x_enu_adjusted * np.sin(-self.init_yaw) + y_enu_adjusted * np.cos(-self.init_yaw)

        return x_local, y_local

    def transform_waypoints_to_odom(self):
        if not self.initialized_gps:
            rospy.logwarn("GPS not initialized yet!")
            return []
        curr_pose = Odometry()
        curr_pose.pose = self.pose_stamped
        transformed_waypoints = []
        for waypoint in self.waypoint:
            enu_point = self.gps_to_enu(waypoint)
            odom_point = self.enu_to_odom(enu_point, curr_pose)
            transformed_waypoints.append(odom_point)
        
        self.publish_waypoints_as_marker_allpoint(transformed_waypoints)
        return transformed_waypoints

    def transform_gps_waypoints_to_odom(self,waypoint):
        if not self.initialized_gps:
            rospy.logwarn("GPS not initialized yet!")
            return []
        curr_pose = Odometry()
        curr_pose.pose = self.pose_stamped
        transformed_waypoints = []
        # for waypoint in waypoints:
        enu_point = self.gps_to_enu(waypoint)
        odom_point = self.enu_to_odom(enu_point, curr_pose)
        transformed_waypoints.append(odom_point)

        self.publish_waypoints_as_marker(odom_point)
        return transformed_waypoints

    def publish_waypoints_as_marker(self, waypoint):
        marker = Marker()
        marker.header.frame_id = "map" 
        marker.header.stamp = rospy.Time.now()
        marker.ns = "waypoint"
        marker.id = 0
        marker.type = Marker.POINTS  
        marker.action = Marker.ADD
        marker.scale.x = 5  
        marker.scale.y = 5
        marker.color.a = 1.0  
        marker.color.r = 1.0  
        marker.color.g = 0.0  
        marker.color.b = 0.0  

        point = Point()
        point.x = waypoint[0]
        point.y = waypoint[1]
        point.z = 0
        marker.points.append(point)
        self.marker_gps_pub.publish(marker)

    def publish_waypoints_as_marker_allpoint(self, waypoints):
        marker = Marker()
        marker.header.frame_id = "map" 
        marker.header.stamp = rospy.Time.now()
        marker.ns = "waypoint"
        marker.id = 1
        marker.type = Marker.POINTS  
        marker.action = Marker.ADD
        marker.scale.x = 3  
        marker.scale.y = 3
        marker.color.a = 1.0  
        marker.color.r = 0.0  
        marker.color.g = 0.0  
        marker.color.b = 1.0  

        for wp in waypoints:
            point = Point()
            point.x = wp[0]
            point.y = wp[1]
            point.z = 0
            marker.points.append(point)

        self.marker_all_gps_pub.publish(marker)

    def transform_point(self, local_point):
        curr_pose = Odometry()
        curr_pose.pose = self.pose_stamped
        yaw = self.get_yaw_from_pose(curr_pose)

        current_x = self.pose_stamped.pose.position.x
        current_y = self.pose_stamped.pose.position.y

        x_local = local_point.x
        y_local = local_point.y 

        x_global = x_local * math.cos(yaw) - y_local * math.sin(yaw) + current_x
        y_global = x_local * math.sin(yaw) + y_local * math.cos(yaw) + current_y

        transformed_point = Point()
        transformed_point.x = x_global
        transformed_point.y = y_global
        transformed_point.z = 0 

        return transformed_point

    def path_callback(self, msg):

        temp = Point()
        temp.x = msg.pose.position.x 
        temp.y = msg.pose.position.y 

        transformed_point = self.transform_point(temp)

        path_marker = Marker()
        path_marker.header.frame_id = "map"
        path_marker.header.stamp = rospy.Time.now()
        path_marker.ns = "path_marker"
        path_marker.id = 0  
        path_marker.type = Marker.SPHERE
        path_marker.action = Marker.ADD
        path_marker.pose.position.x = transformed_point.x
        path_marker.pose.position.y = transformed_point.y
        path_marker.pose.position.z = transformed_point.z
        path_marker.pose.orientation.w = 1.0
        path_marker.scale.x = 1
        path_marker.scale.y = 1
        path_marker.scale.z = 1
        path_marker.color.a = 1.0
        path_marker.color.r = 0.0
        path_marker.color.g = 0.0
        path_marker.color.b = 1.0

        self.marker_pub.publish(path_marker)

    def ego_cb(self, msg):
        self.curr_v = msg.data

    def is_near_corner(self, x, y, threshold=5):
        condition = self.transform_gps_waypoints_to_odom(CORNERS)
        current_pos = (x, y)
        for idx, corner in enumerate(condition):
            distance = math.sqrt((corner[0]-current_pos[0])**2 + (corner[1]-current_pos[1])**2)
            if distance < threshold:
                return idx
        return None

    def find_waypoint(self,allwaypoint,curr_gps):

        if not allwaypoint:
            rospy.logwarn("No waypoints available!")
            return

        curr_x , curr_y = curr_gps[0]
        curr_pos = np.array([curr_x,curr_y])
        remaining_idx = list(range(len(allwaypoint)))
        remain_wp = [allwaypoint[i] for i in remaining_idx]
        
        distance = [np.linalg.norm(np.array([wp[0], wp[1]]) - curr_pos)
                    for wp in remain_wp
        ]
        closest_idx = np.argmin(distance)
        closest_dist = distance[closest_idx]
        
        th = 7  

        if(closest_dist < th):
            close_idx_wp = self.waypoint[closest_idx]
            
            wp = self.transform_gps_waypoints_to_odom(close_idx_wp)
            wp_x , wp_y = wp[0]
            marker = Marker()
            marker.header.frame_id = "map" 
            marker.header.stamp = rospy.Time.now()
            marker.ns = "waypoints"
            marker.id = 0 
            marker.type = Marker.SPHERE 
            marker.action = Marker.ADD

            marker.pose.position.x = wp_x
            marker.pose.position.y = wp_y
            marker.pose.position.z = closest_idx

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

            self.global_wp_pub.publish(marker)
            self.waypoint.pop(closest_idx)
            rospy.loginfo("Pop global waypoint")

    def run_setting_loop(self):
        rate = rospy.Rate(20)  # 20 Hz
        while not rospy.is_shutdown():
            local_x = self.pose_stamped.pose.position.x 
            local_y = self.pose_stamped.pose.position.y

            if self.curr_lat is not None and self.curr_lon is not None and self.initial_pose and self.initial_gps:
                self.start_flga_pub.publish(True)
                allwaypoint = self.transform_waypoints_to_odom()
                way = [self.curr_lat,self.curr_lon]
                curr_gps = self.transform_gps_waypoints_to_odom(way)
                
                curr_x , curr_y = curr_gps[0]
                point = Point()

                point.x = curr_x
                point.y = curr_y

                self.gps_pos.publish(point)
                self.find_waypoint(allwaypoint,curr_gps)

            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('path_tracker_node')

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
        (37.386214, 126.652223),
        ## 1
        (37.386988, 126.648747),
        (37.386944, 126.648705),
        (37.386901, 126.648664),
        (37.386859, 126.648623),
        (37.386818, 126.648584),
        (37.386776, 126.648543),
        (37.386764, 126.648531),
        ## 2
        (37.385401, 126.648788),
        (37.385360, 126.648856),
        (37.385322, 126.648918),
        (37.385288, 126.648973),
        (37.385254, 126.649029),
        (37.385224, 126.649079),
        ## 3
        (37.384455, 126.650347),
        (37.384404, 126.650428),
        (37.384354, 126.650510),
        (37.384303, 126.650592),
        (37.384268, 126.650650),
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
        (37.382681, 126.653555),
        ## 7
        (37.383053, 126.653905),
        (37.383119, 126.653969),
        (37.383170, 126.654019),
        (37.383241, 126.654088),
        (37.383289, 126.654134),
        ## 8
        (37.383612, 126.654445),
        (37.383697, 126.654525),
        (37.383768, 126.654591),
        (37.383836, 126.654656),
        (37.383875, 126.654693)
    ]


    tracker = PathTracker(waypoints)

    rospy.loginfo("Waiting for GPS initialization...")
    while not tracker.initialized_gps:
        rospy.sleep(0.1)  

    rospy.loginfo("GPS initialized. Starting control loop.")

    try:
        tracker.run_setting_loop()
    except rospy.ROSInterruptException:
        pass
