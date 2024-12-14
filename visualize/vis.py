#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3
import math
from jsk_rviz_plugins.msg import OverlayText


class RVizVisualization:
    def __init__(self):
        # ROS Publisher
        self.point_marker_pub = rospy.Publisher('/rviz_point_marker', Marker, queue_size=10)
        self.heading_marker_pub = rospy.Publisher('/rviz_heading_marker', Marker, queue_size=10)
        self.text_marker_pub = rospy.Publisher('/rviz_overlay_text_marker', OverlayText, queue_size=10)
        
        # ROS Subscriber
        rospy.Subscriber('/last_target_point', Marker, self.point_callback)
        rospy.Subscriber('/target_actuator', Vector3, self.actuator_callback)
        rospy.Subscriber('/global_odom_frame_point', Marker, self.global_callback)
        rospy.Subscriber('/vehicle/steering_angle', Float32, self.steer_callback)
        rospy.Subscriber('/vehicle/velocity_RL', Float32, self.rl_callback)
        rospy.Subscriber('/vehicle/velocity_RR', Float32, self.rr_callback)
        # Variables to store received data
        self.current_point = Point()
        self.global_current_point = Point()
        self.target_accel = 0.0
        self.target_steer = 0.0 
        self.target_waypoint = 0
        self.curr_v = 0

        self.steer = 0
        self.rr_v = 0
        self.rl_v = 0

    def steer_callback(self,msg):
        self.steer = msg.data

    def rl_callback(self,msg):
        self.rl_v = msg.data

    def rr_callback(self,msg):
        self.rr_v = msg.data

    def global_callback(self,msg):
        self.global_current_point.x = msg.pose.position.x
        self.global_current_point.y = msg.pose.position.y

    def point_callback(self, msg):
        self.current_point.x = msg.pose.position.x 
        self.current_point.y = msg.pose.position.y 

    def actuator_callback(self, msg):
        self.target_accel = msg.x
        self.target_steer = msg.y / 12
        self.target_waypoint = msg.z

    def publish_markers(self):
        # Publish point marker

        # Publish heading marker
        self.publish_heading_marker()
        self.curr_v = (self.rl_v + self.rr_v)/7.2
        self.publish_text_marker()
        # Publish text markers
        # if self.target_waypoint == 0:
        #     self.publish_text_marker("GPS MODE", 0, 4, 3, marker_id=1)
        #     self.publish_global_point_marker()
        # elif self.target_waypoint == 1:
        #     self.publish_text_marker("VISION MODE", 0, 4, 3, marker_id=1)
        #     self.publish_point_marker()
        # self.publish_text_marker(f"Actuator [%]: {self.target_accel: f}",0, 4, 4, marker_id = 2)
        # self.publish_text_marker(f"Steering angle [deg]: {self.target_steer:.2f}", 0, 4, 5, marker_id=3)
        # self.publish_text_marker(f"Steering wheel angle [deg]: {self.target_steer * 12:.2f}", 0, 4, 6, marker_id=4)
        # self.publish_text_marker(f"Real Steering wheel angle [deg]: {self.steer:.2f}", 0, 4, 7, marker_id=5)
        # self.publish_text_marker(f"current velocity [m/s]: {self.curr_v:.2f}", 0, 4, 8, marker_id=6)
        

    def publish_point_marker(self):
        point_marker = Marker()
        point_marker.header.frame_id = "ego_car"
        point_marker.header.stamp = rospy.Time.now()
        point_marker.ns = "point_marker"
        point_marker.id = 0
        point_marker.type = Marker.SPHERE
        point_marker.action = Marker.ADD

        # Position and scale
        point_marker.pose.position = self.current_point
        point_marker.pose.orientation.x = 0.0
        point_marker.pose.orientation.y = 0.0
        point_marker.pose.orientation.z = 0.0
        point_marker.pose.orientation.w = 1.0
        point_marker.scale.x = 1
        point_marker.scale.y = 1
        point_marker.scale.z = 1

        # Color
        point_marker.color.a = 1.0  # Fully opaque
        point_marker.color.r = 1.0
        point_marker.color.g = 0.0
        point_marker.color.b = 0.0

        # Publish the marker
        self.point_marker_pub.publish(point_marker)

    def publish_global_point_marker(self):
        point_marker = Marker()
        point_marker.header.frame_id = "ego_car"
        point_marker.header.stamp = rospy.Time.now()
        point_marker.ns = "point_marker"
        point_marker.id = 0
        point_marker.type = Marker.SPHERE
        point_marker.action = Marker.ADD

        # Position and scale
        point_marker.pose.position = self.global_current_point
        point_marker.pose.orientation.x = 0.0
        point_marker.pose.orientation.y = 0.0
        point_marker.pose.orientation.z = 0.0
        point_marker.pose.orientation.w = 1.0
        point_marker.scale.x = 1
        point_marker.scale.y = 1
        point_marker.scale.z = 1

        # Color
        point_marker.color.a = 1.0  # Fully opaque
        point_marker.color.r = 0.0
        point_marker.color.g = 0.0
        point_marker.color.b = 1.0

        # Publish the marker
        self.point_marker_pub.publish(point_marker)

    def publish_heading_marker(self):
        heading_marker = Marker()
        heading_marker.header.frame_id = "ego_car"
        heading_marker.header.stamp = rospy.Time.now()
        heading_marker.ns = "heading_marker"
        heading_marker.id = 1
        heading_marker.type = Marker.ARROW
        heading_marker.action = Marker.ADD

        # Arrow properties
        arrow_length = 6  # Length of the arrow (can be adjusted)
        
        # Calculate end point based on self.target_steer
        # Assuming self.target_steer is in degrees, convert to radians
        steer_radians = math.radians(self.target_steer)
        end_x = arrow_length * math.cos(steer_radians)
        end_y = arrow_length * math.sin(steer_radians)
        end_z = 0.0  # Keep in the same plane (z=0)

        # Define start and end points of the arrow
        start_x = 0.0
        start_y = 0.0
        start_z = 0.0

        # Set points
        heading_marker.points.append(Point(start_x, start_y, start_z))  # Starting point (origin)
        heading_marker.points.append(Point(end_x, end_y, end_z))  # End point (calculated based on steer angle)

        # Scale and color
        heading_marker.scale.x = 0.25  # Shaft diameter
        heading_marker.scale.y = 0.5  # Arrowhead diameter
        heading_marker.scale.z = 0.5  # Arrowhead length

        heading_marker.color.a = 1.0  # Fully opaque
        heading_marker.color.r = 0.0
        heading_marker.color.g = 1.0
        heading_marker.color.b = 0.0

        # Publish the marker
        self.heading_marker_pub.publish(heading_marker)

    # def publish_text_marker(self, text, position_x, position_y, position_z, marker_id):
    #     text_marker = Marker()
    #     text_marker.header.frame_id = "ego_car"
    #     text_marker.header.stamp = rospy.Time.now()
    #     text_marker.ns = "text_marker"
    #     text_marker.id = marker_id
    #     text_marker.type = Marker.TEXT_VIEW_FACING  # Text type marker
    #     text_marker.action = Marker.ADD

    #     # Position of the text
    #     text_marker.pose.position.x = position_x
    #     text_marker.pose.position.y = position_y
    #     text_marker.pose.position.z = position_z + 1.0  # Slightly above the point

    #     text_marker.pose.orientation.x = 0.0
    #     text_marker.pose.orientation.y = 0.0
    #     text_marker.pose.orientation.z = 0.0
    #     text_marker.pose.orientation.w = 1.0

    #     # Text scale (size)
    #     text_marker.scale.z = 0.5  # Font size

    #     # Text color
    #     text_marker.color.a = 1.0  # Fully opaque
    #     text_marker.color.r = 1.0
    #     text_marker.color.g = 1.0
    #     text_marker.color.b = 1.0

    #     # Text content
    #     text_marker.text = text

    #     # Publish the marker
    #     self.text_marker_pub.publish(text_marker)

    def publish_text_marker(self):
        overlay_text = OverlayText()
        overlay_text.action = OverlayText.ADD
        overlay_text.width = 450  # 오버레이 텍스트 너비
        overlay_text.height = 180  # 오버레이 텍스트 높이
        overlay_text.left = 10   # x 위치 조정
        overlay_text.top = 200  # y 위치 조정
        overlay_text.text_size = 14  # 글씨 크기
        overlay_text.line_width = 2  # 글씨 외곽선 두께
        overlay_text.font = "DejaVu Sansa Mono"
        # 텍스트 내용

        text_content = (
            f"MODE: {'GPS MODE' if self.target_waypoint == 0 else 'VISION MODE'}\n"
            f"Actuator [%]: {self.target_accel: .2f}\n"
            f"Steering angle [deg]: {self.target_steer:.2f}\n"
            f"Steering wheel angle [deg]: {self.target_steer * 12:.2f}\n"
            f"Real Steering wheel angle [deg]: {self.steer:.2f}\n"
            f"Current velocity [m/s]: {self.curr_v:.2f}"
        )
        overlay_text.text = text_content

        # 텍스트 색상 설정 (RGBA)
        overlay_text.fg_color.r = 0.0
        overlay_text.fg_color.g = 1.0
        overlay_text.fg_color.b = 0.0
        overlay_text.fg_color.a = 1.0

        # 배경 색상 설정 (RGBA)
        overlay_text.bg_color.r = 0.0
        overlay_text.bg_color.g = 0.0
        overlay_text.bg_color.b = 0.0
        overlay_text.bg_color.a = 0.5

        # 오버레이 텍스트 발행
        self.text_marker_pub.publish(overlay_text)


if __name__ == '__main__':
    rospy.init_node('rviz_visualization_node')
    rviz_visualization = RVizVisualization()

    # Set loop rate to 5Hz
    rate = rospy.Rate(5)

    try:
        while not rospy.is_shutdown():
            rviz_visualization.publish_markers()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
