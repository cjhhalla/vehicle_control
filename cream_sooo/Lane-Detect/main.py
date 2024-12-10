#!/usr/bin/env python3
# main.py
import rospy
from camera import Camera
import argparse

def run():
    parser = argparse.ArgumentParser(description="Lane Detection Node")
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with visualization windows')
    args = parser.parse_args()

    # ROS 노드 초기화 시 로그 레벨 설정
    log_level = rospy.DEBUG if args.debug else rospy.INFO
    rospy.init_node("camera_sensor_subscriber", log_level=log_level)
    camera = Camera(debug=args.debug)
    rospy.spin()

if __name__ == '__main__':
    run()
