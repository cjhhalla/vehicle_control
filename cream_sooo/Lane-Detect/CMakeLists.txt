cmake_minimum_required(VERSION 3.0.2)
project(lane_detect)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  cv_bridge
  tf2_ros
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs cv_bridge tf2_ros geometry_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
