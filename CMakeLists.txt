cmake_minimum_required(VERSION 3.0.2)
project(vehicle_control)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  start_code/run_setting.py
  start_code/can2morai.py
  start_code/run_control.py
  visualize/vis.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
