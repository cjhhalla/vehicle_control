run_setting.py - initialize local frame and processing ros topic
run_control.py - control input generation
sim_data.bag - simulation for test code
visualize/vis.py - text and point visualization
rviz/local_frame.rviz - local vehicle frame rviz
rviz/visual.rviz - global vehicle frame rviz




Start Line

Have to run novatel (GNSS)
./can.sh (CAN connect)

/start_code/can2morai.py 

77 -> autonomous mode
1001 -> reset processing

roslaunch setting.launch


