@echo off
:: 启动 car_loc_bridge ROS2 节点
:: 接收 run_tracker UDP 数据，发布到 /pc_car_loc topic
call "%~dp0run_ros2.bat" "%~dp0car_loc_bridge.py"
