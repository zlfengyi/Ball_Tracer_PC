@echo off
:: 启动 predict_hit_bridge ROS2 节点
:: 接收 run_tracker UDP 数据，发布到 /predict_hit_pos topic
call "%~dp0run_ros2.bat" "%~dp0predict_hit_bridge.py"
