@echo off
cd /d C:\dev\ros2_jazzy
call C:\dev\ros2_jazzy\local_setup.bat
set PYTHONPATH=C:\dev\ros2_jazzy\Lib\site-packages;%PYTHONPATH%
set ROS_DISTRO=jazzy
set RMW_IMPLEMENTATION=rmw_fastrtps_cpp
set PATH=C:\dev\ros2_jazzy\.pixi\envs\default;C:\dev\ros2_jazzy\.pixi\envs\default\Library\bin;%PATH%
cd /d C:\Users\zlfen\Desktop\ball_tracer_pc
C:\dev\ros2_jazzy\.pixi\envs\default\python.exe -u src/win_time_sync.py
