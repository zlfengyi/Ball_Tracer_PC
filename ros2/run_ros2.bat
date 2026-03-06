@echo off
:: ROS2 环境启动脚本
:: 用法：run_ros2.bat <python_script>
:: 示例：run_ros2.bat test_publisher.py

cd /d C:\dev\ros2_jazzy
call C:\dev\ros2_jazzy\local_setup.bat
set PYTHONPATH=C:\dev\ros2_jazzy\Lib\site-packages;%PYTHONPATH%
set ROS_DISTRO=jazzy
set RMW_IMPLEMENTATION=rmw_fastrtps_cpp
set PATH=C:\dev\ros2_jazzy\.pixi\envs\default;C:\dev\ros2_jazzy\.pixi\envs\default\Library\bin;%PATH%

cd /d %~dp0
C:\dev\ros2_jazzy\.pixi\envs\default\python.exe %*
