@echo off
setlocal

set "ROS2_ROOT=C:\dev\ros2_jazzy"
set "CYCLONEDDS_XML=%~dp0cyclonedds.xml"

cd /d %ROS2_ROOT%
call "%ROS2_ROOT%\local_setup.bat"
set "PYTHONPATH=%ROS2_ROOT%\Lib\site-packages;%PYTHONPATH%"
set "ROS_DISTRO=jazzy"
if not defined BALL_TRACER_ROS_DOMAIN_ID (
    if not defined ROS_DOMAIN_ID (
        set "ROS_DOMAIN_ID=2"
    )
) else (
    set "ROS_DOMAIN_ID=%BALL_TRACER_ROS_DOMAIN_ID%"
)
set "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
set "CYCLONEDDS_URI=file://%CYCLONEDDS_XML:\=/%"
set FASTRTPS_DEFAULT_PROFILES_FILE=
set FASTDDS_DEFAULT_PROFILES_FILE=
set "PATH=%ROS2_ROOT%\.pixi\envs\default;%ROS2_ROOT%\.pixi\envs\default\Library\bin;%PATH%"

cd /d %~dp0
"%ROS2_ROOT%\.pixi\envs\default\python.exe" %*
