# 新版只增加在线逐抛 PC/RK 对时；其余启动、环境和楼层选择始终复用当前 run_tracker。
$launcher = Join-Path $PSScriptRoot "run_tracker.ps1"

& $launcher @args -EnableRkTimeAlign
exit $LASTEXITCODE
