# 兼容用的薄壳：18F 的相机/标定/REVERSE_180/网段现在都在 run_tracker.ps1 的
# floor profile 里，直接跑 .\run_tracker.ps1 会按本机 IP 自动认出 18F。
# 这里只留一个显式入口，给不在 18F 网上还想指定 18F 配置的场合用。
$launcher = Join-Path $PSScriptRoot "run_tracker.ps1"

& $launcher @args -Floor 18
exit $LASTEXITCODE
