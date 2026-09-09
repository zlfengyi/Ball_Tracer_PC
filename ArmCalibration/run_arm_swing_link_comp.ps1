# Operator launcher for ArmCalibration\arm_swing_link_comp.py (18F PC ROS2 env, same as run_v04_ht_replay.ps1).
# Usage:  .\ArmCalibration\run_arm_swing_link_comp.ps1 --dry-run --mode wiggle
#         .\ArmCalibration\run_arm_swing_link_comp.ps1 --mode wiggle --wiggle-deg 5,10,20 --wiggle-cycles 3
#         .\ArmCalibration\run_arm_swing_link_comp.ps1 --speeds 3,4,5.5,7 --repeats 2
# WARNING: this moves the arm (wiggle = READY reparks, default mode = real swings).
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot '.venv_ros2\Scripts\python.exe'
$script = Join-Path $PSScriptRoot 'arm_swing_link_comp.py'
if (-not (Test-Path -LiteralPath $python)) { throw "Python environment not found: $python" }

$expectedPcIp = '192.168.50.230'
$localIps = @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop | Select-Object -ExpandProperty IPAddress)
if ($localIps -notcontains $expectedPcIp) {
    throw "18F PC address $expectedPcIp is not up. Local IPv4: $($localIps -join ', ')"
}

$rosRoot = 'C:\dev\ros2_jazzy'
$setupPs1 = Join-Path $rosRoot 'local_setup.ps1'
$venvActivate = Join-Path $repoRoot '.venv_ros2\Scripts\Activate.ps1'
$rosBinRoot = Join-Path $rosRoot '.pixi\envs\default'
$rosBinLib = Join-Path $rosBinRoot 'Library\bin'
$cycloneXml = Join-Path $repoRoot 'ros2\cyclonedds_18.xml'
foreach ($required in @($setupPs1, $venvActivate, $cycloneXml)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required runtime path not found: $required" }
}

. $venvActivate
$env:PATH = "$rosBinRoot;$rosBinLib;$env:PATH"
. $setupPs1
$env:PYTHONPATH = "$repoRoot;$env:PYTHONPATH"
$env:ROS_DISTRO = 'jazzy'
$env:ROS_DOMAIN_ID = '2'
$env:RMW_IMPLEMENTATION = 'rmw_cyclonedds_cpp'
$env:CYCLONEDDS_URI = 'file://' + ($cycloneXml -replace '\\', '/')
Remove-Item Env:FASTRTPS_DEFAULT_PROFILES_FILE -ErrorAction SilentlyContinue
Remove-Item Env:FASTDDS_DEFAULT_PROFILES_FILE -ErrorAction SilentlyContinue
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

Set-Location $repoRoot
& $python $script @args
exit $LASTEXITCODE
