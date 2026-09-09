# Generic launcher: run any python script inside the 18F PC ROS2 env (same setup as run_arm_swing_link_comp.ps1).
# Usage: ArmCalibration/run_ros2_py.ps1 <script.py> [args...]   (from D:/Ball_Tracer_PC)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot '.venv_ros2\Scripts\python.exe'
if ($args.Count -lt 1) { throw 'usage: run_ros2_py.ps1 <script.py> [args...]' }
$script = $args[0]
$rest = @($args | Select-Object -Skip 1)
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
& $python $script @rest
exit $LASTEXITCODE
