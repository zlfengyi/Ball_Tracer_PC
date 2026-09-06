param(
    [ValidateSet('session', 'sweep')]
    [string]$Plan = 'session',
    [string]$Session = 'tracker_20260904_063357',
    [int]$Repeats = 0,
    [int]$SweepCount = 30,
    [double[]]$SweepX = @(0.75, 1.20),
    [double[]]$SweepZGround = @(1.10, 1.50),
    [switch]$Execute
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot '.venv_ros2\Scripts\python.exe'
$script = Join-Path $PSScriptRoot 'capture_v04_ht_replay.py'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}

if ($Execute) {
    $expectedPcIp = '192.168.50.230'
    $localIps = @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop |
        Select-Object -ExpandProperty IPAddress)
    if ($localIps -notcontains $expectedPcIp) {
        throw "18F PC address $expectedPcIp is not up. Local IPv4: $($localIps -join ', ')"
    }

    $rosRoot = 'C:\dev\ros2_jazzy'
    $setupPs1 = Join-Path $rosRoot 'local_setup.ps1'
    $venvActivate = Join-Path $repoRoot '.venv_ros2\Scripts\Activate.ps1'
    $rosBinRoot = Join-Path $rosRoot '.pixi\envs\default'
    $rosBinLib = Join-Path $rosBinRoot 'Library\bin'
    $cycloneXml = Join-Path $repoRoot 'ros2\cyclonedds_18.xml'
    $mvsImport = 'C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport'

    foreach ($required in @($setupPs1, $venvActivate, $cycloneXml, $mvsImport)) {
        if (-not (Test-Path -LiteralPath $required)) {
            throw "Required runtime path not found: $required"
        }
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
    $env:MVS_MVIMPORT_DIR = $mvsImport
    $env:BALL_TRACER_SOFTWARE_ROTATE_180 = '0'
}

Set-Location $repoRoot
$arguments = @('--plan', $Plan, '--session', $Session)
if ($Repeats -gt 0) {
    $arguments += @('--repeats', $Repeats)
}
if ($Plan -eq 'sweep') {
    $arguments += @('--sweep-count', $SweepCount,
        '--sweep-x', $SweepX[0], $SweepX[1],
        '--sweep-z-ground', $SweepZGround[0], $SweepZGround[1])
}
if ($Execute) {
    $arguments += '--execute'
}
& $python $script @arguments
exit $LASTEXITCODE
