param(
    [double]$Duration = 300,
    [switch]$NoVideo,
    [switch]$NoLog,
    [switch]$FullResVideo,
    [ValidateSet('auto', 'direct', 'bridge', 'off')]
    [string]$Ros2Mode = 'direct',
    [ValidateSet('auto', 'ros2', 'clean')]
    [string]$PreferredEnv = 'auto',
    [int]$RosDomainId = 2,
    [string]$CameraConfig = '',
    [string]$CalibrationConfig = '',
    [ValidateRange(0, 1)]
    [int]$CameraReverse180 = 1,
    [ValidateSet(16, 18)]
    [int]$Floor = 16,
    [switch]$ProbeOnly
)

if ($FullResVideo -and $NoVideo) {
    throw "-FullResVideo and -NoVideo are mutually exclusive."
}

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $utf8NoBom
[Console]::OutputEncoding = $utf8NoBom
$OutputEncoding = $utf8NoBom
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
try {
    & chcp.com 65001 > $null
} catch {
}

$env:BALL_TRACER_CAMERA_REVERSE_180 = $CameraReverse180.ToString()
$env:BALL_TRACER_CAMERA_REVERSE_X = $CameraReverse180.ToString()
$env:BALL_TRACER_CAMERA_REVERSE_Y = $CameraReverse180.ToString()
$env:BALL_TRACER_SOFTWARE_ROTATE_180 = '0'

$script = Join-Path $PSScriptRoot "src\run_tracker.py"
$configDir = Join-Path $PSScriptRoot "src\config"
if ([string]::IsNullOrWhiteSpace($CameraConfig)) {
    $CameraConfig = Join-Path $configDir "camera.json"
}
if ([string]::IsNullOrWhiteSpace($CalibrationConfig)) {
    $CalibrationConfig = Join-Path $configDir "four_camera_calib.json"
}
$cleanPython = Join-Path $PSScriptRoot ".venv_clean\Scripts\python.exe"
$ros2Python = Join-Path $PSScriptRoot ".venv_ros2\Scripts\python.exe"
$ros2Activate = Join-Path $PSScriptRoot ".venv_ros2\Scripts\Activate.ps1"
$ros2Setup = 'C:\dev\ros2_jazzy\local_setup.ps1'
$ros2SitePackages = 'C:\dev\ros2_jazzy\Lib\site-packages'
$ros2PixiRoot = 'C:\dev\ros2_jazzy\.pixi\envs\default'
$ros2PixiLibraryBin = Join-Path $ros2PixiRoot 'Library\bin'
$trackerPcIp = if ($Floor -eq 18) { '192.168.31.78' } else { '192.168.50.230' }
$armRkIp = if ($Floor -eq 18) { '192.168.31.23' } else { '192.168.50.17' }
$cycloneConfigName = if ($Floor -eq 18) { 'cyclonedds_18.xml' } else { 'cyclonedds.xml' }
$cycloneXml = Join-Path $PSScriptRoot "ros2\$cycloneConfigName"
$mvsMvImport = 'C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport'

$env:BALL_TRACER_PC_IP = $trackerPcIp
$env:BALL_TRACER_ARM_RK_IP = $armRkIp
$env:BALL_TRACER_CHASSIS_RK_IP = '192.168.50.143'

if (-not (Test-Path Env:MVS_MVIMPORT_DIR)) {
    $env:MVS_MVIMPORT_DIR = $mvsMvImport
}
if (-not (Test-Path $env:MVS_MVIMPORT_DIR)) {
    throw "MVS MvImport directory not found: $($env:MVS_MVIMPORT_DIR)"
}

function Add-UniqueEnvPrefix {
    param(
        [string]$Name,
        [string]$Value
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return
    }

    $current = (Get-Item -Path "Env:$Name" -ErrorAction SilentlyContinue).Value
    $separator = [System.IO.Path]::PathSeparator
    $entries = @()
    if (-not [string]::IsNullOrWhiteSpace($current)) {
        $entries = $current -split [Regex]::Escape([string]$separator)
    }
    if ($entries -contains $Value) {
        return
    }

    $newValue = if ([string]::IsNullOrWhiteSpace($current)) {
        $Value
    } else {
        "$Value$separator$current"
    }
    Set-Item -Path "Env:$Name" -Value $newValue
}

function Enable-TrackerRos2Networking {
    if (-not (Test-Path $ros2Setup)) {
        throw "ROS 2 setup script not found: $ros2Setup"
    }
    if (-not (Test-Path $cycloneXml)) {
        throw "CycloneDDS config not found: $cycloneXml"
    }

    Add-UniqueEnvPrefix -Name 'PATH' -Value $ros2PixiLibraryBin
    Add-UniqueEnvPrefix -Name 'PATH' -Value $ros2PixiRoot
    . $ros2Setup
    Add-UniqueEnvPrefix -Name 'PYTHONPATH' -Value $ros2SitePackages
    $env:ROS_DISTRO = 'jazzy'
    $env:ROS_DOMAIN_ID = $RosDomainId.ToString()
    $env:RMW_IMPLEMENTATION = 'rmw_cyclonedds_cpp'
    $env:CYCLONEDDS_URI = "file://" + ((Resolve-Path $cycloneXml).Path -replace '\\', '/')
    Remove-Item Env:FASTRTPS_DEFAULT_PROFILES_FILE -ErrorAction SilentlyContinue
    Remove-Item Env:FASTDDS_DEFAULT_PROFILES_FILE -ErrorAction SilentlyContinue

    Write-Host "ROS_DOMAIN_ID: $($env:ROS_DOMAIN_ID)"
    Write-Host "ROS2 middleware: $($env:RMW_IMPLEMENTATION)"
    Write-Host "CycloneDDS config: $($env:CYCLONEDDS_URI)"
}

function Get-Ros2EnvInfo {
    param(
        [string]$PythonPath
    )

    if (-not (Test-Path $PythonPath)) {
        return $null
    }

    $probeScript = @'
import importlib.util
import json

result = {
    "cuda": False,
    "has_tensorrt": False,
}

try:
    import torch
    result["cuda"] = bool(torch.cuda.is_available())
    result["torch_version"] = getattr(torch, "__version__", "")
except Exception as exc:
    result["torch_error"] = repr(exc)

result["has_tensorrt"] = importlib.util.find_spec("tensorrt") is not None
print(json.dumps(result))
'@

    try {
        $output = $probeScript | & $PythonPath -
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($output)) {
            return $null
        }
        return $output | ConvertFrom-Json
    } catch {
        return $null
    }
}

function Select-TrackerEnv {
    param(
        [string]$RequestedEnv
    )

    $ros2Info = Get-Ros2EnvInfo -PythonPath $ros2Python
    $hasRos2 = (Test-Path $ros2Python)
    $hasClean = (Test-Path $cleanPython)
    $ros2Ready = $hasRos2 -and $null -ne $ros2Info -and ($ros2Info.cuda -or $ros2Info.has_tensorrt)

    switch ($RequestedEnv) {
        'ros2' {
            if (-not $hasRos2) {
                throw "ROS2 environment not found: $ros2Python"
            }
            return [pscustomobject]@{
                Name = 'ros2'
                Python = $ros2Python
                Activate = $ros2Activate
                Info = $ros2Info
            }
        }
        'clean' {
            if (-not $hasClean) {
                throw "Clean environment not found: $cleanPython"
            }
            return [pscustomobject]@{
                Name = 'clean'
                Python = $cleanPython
                Activate = $null
                Info = $null
            }
        }
        default {
            if ($ros2Ready) {
                return [pscustomobject]@{
                    Name = 'ros2'
                    Python = $ros2Python
                    Activate = $ros2Activate
                    Info = $ros2Info
                }
            }
            if ($hasClean) {
                return [pscustomobject]@{
                    Name = 'clean'
                    Python = $cleanPython
                    Activate = $null
                    Info = $null
                }
            }
            if ($hasRos2) {
                return [pscustomobject]@{
                    Name = 'ros2'
                    Python = $ros2Python
                    Activate = $ros2Activate
                    Info = $ros2Info
                }
            }
            throw "No tracker Python environment found under .venv_ros2 or .venv_clean"
        }
    }
}

$selection = Select-TrackerEnv -RequestedEnv $PreferredEnv

if ($selection.Name -eq 'ros2') {
    $cuda = if ($selection.Info) { [bool]$selection.Info.cuda } else { $false }
    $hasTensorRT = if ($selection.Info) { [bool]$selection.Info.has_tensorrt } else { $false }
    Write-Host ("Selected tracker env: ros2 (cuda={0}, tensorrt={1})" -f $cuda, $hasTensorRT)
} else {
    Write-Host "Selected tracker env: clean (CPU fallback)"
}

Write-Host "Camera config: $CameraConfig"
Write-Host "Calibration config: $CalibrationConfig"
Write-Host "Tracker floor: ${Floor}F"
Write-Host "Tracker PC IP: $trackerPcIp"
Write-Host "CycloneDDS config: $cycloneXml"

if ($ProbeOnly) {
    exit 0
}

if ($selection.Name -eq 'ros2') {
    if (-not (Test-Path $selection.Activate)) {
        throw "ROS2 activate script not found: $($selection.Activate)"
    }
    . $selection.Activate
    if ($Ros2Mode -ne 'off') {
        Enable-TrackerRos2Networking
    }
}

Write-Host "BALL_TRACER_CAMERA_REVERSE_180=$($env:BALL_TRACER_CAMERA_REVERSE_180)"
Write-Host "BALL_TRACER_CAMERA_REVERSE_X=$($env:BALL_TRACER_CAMERA_REVERSE_X)"
Write-Host "BALL_TRACER_CAMERA_REVERSE_Y=$($env:BALL_TRACER_CAMERA_REVERSE_Y)"
Write-Host "BALL_TRACER_SOFTWARE_ROTATE_180=$($env:BALL_TRACER_SOFTWARE_ROTATE_180)"

if (-not $ProbeOnly) {
    $saveVideoText = if ($NoVideo) { "off" } elseif ($FullResVideo) { "full_res_per_camera" } else { "on" }
    $saveLogText = if ($NoLog) { "off" } else { "on" }
    Write-Host "Starting tracker (duration=${Duration}s, save_video=$saveVideoText, save_log=$saveLogText)."
    Write-Host "Press Ctrl+C to stop; run_tracker.py will finish shutdown cleanly before exit."
}

$args = @(
    $script,
    "--duration", $Duration.ToString(),
    "--ros2-mode", $Ros2Mode,
    "--camera-config", $CameraConfig,
    "--calib-config", $CalibrationConfig
)
if ($NoVideo) {
    $args += "--no-video"
}
if ($NoLog) {
    $args += "--no-log"
}
if ($FullResVideo) {
    $args += "--full-res-video"
}

& $selection.Python @args
exit $LASTEXITCODE
