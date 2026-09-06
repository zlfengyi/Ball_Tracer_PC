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
    # 曝光临时覆盖（不改配置文件）。0 / 负数 = 用配置文件里的值。
    # 18F 相机模拟 Gain 上限只有 12.78dB 且出厂就顶格，-GainDb 基本没用；
    # -DigitalShift 可调数字增益，但不会改善传感器信噪比，并会增加高光截断风险。
    # 曝光越长运动拖影越大，回球段最吃亏（像面速度 ~1400px/s，是来球段的 2~3 倍）：
    # 9000μs 拖影 12.6px、和球直径（13~18px）同量级，2026-08-09 场击球后连丢 4 帧、
    # 回球统计整场为空。当前默认 9000μs 是光线过暗下的被动选择（当晚亮度一路下滑），
    # **回球段数据在这一档本来就不可靠**，看 summary.detection_shape_gate 确认是不是形状门在拦。
    # 光线一好转就调短：-ExposureUs 6000（拖影 8.4px）或 4000（5.6px，回球段最稳）
    [double]$ExposureUs = 0,
    [double]$GainDb = -1,
    [Nullable[double]]$DigitalShift = $null,
    [string]$CalibrationConfig = '',
    # 车型必选，没有默认值：两台车的车载 AprilTag 布局完全不同，选错了车定位会静默
    # 偏几十 cm、yaw 还可能翻 180°（拟合照样收敛，只有 car_loc 重投影会从 ~2px 涨到
    # 40px+，要等出报告才看得见）。不传就在这里弹提示让你选，别猜。
    # HelpMessage 保持 ASCII：本文件无 BOM，PS 5.1 按 ANSI 读，中文字面量会变乱码
    [Parameter(Mandatory = $true, HelpMessage = "Pick the car: v03 (old car) or v04 (new car)")]
    [ValidateSet('v03', 'v04')]
    [string]$Car,
    # 只在要跑非标布局文件时用；给了就覆盖 -Car 选出来的那份
    [string]$CarConfig = '',
    [switch]$EnableRkTimeAlign,
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

$script = Join-Path $PSScriptRoot "src\run_tracker.py"
$configDir = Join-Path $PSScriptRoot "src\config"
$cleanPython = Join-Path $PSScriptRoot ".venv_clean\Scripts\python.exe"
$ros2Python = Join-Path $PSScriptRoot ".venv_ros2\Scripts\python.exe"
$ros2Activate = Join-Path $PSScriptRoot ".venv_ros2\Scripts\Activate.ps1"
$ros2Setup = 'C:\dev\ros2_jazzy\local_setup.ps1'
$ros2SitePackages = 'C:\dev\ros2_jazzy\Lib\site-packages'
$ros2PixiRoot = 'C:\dev\ros2_jazzy\.pixi\envs\default'
$ros2PixiLibraryBin = Join-Path $ros2PixiRoot 'Library\bin'
$mvsMvImport = 'C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport'

# 18F 是唯一在用的场地（2026-09-06 用户定案，16F 已停用、相机不会再变）：相机
# DB026xxx + action 广播触发、标定 four_camera_calib_18.json、DDS cyclonedds_18.xml、
# PC 走 Wi-Fi 的 192.168.50.230。没有第二套了，直接写死；别再引入 -Floor 这种
# 「一个开关选一整套」的分支，它上一次的作用就是让人猜错默认值。
# 车的 RK IP 不写在这里：一台车一个 IP（车上静态配置），跟着 -Car 走，见 $carRkIps。
$trackerPcIp      = '192.168.50.230'
$cycloneXmlName   = 'cyclonedds_18.xml'
$cameraConfigName = 'camera_18.json'
$calibConfigName  = 'four_camera_calib_18.json'

function Get-LocalIPv4Addresses {
    try {
        return @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop |
            Select-Object -ExpandProperty IPAddress)
    } catch {
        return @([System.Net.Dns]::GetHostAddresses([System.Net.Dns]::GetHostName()) |
            Where-Object { $_.AddressFamily -eq 'InterNetwork' } |
            ForEach-Object { $_.IPAddressToString })
    }
}

# 一台车只有一个 RK IP（车上静态配置，换路由器/换楼层都不变，2026-08-18 定）。
# 以前配置里的 ArmRkIp/ChassisRkIp 两个字段其实就是两台车各自的 IP，概念已废。
$carRkIps = @{
    'v03' = '192.168.50.143'
    'v04' = '192.168.50.68'
}

$localIps = Get-LocalIPv4Addresses
$carRkIp = $carRkIps[$Car]
$cycloneXml = Join-Path $PSScriptRoot ("ros2\" + $cycloneXmlName)

# 拦住 "does not match an available interface"：CycloneDDS 里的 NetworkInterface
# 是写死的本机地址，网卡没起来就只会在建 node 时炸，报错还不提是哪里选错了。
if ($localIps -notcontains $trackerPcIp) {
    $ipMsg = "Tracker PC is expected at $trackerPcIp, but that address is not up. local: $($localIps -join ', ')"
    if ($Ros2Mode -eq 'off' -or $ProbeOnly) {
        Write-Warning "$ipMsg (not creating a DDS node, continuing)"
    } else {
        throw "$ipMsg`nCycloneDDS would fail to bind ($cycloneXmlName). Fix the network, or use -Ros2Mode off to run offline."
    }
}

if ($Car -ne 'v04') {
    Write-Warning "18F normally runs v04, but -Car $Car was given. Check the AprilTag layout is really $Car."
}

if ([string]::IsNullOrWhiteSpace($CameraConfig)) {
    $CameraConfig = Join-Path $configDir $cameraConfigName
}
if ([string]::IsNullOrWhiteSpace($CalibrationConfig)) {
    $CalibrationConfig = Join-Path $configDir $calibConfigName
}

# 相机 ReverseX/Y 写死在 src/ball_grabber.py（18F 正装 = False），不再走环境变量：
# 那个默认值曾经是 16F 的 True，不设变量直接开相机就会把四台相机翻 180° 并留在相机里。
$env:BALL_TRACER_SOFTWARE_ROTATE_180 = '0'

$env:BALL_TRACER_PC_IP = $trackerPcIp
$env:BALL_TRACER_RK_CAR_IP = $carRkIp

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

Write-Host "Tracker site: 18F ($trackerPcIp)"
Write-Host "Camera config: $CameraConfig"
Write-Host "Calibration config: $CalibrationConfig"
Write-Host "Car: $Car$(if (-not [string]::IsNullOrWhiteSpace($CarConfig)) { " (overridden by -CarConfig $CarConfig)" })"
Write-Host "Car RK IP: $carRkIp"
Write-Host "Tracker PC IP: $trackerPcIp"
Write-Host "CycloneDDS config: $cycloneXml"
if ($EnableRkTimeAlign) {
    Write-Host "Online PC/RK per-throw time alignment: enabled"
}

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
    "--calib-config", $CalibrationConfig,
    "--car", $Car
)
if (-not [string]::IsNullOrWhiteSpace($CarConfig)) {
    $args += @("--car-config", $CarConfig)
}
if ($NoVideo) {
    $args += "--no-video"
}
if ($NoLog) {
    $args += "--no-log"
}
if ($FullResVideo) {
    $args += "--full-res-video"
}
if ($ExposureUs -gt 0) {
    $args += @("--exposure-us", $ExposureUs.ToString())
}
if ($GainDb -ge 0) {
    $args += @("--gain-db", $GainDb.ToString())
}
if ($null -ne $DigitalShift) {
    $args += @("--digital-shift", $DigitalShift.ToString())
}
if ($EnableRkTimeAlign) {
    $args += "--online-time-align"
}

& $selection.Python @args
exit $LASTEXITCODE
