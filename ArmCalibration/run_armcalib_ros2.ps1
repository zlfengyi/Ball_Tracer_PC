param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Script,

    [ValidateSet('auto', 'ros2', 'clean')]
    [string]$PreferredEnv = 'auto',

    [switch]$ProbeOnly,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$rosRoot = "C:\dev\ros2_jazzy"
$setupBat = Join-Path $rosRoot "local_setup.bat"
$rosSitePackages = Join-Path $rosRoot "Lib\site-packages"
$rosBinRoot = Join-Path $rosRoot ".pixi\envs\default"
$rosBinLib = Join-Path $rosRoot ".pixi\envs\default\Library\bin"
$fastDdsProfile = Join-Path $repoRoot "ros2\fastdds.xml"
$cleanPython = Join-Path $repoRoot ".venv_clean\Scripts\python.exe"
$ros2Python = Join-Path $repoRoot ".venv_ros2\Scripts\python.exe"
$scriptPath = Join-Path $PSScriptRoot $Script

if (-not (Test-Path $setupBat)) {
    throw "ROS2 setup script not found: $setupBat"
}
if (-not (Test-Path $scriptPath)) {
    throw "Target script not found: $scriptPath"
}

function Get-EnvInfo {
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

function Select-ArmCalibEnv {
    param(
        [string]$RequestedEnv
    )

    $ros2Info = Get-EnvInfo -PythonPath $ros2Python
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
                Info = $null
            }
        }
        default {
            if ($ros2Ready) {
                return [pscustomobject]@{
                    Name = 'ros2'
                    Python = $ros2Python
                    Info = $ros2Info
                }
            }
            if ($hasClean) {
                return [pscustomobject]@{
                    Name = 'clean'
                    Python = $cleanPython
                    Info = $null
                }
            }
            if ($hasRos2) {
                return [pscustomobject]@{
                    Name = 'ros2'
                    Python = $ros2Python
                    Info = $ros2Info
                }
            }
            throw "No ArmCalibration Python environment found under .venv_ros2 or .venv_clean"
        }
    }
}

$selection = Select-ArmCalibEnv -RequestedEnv $PreferredEnv
if ($selection.Name -eq 'ros2') {
    $cuda = if ($selection.Info) { [bool]$selection.Info.cuda } else { $false }
    $hasTensorRT = if ($selection.Info) { [bool]$selection.Info.has_tensorrt } else { $false }
    Write-Host ("Selected ArmCalibration env: ros2 (cuda={0}, tensorrt={1})" -f $cuda, $hasTensorRT)
} else {
    Write-Host "Selected ArmCalibration env: clean (CPU fallback)"
}

if ($ProbeOnly) {
    return
}

$envDump = cmd /c "call `"$setupBat`" >nul && set"
foreach ($line in $envDump) {
    if ($line -match "^(.*?)=(.*)$") {
        Set-Item -Path ("Env:" + $matches[1]) -Value $matches[2]
    }
}

$env:PYTHONPATH = "$rosSitePackages;$env:PYTHONPATH"
$env:ROS_DISTRO = "jazzy"
$env:RMW_IMPLEMENTATION = "rmw_fastrtps_cpp"
$env:FASTRTPS_DEFAULT_PROFILES_FILE = $fastDdsProfile
$env:PATH = "$rosBinRoot;$rosBinLib;$env:PATH"

Set-Location $repoRoot
& $selection.Python $scriptPath @ScriptArgs
exit $LASTEXITCODE
