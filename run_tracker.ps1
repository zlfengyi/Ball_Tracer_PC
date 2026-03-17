param(
    [double]$Duration = 120,
    [switch]$NoVideo,
    [ValidateSet('auto', 'direct', 'bridge', 'off')]
    [string]$Ros2Mode = 'auto',
    [ValidateSet('auto', 'ros2', 'clean')]
    [string]$PreferredEnv = 'auto',
    [switch]$ProbeOnly
)

$script = Join-Path $PSScriptRoot "src\run_tracker.py"
$cleanPython = Join-Path $PSScriptRoot ".venv_clean\Scripts\python.exe"
$ros2Python = Join-Path $PSScriptRoot ".venv_ros2\Scripts\python.exe"
$ros2Activate = Join-Path $PSScriptRoot ".venv_ros2\Scripts\Activate.ps1"

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

if ($ProbeOnly) {
    return
}

if ($selection.Name -eq 'ros2') {
    if (-not (Test-Path $selection.Activate)) {
        throw "ROS2 activate script not found: $($selection.Activate)"
    }
    . $selection.Activate
}

$args = @($script, "--duration", $Duration.ToString(), "--ros2-mode", $Ros2Mode)
if ($NoVideo) {
    $args += "--no-video"
}

& $selection.Python @args
exit $LASTEXITCODE
