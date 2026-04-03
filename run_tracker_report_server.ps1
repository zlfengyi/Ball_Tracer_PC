param(
    [string]$ListenHost = "127.0.0.1",
    [int]$Port = 8765,
    [string]$TrackerOutputDir = "",
    [string]$PoeConfig = ""
)

$ErrorActionPreference = "Stop"

function Test-TcpPortAvailable {
    param(
        [string]$BindHost,
        [int]$Port
    )

    try {
        $listeners = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
    } catch {
        return $true
    }

    foreach ($endpoint in $listeners) {
        if ($endpoint.Port -ne $Port) {
            continue
        }
        if ($endpoint.Address.ToString() -in @($BindHost, "0.0.0.0", "::", "::1")) {
            return $false
        }
    }
    return $true
}

function Get-TcpPortOwnerText {
    param(
        [int]$Port
    )

    try {
        $line = netstat -ano -p tcp | Select-String -Pattern (":{0}\s" -f $Port) | Select-Object -First 1
        if (-not $line) {
            return $null
        }
        $parts = ($line.ToString() -replace "\s+", " ").Trim().Split(" ")
        $pid = $parts[-1]
        if (-not $pid) {
            return $null
        }
        $proc = Get-Process -Id ([int]$pid) -ErrorAction SilentlyContinue
        if ($proc) {
            return ("{0} (PID={1})" -f $proc.ProcessName, $proc.Id)
        }
        return ("PID={0}" -f $pid)
    } catch {
        return $null
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$candidates = @(
    (Join-Path $scriptDir ".venv_ros2\Scripts\python.exe"),
    (Join-Path $scriptDir ".venv_clean\Scripts\python.exe")
)

$pythonExe = $null
foreach ($candidate in $candidates) {
    if (Test-Path $candidate) {
        $pythonExe = $candidate
        break
    }
}
if (-not $pythonExe) {
    $pythonExe = "python"
}

$serverScript = Join-Path $scriptDir "test_src\tracker_report_server.py"
if (-not (Test-Path $serverScript)) {
    throw "Missing report server script: $serverScript"
}

$requestedPort = $Port
$userSpecifiedPort = $PSBoundParameters.ContainsKey("Port")
if (-not (Test-TcpPortAvailable -BindHost $ListenHost -Port $Port)) {
    $ownerText = Get-TcpPortOwnerText -Port $Port
    if ($userSpecifiedPort) {
        if ($ownerText) {
            throw ("Port {0} on {1} is already in use by {2}. Try e.g. .\run_tracker_report_server.ps1 -Port 8876" -f $Port, $ListenHost, $ownerText)
        }
        throw ("Port {0} on {1} is already in use. Try e.g. .\run_tracker_report_server.ps1 -Port 8876" -f $Port, $ListenHost)
    }

    $candidatePort = $Port + 1
    $maxPort = $Port + 50
    while ($candidatePort -le $maxPort) {
        if (Test-TcpPortAvailable -BindHost $ListenHost -Port $candidatePort) {
            if ($ownerText) {
                Write-Host ("Port {0} is already in use by {1}; switching to {2}." -f $Port, $ownerText, $candidatePort) -ForegroundColor Yellow
            } else {
                Write-Host ("Port {0} is already in use; switching to {1}." -f $Port, $candidatePort) -ForegroundColor Yellow
            }
            $Port = $candidatePort
            break
        }
        $candidatePort += 1
    }
    if ($Port -eq $requestedPort) {
        throw ("No available TCP port found in range {0}-{1} on {2}" -f $requestedPort, $maxPort, $ListenHost)
    }
}

$argsList = @($serverScript, "--host", $ListenHost, "--port", "$Port")
if ($TrackerOutputDir) {
    $argsList += @("--tracker-output-dir", $TrackerOutputDir)
}
if ($PoeConfig) {
    $argsList += @("--poe-config", $PoeConfig)
}

Write-Host ("Using python: {0}" -f $pythonExe)
Write-Host ("Serving tracker reports at http://{0}:{1}" -f $ListenHost, $Port)
& $pythonExe @argsList
