param(
    [switch]$Execute
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot '.venv_ros2\Scripts\python.exe'
$script = Join-Path $PSScriptRoot 'capture_v04_sweet_spot_map.py'

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
    $setupBat = Join-Path $rosRoot 'local_setup.bat'
    $rosSitePackages = Join-Path $rosRoot 'Lib\site-packages'
    $rosBinRoot = Join-Path $rosRoot '.pixi\envs\default'
    $rosBinLib = Join-Path $rosBinRoot 'Library\bin'
    $cycloneXml = Join-Path $repoRoot 'ros2\cyclonedds_18.xml'
    $mvsImport = 'C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport'

    foreach ($required in @($setupBat, $cycloneXml, $mvsImport)) {
        if (-not (Test-Path -LiteralPath $required)) {
            throw "Required runtime path not found: $required"
        }
    }

    $envDump = cmd /c "call `"$setupBat`" >nul && set"
    foreach ($line in $envDump) {
        if ($line -match '^(.*?)=(.*)$') {
            Set-Item -Path ('Env:' + $matches[1]) -Value $matches[2]
        }
    }

    $env:PYTHONPATH = "$rosSitePackages;$env:PYTHONPATH"
    $env:PATH = "$rosBinRoot;$rosBinLib;$env:PATH"
    $env:ROS_DISTRO = 'jazzy'
    $env:ROS_DOMAIN_ID = '0'
    $env:RMW_IMPLEMENTATION = 'rmw_cyclonedds_cpp'
    $env:CYCLONEDDS_URI = ([System.Uri]$cycloneXml).AbsoluteUri
    Remove-Item Env:FASTRTPS_DEFAULT_PROFILES_FILE -ErrorAction SilentlyContinue
    Remove-Item Env:FASTDDS_DEFAULT_PROFILES_FILE -ErrorAction SilentlyContinue
    $env:MVS_MVIMPORT_DIR = $mvsImport
    $env:BALL_TRACER_SOFTWARE_ROTATE_180 = '0'
}

Set-Location $repoRoot
$arguments = @()
if ($Execute) {
    $arguments += '--execute'
}
& $python $script @arguments
exit $LASTEXITCODE
