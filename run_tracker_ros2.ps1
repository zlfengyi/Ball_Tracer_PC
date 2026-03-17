param(
    [double]$Duration = 120,
    [switch]$NoVideo,
    [ValidateSet('auto', 'direct', 'bridge', 'off')]
    [string]$Ros2Mode = 'direct',
    [switch]$ProbeOnly
)

$launcher = Join-Path $PSScriptRoot "run_tracker.ps1"
$params = @{
    Duration = $Duration
    Ros2Mode = $Ros2Mode
    PreferredEnv = "ros2"
}
if ($NoVideo) {
    $params.NoVideo = $true
}
if ($ProbeOnly) {
    $params.ProbeOnly = $true
}

& $launcher @params
exit $LASTEXITCODE
