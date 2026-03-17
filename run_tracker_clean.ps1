param(
    [double]$Duration = 120,
    [switch]$NoVideo,
    [ValidateSet('auto', 'direct', 'bridge', 'off')]
    [string]$Ros2Mode = 'auto',
    [switch]$ForceClean,
    [switch]$ProbeOnly
)

$launcher = Join-Path $PSScriptRoot "run_tracker.ps1"
$preferredEnv = if ($ForceClean) { 'clean' } else { 'auto' }
$params = @{
    Duration = $Duration
    Ros2Mode = $Ros2Mode
    PreferredEnv = $preferredEnv
}
if ($NoVideo) {
    $params.NoVideo = $true
}
if ($ProbeOnly) {
    $params.ProbeOnly = $true
}

& $launcher @params
exit $LASTEXITCODE
