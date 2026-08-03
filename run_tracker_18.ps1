$launcher = Join-Path $PSScriptRoot "run_tracker.ps1"
$cameraConfig = Join-Path $PSScriptRoot "src\config\camera_18.json"
$calibrationConfig = Join-Path $PSScriptRoot "src\config\four_camera_calib_18.json"

& $launcher @args `
    -Floor 18 `
    -CameraConfig $cameraConfig `
    -CalibrationConfig $calibrationConfig `
    -CameraReverse180 0
exit $LASTEXITCODE
